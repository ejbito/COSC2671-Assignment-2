from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COMBINED_CLEAN, FIGURES_DIR, TABLES_DIR, SENTIMENT_METHOD, ensure_dirs


CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
SENTIMENT_CACHE = CACHE_DIR / f"{SENTIMENT_METHOD}_sentiment_cache.pkl"
ROBERTA_BATCH_SIZE = int(os.getenv("ROBERTA_BATCH_SIZE", "64"))
ROBERTA_MAX_LENGTH = int(os.getenv("ROBERTA_MAX_LENGTH", "256"))

VADER_OUTPUT_COLS = [
    "sentiment_method",
    "vader_neg",
    "vader_neu",
    "vader_pos",
    "vader_compound",
    "vader_label",
    "sentiment_label",
    "sentiment_score",
]
ROBERTA_OUTPUT_COLS = [
    "sentiment_method",
    "roberta_raw_label",
    "roberta_label",
    "roberta_confidence",
    "sentiment_label",
    "sentiment_score",
]


def get_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer(), "vader"
    except LookupError:
        try:
            import nltk

            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer(), "vader"
        except Exception as exc:
            raise RuntimeError(
                "NLTK VADER is installed, but the vader_lexicon resource is missing. "
                "Run: python -m nltk.downloader vader_lexicon"
            ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "NLTK is required for sentiment analysis. Run: python -m pip install nltk"
        ) from exc


def get_cache_output_cols():
    return ROBERTA_OUTPUT_COLS if SENTIMENT_METHOD == "roberta" else VADER_OUTPUT_COLS


def get_cache_key_cols():
    return ["commentId", "textClean"]


def load_sentiment_cache():
    if not SENTIMENT_CACHE.exists():
        return pd.DataFrame(columns=get_cache_key_cols() + get_cache_output_cols())
    cache = pd.read_pickle(SENTIMENT_CACHE)
    required = set(get_cache_key_cols() + get_cache_output_cols())
    missing = required.difference(cache.columns)
    if missing:
        print(f"Ignoring incomplete sentiment cache; missing columns: {sorted(missing)}")
        return pd.DataFrame(columns=get_cache_key_cols() + get_cache_output_cols())
    return cache[get_cache_key_cols() + get_cache_output_cols()].drop_duplicates(
        subset=get_cache_key_cols(),
        keep="last",
    )


def save_sentiment_cache(cache):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_pickle(SENTIMENT_CACHE)


def merge_cache(df, cache):
    if cache.empty:
        df = df.copy()
        df["_cache_hit"] = False
        return df
    merged = df.merge(cache, on=get_cache_key_cols(), how="left")
    merged["_cache_hit"] = merged["sentiment_label"].notna()
    return merged


def apply_vader_to_uncached(df):
    vader, method = get_vader()
    scores = []
    for text in df["textClean"].fillna(""):
        scores.append(vader.polarity_scores(str(text)))

    score_df = pd.DataFrame(scores)
    df = df.copy()
    df["sentiment_method"] = method
    df["vader_neg"] = score_df["neg"]
    df["vader_neu"] = score_df["neu"]
    df["vader_pos"] = score_df["pos"]
    df["vader_compound"] = score_df["compound"]
    df["vader_label"] = df["vader_compound"].map(label_from_compound)
    df["sentiment_label"] = df["vader_label"]
    df["sentiment_score"] = df["vader_compound"]
    return df


def load_roberta():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "RoBERTa sentiment requires transformers and torch. "
            "Run: python -m pip install transformers torch"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "PyTorch is installed but failed to load a required native DLL. "
            "On Windows this usually means either the PyTorch wheel is not right for "
            "this environment, or the Microsoft Visual C++ Redistributable runtime is "
            "missing. Reinstall the CPU PyTorch wheel from the official PyTorch index "
            "inside the venv, then test with: python -c \"import torch; print(torch.__version__)\""
        ) from exc

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except OSError as exc:
        raise RuntimeError(
            "RoBERTa model loading failed. Check internet access for the first model "
            "download, and check that PyTorch loads correctly in this Python environment."
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    print(f"RoBERTa device: {'cuda:0' if device == 0 else 'cpu'}")

    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=ROBERTA_MAX_LENGTH,
        device=device,
    )


def normalise_roberta_label(label):
    label = str(label).lower()
    if "positive" in label or label == "label_2":
        return "Positive"
    if "negative" in label or label == "label_0":
        return "Negative"
    return "Neutral"


def append_to_cache(cache, new_rows):
    updated_cache = pd.concat(
        [
            cache[get_cache_key_cols() + get_cache_output_cols()],
            new_rows[get_cache_key_cols() + get_cache_output_cols()],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=get_cache_key_cols(), keep="last")
    save_sentiment_cache(updated_cache)
    return updated_cache


def apply_roberta_to_uncached(df, cache, batch_size=ROBERTA_BATCH_SIZE):
    classifier = load_roberta()
    texts = df["textClean"].fillna("").astype(str).tolist()
    labelled_batches = []
    running_cache = cache
    print(f"RoBERTa batch size: {batch_size}")
    print(f"RoBERTa max token length: {ROBERTA_MAX_LENGTH}")

    for start in tqdm(range(0, len(texts), batch_size), desc="RoBERTa sentiment"):
        batch = texts[start:start + batch_size]
        outputs = classifier(batch, batch_size=batch_size)
        out_df = pd.DataFrame(outputs)
        batch_df = df.iloc[start:start + batch_size].copy()
        batch_df["sentiment_method"] = "roberta"
        batch_df["roberta_raw_label"] = out_df["label"].to_numpy()
        batch_df["roberta_label"] = out_df["label"].map(normalise_roberta_label).to_numpy()
        batch_df["roberta_confidence"] = out_df["score"].to_numpy()
        batch_df["sentiment_label"] = batch_df["roberta_label"]
        batch_df["sentiment_score"] = batch_df["roberta_confidence"]
        labelled_batches.append(batch_df)
        running_cache = append_to_cache(
            running_cache,
            batch_df[get_cache_key_cols() + get_cache_output_cols()],
        )

    if not labelled_batches:
        return pd.DataFrame(columns=df.columns.tolist() + get_cache_output_cols())
    return pd.concat(labelled_batches, ignore_index=False)


def label_from_compound(compound):
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def apply_sentiment(df):
    df = df.copy()
    df["_row_order"] = range(len(df))
    cache = load_sentiment_cache()
    df_with_cache = merge_cache(df, cache)
    cache_hits = int(df_with_cache["_cache_hit"].sum())
    total = len(df_with_cache)
    print(f"Sentiment cache: {cache_hits:,}/{total:,} comments already labelled")

    cached_rows = df_with_cache[df_with_cache["_cache_hit"]].copy()
    uncached_base = df_with_cache.loc[
        ~df_with_cache["_cache_hit"],
        [col for col in df.columns],
    ].copy()

    if not uncached_base.empty:
        print(f"Running {SENTIMENT_METHOD} sentiment for {len(uncached_base):,} uncached comments")
        if SENTIMENT_METHOD == "roberta":
            uncached_labelled = apply_roberta_to_uncached(uncached_base, cache)
        else:
            uncached_labelled = apply_vader_to_uncached(uncached_base)
            append_to_cache(cache, uncached_labelled[get_cache_key_cols() + get_cache_output_cols()])
    else:
        uncached_labelled = pd.DataFrame(columns=df.columns.tolist() + get_cache_output_cols())

    if not cached_rows.empty:
        cached_labelled = cached_rows.drop(columns=["_cache_hit"])
    else:
        cached_labelled = pd.DataFrame(columns=df.columns.tolist() + get_cache_output_cols())

    labelled = pd.concat([cached_labelled, uncached_labelled], ignore_index=True)
    labelled = labelled.sort_values("_row_order", kind="stable").drop(columns=["_row_order"])
    print(f"Saved sentiment cache to {SENTIMENT_CACHE}")
    return labelled


def distribution_table(df, group_cols, label_col="sentiment_label"):
    counts = (
        df.groupby(group_cols + [label_col])
        .size()
        .reset_index(name="count")
        .pivot_table(index=group_cols, columns=label_col, values="count", fill_value=0)
        .reset_index()
    )
    for label in ["Positive", "Neutral", "Negative"]:
        if label not in counts.columns:
            counts[label] = 0
    counts["total"] = counts[["Positive", "Neutral", "Negative"]].sum(axis=1)
    for label in ["Positive", "Neutral", "Negative"]:
        counts[f"{label.lower()}_pct"] = (counts[label] / counts["total"] * 100).round(2)
    return counts


def plot_stacked_pct(table, index_col, title, output_path):
    labels = ["Positive", "Neutral", "Negative"]
    pct_cols = [f"{label.lower()}_pct" for label in labels]
    plot_df = table.set_index(index_col)[pct_cols].rename(
        columns=dict(zip(pct_cols, labels))
    )
    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        color=["#2ca02c", "#868e96", "#da1010"],
        figsize=(10, 5),
    )
    ax.set_ylabel("Comments (%)")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    ensure_dirs()
    df = pd.read_csv(COMBINED_CLEAN)
    df = apply_sentiment(df)
    df.to_csv(COMBINED_CLEAN, index=False, encoding="utf-8-sig")

    by_game = distribution_table(df, ["gameLabel"])
    by_video = distribution_table(df, ["gameLabel", "videoId", "videoTitle"])

    by_game.to_csv(TABLES_DIR / "sentiment_summary.csv", index=False)
    by_video.to_csv(TABLES_DIR / "sentiment_by_video.csv", index=False)

    plot_stacked_pct(
        by_game,
        "gameLabel",
        "Sentiment Distribution by Game",
        FIGURES_DIR / "sentiment_distribution_by_game.png",
    )

    top_videos = (
        by_video.assign(video_label=lambda x: x["gameLabel"] + ": " + x["videoTitle"].astype(str).str.slice(0, 45))
        .sort_values("total", ascending=False)
        .head(20)
    )
    plot_stacked_pct(
        top_videos,
        "video_label",
        "Sentiment Distribution by Video",
        FIGURES_DIR / "sentiment_distribution_by_video.png",
    )

    print(f"Sentiment method: {df['sentiment_method'].iloc[0] if len(df) else 'none'}")
    print(by_game.to_string(index=False))


if __name__ == "__main__":
    main()
