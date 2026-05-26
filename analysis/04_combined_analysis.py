from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COMBINED_WITH_NETWORK, FIGURES_DIR, TABLES_DIR, ensure_dirs


STOPWORDS = sorted(ENGLISH_STOP_WORDS)


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
    counts["comments"] = counts[["Positive", "Neutral", "Negative"]].sum(axis=1)
    for label in ["Positive", "Neutral", "Negative"]:
        counts[f"{label.lower()}_pct"] = (counts[label] / counts["comments"] * 100).round(2)
    return counts


def plot_stacked_pct(table, label_col, title, output_path):
    pct_cols = ["positive_pct", "neutral_pct", "negative_pct"]
    plot_df = table.set_index(label_col)[pct_cols].rename(
        columns={
            "positive_pct": "Positive",
            "neutral_pct": "Neutral",
            "negative_pct": "Negative",
        }
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


def top_tfidf_terms(texts, top_n=8):
    cleaned = [str(text) for text in texts if isinstance(text, str) and text.strip()]
    if len(cleaned) < 3:
        return ""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_']+\b",
    )
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return ""
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)
    return ", ".join(term for term, _ in ranked[:top_n])


def build_community_summary(df):
    valid = df[df["community_id"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["community_id"] = valid["community_id"].astype(int)

    sentiment = distribution_table(valid, ["gameLabel", "community_id"])
    users = (
        valid.groupby(["gameLabel", "community_id"])["authorChannelId"]
        .nunique()
        .reset_index(name="users")
    )
    terms = (
        valid.groupby(["gameLabel", "community_id"])["textClean"]
        .apply(top_tfidf_terms)
        .reset_index(name="top_terms")
    )
    central_users = (
        valid.sort_values(["gameLabel", "community_id", "pagerank"], ascending=[True, True, False])
        .groupby(["gameLabel", "community_id"])["authorDisplayName"]
        .apply(lambda s: ", ".join(pd.Series(s.dropna().astype(str).unique()).head(5)))
        .reset_index(name="top_central_users")
    )

    summary = sentiment.merge(users, on=["gameLabel", "community_id"], how="left")
    summary = summary.merge(terms, on=["gameLabel", "community_id"], how="left")
    summary = summary.merge(central_users, on=["gameLabel", "community_id"], how="left")
    return summary.rename(columns={"gameLabel": "game"})


def main():
    ensure_dirs()
    df = pd.read_csv(COMBINED_WITH_NETWORK)

    community_summary = build_community_summary(df)
    if not community_summary.empty:
        community_summary.to_csv(TABLES_DIR / "community_summary.csv", index=False)
        top_communities = community_summary.sort_values(["game", "users"], ascending=[True, False]).groupby("game").head(8)
        top_communities = top_communities.assign(
            community_label=lambda x: x["game"] + " C" + x["community_id"].astype(str)
        )
        plot_stacked_pct(
            top_communities,
            "community_label",
            "Sentiment by Reply Network Community",
            FIGURES_DIR / "sentiment_by_community.png",
        )

    if "centrality_group" in df.columns:
        centrality_sentiment = distribution_table(
            df[df["centrality_group"].notna()].copy(),
            ["gameLabel", "centrality_group"],
        )
        centrality_sentiment.to_csv(TABLES_DIR / "sentiment_by_centrality_group.csv", index=False)
        centrality_sentiment = centrality_sentiment.assign(
            group_label=lambda x: x["gameLabel"] + ": " + x["centrality_group"]
        )
        plot_stacked_pct(
            centrality_sentiment,
            "group_label",
            "Sentiment by Centrality Group",
            FIGURES_DIR / "central_vs_peripheral_sentiment.png",
        )

    volume = (
        df.groupby(["gameLabel", "isTopLevel"])
        .size()
        .reset_index(name="comments")
        .assign(comment_type=lambda x: x["isTopLevel"].map(lambda v: "Top-level" if str(v).lower() in ["true", "1"] else "Reply"))
    )
    volume_pivot = volume.pivot_table(index="gameLabel", columns="comment_type", values="comments", fill_value=0)
    ax = volume_pivot.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#228be6", "#f08c00"])
    ax.set_ylabel("Comments")
    ax.set_xlabel("")
    ax.set_title("Collected Comment Volume by Game")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "data_collection_summary.png", dpi=200)
    plt.close()

    print("Combined outputs written to tables and figures directories.")
    if not community_summary.empty:
        print(community_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
