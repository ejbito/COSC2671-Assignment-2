from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BASE_DIR


DATA_DIR = BASE_DIR / "data"
VADER_COMMENTS = DATA_DIR / "vader" / "processed" / "combined_comments_with_network.csv"
ROBERTA_COMMENTS = DATA_DIR / "roberta" / "processed" / "combined_comments_with_network.csv"
COMPARISON_DIR = DATA_DIR / "sentiment_comparison"
FIGURES_DIR = COMPARISON_DIR / "figures"
TABLES_DIR = COMPARISON_DIR / "tables"

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {
    "Positive": "#2f9e44",
    "Neutral": "#868e96",
    "Negative": "#c92a2a",
}
SENTIMENT_HATCHES = {
    "Positive": ".",
    "Neutral": "",
    "Negative": "//",
}
MODEL_LINE_STYLES = {
    "VADER": ":",
    "RoBERTa": "--",
}


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run both sentiment pipelines first:\n"
            "  python pipeline.py --vader\n"
            "  python pipeline.py --roberta"
        )


def load_model_outputs():
    require_file(VADER_COMMENTS)
    require_file(ROBERTA_COMMENTS)

    vader = pd.read_csv(VADER_COMMENTS)
    roberta = pd.read_csv(ROBERTA_COMMENTS)

    vader_cols = [
        "commentId",
        "gameLabel",
        "videoId",
        "videoTitle",
        "authorChannelId",
        "publishedAt",
        "textClean",
        "sentiment_label",
        "sentiment_score",
        "vader_compound",
        "vader_label",
    ]
    roberta_cols = [
        "commentId",
        "sentiment_label",
        "sentiment_score",
        "roberta_label",
        "roberta_confidence",
    ]

    missing_vader = [col for col in vader_cols if col not in vader.columns]
    missing_roberta = [col for col in roberta_cols if col not in roberta.columns]
    if missing_vader:
        raise ValueError(f"VADER output is missing columns: {missing_vader}")
    if missing_roberta:
        raise ValueError(f"RoBERTa output is missing columns: {missing_roberta}")

    merged = vader[vader_cols].merge(
        roberta[roberta_cols],
        on="commentId",
        how="inner",
        suffixes=("_vader", "_roberta"),
    )
    merged = merged.rename(
        columns={
            "sentiment_label_vader": "vader_label_final",
            "sentiment_score_vader": "vader_score",
            "sentiment_label_roberta": "roberta_label_final",
            "sentiment_score_roberta": "roberta_score",
        }
    )
    merged["models_agree"] = merged["vader_label_final"].eq(merged["roberta_label_final"])
    merged["publishedAt"] = pd.to_datetime(merged["publishedAt"], errors="coerce", utc=True)
    merged["month"] = merged["publishedAt"].dt.to_period("M").astype(str)
    return merged


def distribution_counts(df, model_col, model_name, group_cols=None):
    group_cols = group_cols or []
    counts = (
        df.groupby(group_cols + [model_col])
        .size()
        .reset_index(name="count")
        .rename(columns={model_col: "sentiment"})
    )
    if group_cols:
        totals = counts.groupby(group_cols)["count"].transform("sum")
    else:
        totals = counts["count"].sum()
    counts["percentage"] = (counts["count"] / totals * 100).round(2)
    counts["model"] = model_name
    return counts


def build_overall_distribution_table(df):
    table = pd.concat(
        [
            distribution_counts(df, "vader_label_final", "VADER"),
            distribution_counts(df, "roberta_label_final", "RoBERTa"),
        ],
        ignore_index=True,
    )
    table["sentiment"] = pd.Categorical(table["sentiment"], SENTIMENT_ORDER, ordered=True)
    table = table.sort_values(["model", "sentiment"])
    table.to_csv(TABLES_DIR / "overall_sentiment_distribution_by_model.csv", index=False)
    return table


def build_game_distribution_table(df):
    table = pd.concat(
        [
            distribution_counts(df, "vader_label_final", "VADER", ["gameLabel"]),
            distribution_counts(df, "roberta_label_final", "RoBERTa", ["gameLabel"]),
        ],
        ignore_index=True,
    )
    table["sentiment"] = pd.Categorical(table["sentiment"], SENTIMENT_ORDER, ordered=True)
    table = table.sort_values(["gameLabel", "model", "sentiment"])
    table.to_csv(TABLES_DIR / "sentiment_distribution_by_game_and_model.csv", index=False)
    return table


def build_agreement_tables(df):
    crosstab = pd.crosstab(
        df["vader_label_final"],
        df["roberta_label_final"],
        rownames=["VADER"],
        colnames=["RoBERTa"],
    ).reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0)
    crosstab.to_csv(TABLES_DIR / "vader_roberta_crosstab.csv")

    agreement = (
        df.groupby("gameLabel")["models_agree"]
        .agg(total_comments="count", agreements="sum", agreement_rate="mean")
        .reset_index()
    )
    agreement["agreement_rate_pct"] = (agreement["agreement_rate"] * 100).round(2)

    overall = pd.DataFrame(
        {
            "gameLabel": ["Overall"],
            "total_comments": [len(df)],
            "agreements": [int(df["models_agree"].sum())],
            "agreement_rate": [float(df["models_agree"].mean())],
            "agreement_rate_pct": [round(float(df["models_agree"].mean()) * 100, 2)],
        }
    )
    agreement = pd.concat([overall, agreement], ignore_index=True)
    agreement.to_csv(TABLES_DIR / "model_agreement_by_game.csv", index=False)
    return crosstab, agreement


def build_monthly_table(df):
    valid = df[df["month"].notna()].copy()
    rows = []
    for model_name, col in [("VADER", "vader_label_final"), ("RoBERTa", "roberta_label_final")]:
        monthly = (
            valid.groupby(["month", col])
            .size()
            .reset_index(name="count")
            .rename(columns={col: "sentiment"})
        )
        monthly["model"] = model_name
        rows.append(monthly)
    table = pd.concat(rows, ignore_index=True)
    table["sentiment"] = pd.Categorical(table["sentiment"], SENTIMENT_ORDER, ordered=True)
    table = table.sort_values(["model", "month", "sentiment"])
    table.to_csv(TABLES_DIR / "monthly_sentiment_counts_by_model.csv", index=False)
    return table


def plot_overall_distribution(table):
    plot_data = {
        model: (
            table[table["model"].eq(model)]
            .set_index("sentiment")
            .reindex(SENTIMENT_ORDER)["percentage"]
            .fillna(0)
        )
        for model in ["VADER", "RoBERTa"]
    }

    x = np.arange(len(plot_data))
    width = 0.22
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, sentiment in enumerate(SENTIMENT_ORDER):
        heights = [plot_data[model].loc[sentiment] for model in plot_data]
        bars = ax.bar(
            x + (idx - 1) * width,
            heights,
            width,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="#666666",
            linewidth=0,
        )
        for bar in bars:
            bar.set_hatch(SENTIMENT_HATCHES[sentiment])

    ax.set_xticks(x)
    ax.set_xticklabels(list(plot_data.keys()))
    ax.set_ylabel("Comments (%)")
    ax.set_title("Overall Sentiment Distribution by Model")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(
        handles=[
            Patch(facecolor=SENTIMENT_COLORS[s], edgecolor="#666666", linewidth=0, hatch=SENTIMENT_HATCHES[s], label=s)
            for s in SENTIMENT_ORDER
        ],
        title="Sentiment",
        loc="upper right",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "overall_sentiment_distribution_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_game_distribution(table):
    games = list(table["gameLabel"].dropna().unique())
    fig, axes = plt.subplots(1, len(games), figsize=(6 * max(len(games), 1), 4), sharey=True)
    if len(games) == 1:
        axes = [axes]

    for ax, game in zip(axes, games):
        sub = table[table["gameLabel"].eq(game)]
        model_data = {
            model: (
                sub[sub["model"].eq(model)]
                .set_index("sentiment")
                .reindex(SENTIMENT_ORDER)["percentage"]
                .fillna(0)
            )
            for model in ["VADER", "RoBERTa"]
        }
        x = np.arange(len(model_data))
        width = 0.22
        for idx, sentiment in enumerate(SENTIMENT_ORDER):
            heights = [model_data[model].loc[sentiment] for model in model_data]
            bars = ax.bar(
                x + (idx - 1) * width,
                heights,
                width,
                color=SENTIMENT_COLORS[sentiment],
                edgecolor="#666666",
                linewidth=0,
            )
            for bar in bars:
                bar.set_hatch(SENTIMENT_HATCHES[sentiment])
        ax.set_title(game)
        ax.set_xticks(x)
        ax.set_xticklabels(list(model_data.keys()))
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    axes[0].set_ylabel("Comments (%)")
    axes[-1].legend(
        handles=[
            Patch(facecolor=SENTIMENT_COLORS[s], edgecolor="#666666", linewidth=0, hatch=SENTIMENT_HATCHES[s], label=s)
            for s in SENTIMENT_ORDER
        ],
        title="Sentiment",
        loc="upper right",
    )
    fig.suptitle("Sentiment Distribution by Game and Model", y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "sentiment_distribution_by_game_and_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_lines(monthly_table):
    if monthly_table.empty:
        return
    pivoted = {}
    for model in ["VADER", "RoBERTa"]:
        model_table = monthly_table[monthly_table["model"].eq(model)]
        pivoted[model] = (
            model_table.pivot_table(index="month", columns="sentiment", values="count", aggfunc="sum", fill_value=0)
            .reindex(columns=SENTIMENT_ORDER, fill_value=0)
            .sort_index()
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    for ax, sentiment in zip(axes, SENTIMENT_ORDER):
        for model in ["VADER", "RoBERTa"]:
            if pivoted[model].empty:
                continue
            ax.plot(
                pivoted[model].index,
                pivoted[model][sentiment],
                marker="o",
                linewidth=2.0,
                linestyle=MODEL_LINE_STYLES[model],
                label=model,
            )
        ax.set_title(f"{sentiment} Comments by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Comments")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_sentiment_trends_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_crosstab_heatmap(crosstab):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(crosstab.values, cmap="Blues")

    ax.set_xticks(np.arange(len(SENTIMENT_ORDER)))
    ax.set_yticks(np.arange(len(SENTIMENT_ORDER)))
    ax.set_xticklabels(SENTIMENT_ORDER)
    ax.set_yticklabels(SENTIMENT_ORDER)
    ax.set_xlabel("RoBERTa label")
    ax.set_ylabel("VADER label")
    ax.set_title("VADER vs RoBERTa Label Agreement")

    max_value = crosstab.values.max() if crosstab.values.size else 0
    threshold = max_value / 2 if max_value else 0
    for i in range(len(SENTIMENT_ORDER)):
        for j in range(len(SENTIMENT_ORDER)):
            value = int(crosstab.iloc[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value:,}", ha="center", va="center", color=color)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Number of comments", rotation=90)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "vader_roberta_crosstab_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_agreement_by_game(agreement):
    plot_df = agreement.copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(plot_df["gameLabel"], plot_df["agreement_rate_pct"], color="#4263eb", linewidth=0)
    ax.set_ylabel("Agreement (%)")
    ax.set_ylim(0, 100)
    ax.set_title("VADER and RoBERTa Agreement Rate")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "model_agreement_rate_by_game.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_disagreement_examples(df, n=20):
    disagreements = df[~df["models_agree"]].copy()
    if disagreements.empty:
        disagreements.to_csv(TABLES_DIR / "strongest_model_disagreements.csv", index=False)
        return disagreements

    disagreements["vader_strength"] = disagreements["vader_compound"].abs()
    disagreements["disagreement_type"] = (
        "VADER " + disagreements["vader_label_final"] + " / RoBERTa " + disagreements["roberta_label_final"]
    )
    examples = (
        disagreements.sort_values(["vader_strength", "roberta_confidence"], ascending=[False, False])
        .head(n)
        [
            [
                "gameLabel",
                "videoTitle",
                "disagreement_type",
                "vader_compound",
                "roberta_confidence",
                "textClean",
            ]
        ]
    )
    examples.to_csv(TABLES_DIR / "strongest_model_disagreements.csv", index=False, encoding="utf-8-sig")
    return examples


def main():
    ensure_dirs()
    merged = load_model_outputs()
    merged.to_csv(TABLES_DIR / "merged_vader_roberta_comment_labels.csv", index=False, encoding="utf-8-sig")

    overall_table = build_overall_distribution_table(merged)
    game_table = build_game_distribution_table(merged)
    crosstab, agreement = build_agreement_tables(merged)
    monthly_table = build_monthly_table(merged)
    disagreement_examples = export_disagreement_examples(merged)

    plot_overall_distribution(overall_table)
    plot_game_distribution(game_table)
    plot_monthly_lines(monthly_table)
    plot_crosstab_heatmap(crosstab)
    plot_agreement_by_game(agreement)

    print(f"Compared {len(merged):,} comments with both model outputs.")
    print("\nAgreement by game:")
    print(agreement.to_string(index=False))
    print(f"\nSaved comparison tables to {TABLES_DIR}")
    print(f"Saved comparison figures to {FIGURES_DIR}")
    if not disagreement_examples.empty:
        print("\nExample disagreement types:")
        print(disagreement_examples["disagreement_type"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
