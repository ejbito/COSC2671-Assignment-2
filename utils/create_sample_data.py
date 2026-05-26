import json
import shutil
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BASE_DIR, BANDORI_RAW, PROJECT_SEKAI_RAW


SAMPLE_DIR = BASE_DIR / "sample_data"
DATA_DIR = BASE_DIR / "data"
MAX_SAMPLE_BYTES = 10 * 1024 * 1024
RAW_VIDEOS_PER_GAME = 1
COMMENT_ROWS_PER_GAME = 50
EDGE_ROWS_PER_GAME = 75
GENERIC_ROWS = 100

SUMMARY_TABLES = [
    "data_summary.csv",
    "preprocessing_summary.csv",
    "sentiment_summary.csv",
    "network_summary.csv",
    "community_summary.csv",
    "sentiment_by_centrality_group.csv",
    "top_network_communities.csv",
]


def reset_sample_dir():
    if SAMPLE_DIR.exists():
        shutil.rmtree(SAMPLE_DIR)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def sample_raw_json(input_path: Path, output_path: Path, max_videos: int = RAW_VIDEOS_PER_GAME):
    if not input_path.exists():
        print(f"Skipping missing raw file: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled = {
        key: value
        for key, value in data.items()
        if key != "videos"
    }
    videos = []
    for video in data.get("videos", [])[:max_videos]:
        sampled_video = dict(video)
        sampled_video["sample_note"] = "Comments/replies truncated for readability."
        sampled_video["comments"] = video.get("comments", [])[:50]
        videos.append(sampled_video)

    sampled["sample_note"] = (
        "Representative raw-data sample. Videos and nested comments/replies are "
        "truncated because the assignment requires a sample that shows structure, "
        "not the full dataset."
    )
    sampled["videos"] = videos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)


def sample_comments(df: pd.DataFrame, n_per_game: int = COMMENT_ROWS_PER_GAME):
    if "gameLabel" not in df.columns:
        return df.head(n_per_game * 2)

    sampled = (
        df.groupby("gameLabel", group_keys=False)
        .apply(lambda group: group.head(n_per_game))
        .reset_index(drop=True)
    )
    return sampled


def sample_edges(df: pd.DataFrame, n_per_game: int = EDGE_ROWS_PER_GAME):
    if "gameLabel" not in df.columns:
        return df.head(n_per_game * 2)

    sampled = (
        df.sort_values(["gameLabel", "weight"], ascending=[True, False])
        .groupby("gameLabel", group_keys=False)
        .head(n_per_game)
        .reset_index(drop=True)
    )
    return sampled


def sample_csv(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"Skipping missing CSV: {input_path}")
        return

    df = pd.read_csv(input_path)
    name = input_path.name

    if "comments" in name:
        sample = sample_comments(df)
    elif "edges" in name:
        sample = sample_edges(df)
    else:
        sample = df.head(GENERIC_ROWS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False, encoding="utf-8-sig")


def copy_if_exists(input_path: Path, output_path: Path):
    if input_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
    else:
        print(f"Skipping missing file: {input_path}")


def sample_method_outputs(method: str):
    method_dir = DATA_DIR / method
    processed_dir = method_dir / "processed"
    tables_dir = method_dir / "outputs" / "tables"
    output_dir = SAMPLE_DIR / method

    if not method_dir.exists():
        print(f"Skipping {method}: no output directory found.")
        return

    sample_csv(
        processed_dir / "combined_comments_clean.csv",
        output_dir / "processed" / "sample_combined_comments_clean.csv",
    )
    sample_csv(
        processed_dir / "combined_edges.csv",
        output_dir / "processed" / "sample_combined_edges.csv",
    )
    sample_csv(
        processed_dir / "combined_comments_with_network.csv",
        output_dir / "processed" / "sample_combined_comments_with_network.csv",
    )

    for table_name in SUMMARY_TABLES:
        copy_if_exists(tables_dir / table_name, output_dir / "tables" / table_name)


def copy_comparison_outputs():
    comparison_dir = DATA_DIR / "sentiment_comparison"
    if not comparison_dir.exists():
        print("Skipping sentiment comparison outputs: no comparison directory found.")
        return

    sample_csv(
        comparison_dir / "tables" / "merged_vader_roberta_comment_labels.csv",
        SAMPLE_DIR / "sentiment_comparison" / "tables" / "sample_merged_vader_roberta_comment_labels.csv",
    )

    for path in (comparison_dir / "tables").glob("*.csv"):
        if path.name == "merged_vader_roberta_comment_labels.csv":
            continue
        copy_if_exists(path, SAMPLE_DIR / "sentiment_comparison" / "tables" / path.name)


def build_manifest():
    rows = []
    total = 0
    for path in sorted(SAMPLE_DIR.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            rows.append(
                {
                    "relative_path": str(path.relative_to(SAMPLE_DIR)),
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 2),
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(SAMPLE_DIR / "sample_manifest.csv", index=False)

    if total > MAX_SAMPLE_BYTES:
        print(
            f"WARNING: sample_data is {total / (1024 * 1024):.2f} MB, "
            "which exceeds the 10 MB assignment sample limit."
        )
    else:
        print(f"Sample data size: {total / (1024 * 1024):.2f} MB")


def write_readme():
    readme = """Representative sample data for COSC2671 Assignment 2.

The assignment asks for a representative data sample no larger than 10 MB. The sample should be sufficient to show the structure of the data used for network analysis and NLP/text analysis.

This folder is therefore intentionally compact. It is not intended to reproduce the full analysis results. It shows the schema, key fields, and representative rows used by the pipeline.

Included structure:
- raw/: small raw YouTube JSON samples showing video metadata, comments, replies, author IDs, and parent/reply relationships.
- vader/: sampled processed outputs and summary tables from the VADER pipeline.
- roberta/: sampled processed outputs and summary tables from the RoBERTa pipeline, if generated.
- sentiment_comparison/: sampled VADER-vs-RoBERTa comparison tables, if generated.
- sample_manifest.csv: file list and file sizes for this sample package.

Fields demonstrating NLP/text analysis include text, textOriginal, textClean, sentiment_label, sentiment_score, and model-specific sentiment fields.

Fields demonstrating network analysis include authorChannelId, replyToAuthorChannelId, parentCommentId, isTopLevel, source, target, weight, community_id, degree/centrality fields, and PageRank where available.

The full dataset was used to generate the report outputs. The raw JSON and CSV files here are truncated for readability and submission size.

No API keys, access tokens, passwords, or private credentials are included.
"""
    (SAMPLE_DIR / "README_sample_data.txt").write_text(readme, encoding="utf-8")


def main():
    reset_sample_dir()

    sample_raw_json(PROJECT_SEKAI_RAW, SAMPLE_DIR / "raw" / "sample_project_sekai_raw.json")
    sample_raw_json(BANDORI_RAW, SAMPLE_DIR / "raw" / "sample_bandori_raw.json")

    sample_method_outputs("vader")
    sample_method_outputs("roberta")
    copy_comparison_outputs()

    write_readme()
    build_manifest()

    print(f"Sample data written to: {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
