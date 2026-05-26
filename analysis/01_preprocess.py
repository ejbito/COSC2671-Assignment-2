import html
import json
import re
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    BANDORI_CLEAN,
    BANDORI_EDGES,
    BANDORI_LABEL,
    BANDORI_RAW,
    COMBINED_CLEAN,
    COMBINED_EDGES,
    PROJECT_SEKAI_CLEAN,
    PROJECT_SEKAI_EDGES,
    PROJECT_SEKAI_LABEL,
    PROJECT_SEKAI_RAW,
    TABLES_DIR,
    ensure_dirs,
)


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def clean_text(value):
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = URL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def flatten_raw_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for video in data.get("videos", []):
        video_meta = {
            "videoTitle": video.get("title"),
            "videoId": video.get("videoId"),
            "channelId": video.get("channelId"),
            "channelTitle": video.get("channelTitle"),
            "videoPublishedAt": video.get("publishedAt"),
            "videoViewCount": video.get("viewCount"),
            "videoLikeCount": video.get("likeCount"),
            "videoCommentCount": video.get("commentCount"),
        }
        for comment in video.get("comments", []):
            row = dict(video_meta)
            row.update(comment)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    expected = [
        "gameLabel",
        "queryUsed",
        "videoId",
        "videoTitle",
        "channelId",
        "channelTitle",
        "commentId",
        "parentCommentId",
        "threadId",
        "isTopLevel",
        "authorDisplayName",
        "authorChannelId",
        "replyToAuthorDisplayName",
        "replyToAuthorChannelId",
        "text",
        "textOriginal",
        "publishedAt",
        "updatedAt",
        "likeCount",
        "totalReplyCount",
        "videoPublishedAt",
        "videoViewCount",
        "videoLikeCount",
        "videoCommentCount",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
    return df[expected]


def build_edges(df):
    if df.empty:
        return pd.DataFrame(columns=["gameLabel", "source", "target", "weight"])

    is_reply = df["isTopLevel"].astype(str).str.lower().isin(["false", "0"])
    edge_df = df.loc[
        is_reply
        & df["authorChannelId"].notna()
        & df["replyToAuthorChannelId"].notna()
        & (df["authorChannelId"] != df["replyToAuthorChannelId"]),
        ["gameLabel", "authorChannelId", "replyToAuthorChannelId"],
    ].copy()
    if edge_df.empty:
        return pd.DataFrame(columns=["gameLabel", "source", "target", "weight"])
    edge_df = edge_df.rename(
        columns={"authorChannelId": "source", "replyToAuthorChannelId": "target"}
    )
    return (
        edge_df.groupby(["gameLabel", "source", "target"], dropna=False)
        .size()
        .reset_index(name="weight")
        .sort_values(["gameLabel", "weight"], ascending=[True, False])
    )


def preprocess_one(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    raw = flatten_raw_json(path)
    raw_count = len(raw)
    if raw.empty:
        return raw, {"game": label, "raw_comments_replies": 0}

    df = raw.copy()
    df["textClean"] = df["text"].fillna(df["textOriginal"]).map(clean_text)

    empty_mask = df["textClean"].eq("")
    removed_empty = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()

    before_dupes = len(df)
    df = df.drop_duplicates(subset=["gameLabel", "commentId", "textClean"])
    removed_dupes = before_dupes - len(df)

    edges = build_edges(df)
    summary = {
        "game": label,
        "raw_comments_replies": raw_count,
        "removed_empty_text": removed_empty,
        "removed_duplicates": removed_dupes,
        "final_analysis_comments": len(df),
        "top_level_comments": int(df["isTopLevel"].astype(bool).sum()),
        "replies": int((~df["isTopLevel"].astype(bool)).sum()),
        "unique_users": int(df["authorChannelId"].nunique(dropna=True)),
        "final_reply_edges": len(edges),
        "videos": int(df["videoId"].nunique(dropna=True)),
    }
    return df, summary


def main():
    ensure_dirs()
    project_df, project_summary = preprocess_one(PROJECT_SEKAI_RAW, PROJECT_SEKAI_LABEL)
    bandori_df, bandori_summary = preprocess_one(BANDORI_RAW, BANDORI_LABEL)

    combined = pd.concat([project_df, bandori_df], ignore_index=True)

    project_edges = build_edges(project_df)
    bandori_edges = build_edges(bandori_df)
    combined_edges = build_edges(combined)

    project_df.to_csv(PROJECT_SEKAI_CLEAN, index=False, encoding="utf-8-sig")
    bandori_df.to_csv(BANDORI_CLEAN, index=False, encoding="utf-8-sig")
    combined.to_csv(COMBINED_CLEAN, index=False, encoding="utf-8-sig")
    project_edges.to_csv(PROJECT_SEKAI_EDGES, index=False, encoding="utf-8-sig")
    bandori_edges.to_csv(BANDORI_EDGES, index=False, encoding="utf-8-sig")
    combined_edges.to_csv(COMBINED_EDGES, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([project_summary, bandori_summary])
    summary.to_csv(TABLES_DIR / "preprocessing_summary.csv", index=False)
    summary.to_csv(TABLES_DIR / "data_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved clean comments to {COMBINED_CLEAN}")
    print(f"Saved reply edges to {COMBINED_EDGES}")


if __name__ == "__main__":
    main()
