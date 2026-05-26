import os
import json
from datetime import datetime, timezone
from typing import Iterable, Optional, Union

from src.config import (
    BANDORI_LABEL,
    BANDORI_QUERIES,
    BANDORI_RAW,
    PROJECT_SEKAI_LABEL,
    PROJECT_SEKAI_QUERIES,
    PROJECT_SEKAI_RAW,
    ensure_dirs,
)
from src.youtubeClient import youtubeClient


def _parse_utc_iso8601(ts):
    """Parse a YouTube/ISO-8601 timestamp into timezone-aware UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc)
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _to_youtube_time(ts):
    """Convert ISO datetime string/datetime to YouTube API Zulu format."""
    dt = _parse_utc_iso8601(ts)
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def _in_date_window(published_at, after_dt=None, before_dt=None):
    """Return True if timestamp is inside the optional date window."""
    dt = _parse_utc_iso8601(published_at)
    if dt is None:
        return True
    if after_dt is not None and dt < after_dt:
        return False
    if before_dt is not None and dt > before_dt:
        return False
    return True


def _get_author_channel_id(comment_snippet):
    """
    Extract a stable YouTube author channel ID when available.
    This is better for networks than display names, which can change or duplicate.
    """
    channel = comment_snippet.get("authorChannelId")
    if isinstance(channel, dict):
        return channel.get("value")
    return None


def _normalise_search_queries(search_queries: Union[str, Iterable[str]]):
    if isinstance(search_queries, str):
        return [search_queries]
    return list(search_queries)


def _fetch_all_replies(
    client,
    parent_comment_id,
    video_id,
    game_label,
    query_used,
    thread_id,
    parent_author_display_name,
    parent_author_channel_id,
    after_dt=None,
    before_dt=None,
    max_replies_per_thread=None,
):
    """
    Fetch replies to a top-level comment.

    Network use:
    - top-level comment author = parent node
    - reply author = replying node
    - edge = reply author -> top-level author
    """
    replies = []
    next_page_token = None

    while True:
        remaining = None
        if max_replies_per_thread is not None:
            remaining = max_replies_per_thread - len(replies)
            if remaining <= 0:
                break

        request_limit = 100 if remaining is None else min(100, remaining)

        response = client.comments().list(
            parentId=parent_comment_id,
            part="snippet",
            maxResults=request_limit,
            pageToken=next_page_token,
            textFormat="plainText",
        ).execute()

        for item in response.get("items", []):
            snip = item["snippet"]
            if not _in_date_window(snip.get("publishedAt"), after_dt, before_dt):
                continue

            reply_author_id = _get_author_channel_id(snip)

            replies.append({
                "gameLabel": game_label,
                "queryUsed": query_used,
                "videoId": video_id,
                "threadId": thread_id,
                "commentId": item.get("id"),
                "parentCommentId": parent_comment_id,
                "isTopLevel": False,

                # User/network fields
                "authorDisplayName": snip.get("authorDisplayName"),
                "authorChannelId": reply_author_id,
                "replyToAuthorDisplayName": parent_author_display_name,
                "replyToAuthorChannelId": parent_author_channel_id,

                # Text/NLP fields
                "text": snip.get("textDisplay"),
                "textOriginal": snip.get("textOriginal"),

                # Metadata
                "publishedAt": snip.get("publishedAt"),
                "updatedAt": snip.get("updatedAt"),
                "likeCount": snip.get("likeCount", 0),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return replies


def fetchYoutubeData(
    searchQueries,
    gameLabel,
    maxVideos=25,
    maxTopLevelCommentsPerVideo=500,
    maxRepliesPerThread=None,
    outputFile="youtubeDataDump_A2.json",
    commentPublishedAfterUtc=None,
    commentPublishedBeforeUtc=None,
    videoPublishedAfterUtc=None,
    videoPublishedBeforeUtc=None,
    minCommentsCollectedPerVideo=None,
    searchOrder="viewCount",
):
    """
    Assignment 2 YouTube collector.

    This improves the Assignment 1 collector by collecting fields needed for
    network analysis, especially:
    - commentId
    - parentCommentId
    - threadId
    - authorChannelId
    - reply relationships

    Recommended network:
    - nodes = authorChannelId values
    - directed edges = reply author -> parent comment author
    - edge weight = number of replies between the same pair
    """
    client = youtubeClient()

    search_queries = _normalise_search_queries(searchQueries)

    comment_after = _parse_utc_iso8601(commentPublishedAfterUtc)
    comment_before = _parse_utc_iso8601(commentPublishedBeforeUtc)

    videos = []
    searched_video_ids = set()

    print(f"Collecting YouTube data for: {gameLabel}")
    print(f"Queries: {search_queries}")

    for query in search_queries:
        next_search_page_token = None
        print(f"\nSearching for videos with query: '{query}'...")

        while len(videos) < maxVideos:
            remaining_needed = maxVideos - len(videos)
            search_batch_size = min(50, max(remaining_needed, 10))

            search_params = dict(
                q=query,
                part="snippet",
                type="video",
                order=searchOrder,
                maxResults=search_batch_size,
            )

            if next_search_page_token:
                search_params["pageToken"] = next_search_page_token

            if videoPublishedAfterUtc is not None:
                search_params["publishedAfter"] = _to_youtube_time(videoPublishedAfterUtc)
            if videoPublishedBeforeUtc is not None:
                search_params["publishedBefore"] = _to_youtube_time(videoPublishedBeforeUtc)

            search_response = client.search().list(**search_params).execute()
            items = search_response.get("items", [])

            if not items:
                break

            candidate_video_ids = []
            video_snippets = {}

            for item in items:
                item_id = item.get("id", {})
                video_id = item_id.get("videoId")

                if not video_id:
                    print(f"  Skipping non-video search result: {item_id}")
                    continue

                if video_id in searched_video_ids:
                    continue

                searched_video_ids.add(video_id)
                candidate_video_ids.append(video_id)
                video_snippets[video_id] = item["snippet"]

            if not candidate_video_ids:
                next_search_page_token = search_response.get("nextPageToken")
                if not next_search_page_token:
                    break
                continue

            stats_response = client.videos().list(
                id=",".join(candidate_video_ids),
                part="statistics,snippet",
            ).execute()

            video_stats = {}
            for item in stats_response.get("items", []):
                video_stats[item["id"]] = {
                    "statistics": item.get("statistics", {}),
                    "snippet": item.get("snippet", {}),
                }

            print(f"  Processing {len(candidate_video_ids)} candidate videos...")

            for video_id in candidate_video_ids:
                if len(videos) >= maxVideos:
                    break

                snippet = video_snippets[video_id]
                full_video = video_stats.get(video_id, {})
                stats = full_video.get("statistics", {})

                video = {
                    "gameLabel": gameLabel,
                    "queryUsed": query,
                    "title": snippet.get("title"),
                    "videoId": video_id,
                    "channelTitle": snippet.get("channelTitle"),
                    "channelId": snippet.get("channelId"),
                    "publishedAt": snippet.get("publishedAt"),
                    "viewCount": int(stats.get("viewCount", 0)),
                    "likeCount": int(stats.get("likeCount", 0)),
                    "commentCount": int(stats.get("commentCount", 0)),
                    "comments": [],
                }

                try:
                    top_level_fetched = 0
                    next_comment_page_token = None

                    while True:
                        remaining = None
                        if maxTopLevelCommentsPerVideo is not None:
                            remaining = maxTopLevelCommentsPerVideo - top_level_fetched
                            if remaining <= 0:
                                break

                        request_limit = 100 if remaining is None else min(100, remaining)

                        comment_response = client.commentThreads().list(
                            videoId=video_id,
                            part="snippet",
                            maxResults=request_limit,
                            pageToken=next_comment_page_token,
                            textFormat="plainText",
                            order="time",
                        ).execute()

                        reached_before_start = False

                        for thread in comment_response.get("items", []):
                            thread_id = thread.get("id")
                            thread_snippet = thread.get("snippet", {})
                            top_comment_obj = thread_snippet.get("topLevelComment", {})
                            top_comment_id = top_comment_obj.get("id")
                            top_snip = top_comment_obj.get("snippet", {})

                            top_published_at = top_snip.get("publishedAt")
                            top_dt = _parse_utc_iso8601(top_published_at)

                            if comment_after is not None and top_dt is not None and top_dt < comment_after:
                                reached_before_start = True

                            include_top = _in_date_window(top_published_at, comment_after, comment_before)

                            top_author_id = _get_author_channel_id(top_snip)

                            if include_top:
                                video["comments"].append({
                                    "gameLabel": gameLabel,
                                    "queryUsed": query,
                                    "videoId": video_id,
                                    "threadId": thread_id,
                                    "commentId": top_comment_id,
                                    "parentCommentId": None,
                                    "isTopLevel": True,

                                    # User/network fields
                                    "authorDisplayName": top_snip.get("authorDisplayName"),
                                    "authorChannelId": top_author_id,
                                    "replyToAuthorDisplayName": None,
                                    "replyToAuthorChannelId": None,

                                    # Text/NLP fields
                                    "text": top_snip.get("textDisplay"),
                                    "textOriginal": top_snip.get("textOriginal"),

                                    # Metadata
                                    "publishedAt": top_snip.get("publishedAt"),
                                    "updatedAt": top_snip.get("updatedAt"),
                                    "likeCount": top_snip.get("likeCount", 0),
                                    "totalReplyCount": thread_snippet.get("totalReplyCount", 0),
                                })
                                top_level_fetched += 1

                            # Fetch replies even if the top-level comment is included.
                            # This is the important part for the network analysis.
                            total_reply_count = int(thread_snippet.get("totalReplyCount", 0) or 0)
                            if total_reply_count > 0 and top_comment_id:
                                replies = _fetch_all_replies(
                                    client=client,
                                    parent_comment_id=top_comment_id,
                                    video_id=video_id,
                                    game_label=gameLabel,
                                    query_used=query,
                                    thread_id=thread_id,
                                    parent_author_display_name=top_snip.get("authorDisplayName"),
                                    parent_author_channel_id=top_author_id,
                                    after_dt=comment_after,
                                    before_dt=comment_before,
                                    max_replies_per_thread=maxRepliesPerThread,
                                )
                                video["comments"].extend(replies)

                            if maxTopLevelCommentsPerVideo is not None and top_level_fetched >= maxTopLevelCommentsPerVideo:
                                break

                        if maxTopLevelCommentsPerVideo is not None and top_level_fetched >= maxTopLevelCommentsPerVideo:
                            break

                        if reached_before_start and comment_after is not None:
                            break

                        next_comment_page_token = comment_response.get("nextPageToken")
                        if not next_comment_page_token:
                            break

                    collected_count = len(video["comments"])
                    if minCommentsCollectedPerVideo is None or collected_count >= minCommentsCollectedPerVideo:
                        videos.append(video)
                        print(
                            f"  {str(video['title'])[:60]}... -> kept "
                            f"({collected_count} comments/replies) [{len(videos)}/{maxVideos}]"
                        )
                    else:
                        print(f"  {str(video['title'])[:60]}... -> skipped ({collected_count} comments/replies)")

                except Exception as e:
                    print(f"  {str(video['title'])[:60]}... -> comments disabled or error: {e}")

            next_search_page_token = search_response.get("nextPageToken")
            if not next_search_page_token:
                break

        if len(videos) >= maxVideos:
            break

    data = {
        "gameLabel": gameLabel,
        "searchQueries": search_queries,
        "collectedAtUtc": datetime.now(timezone.utc).isoformat(),
        "videos": videos,
    }

    os.makedirs(os.path.dirname(outputFile) or ".", exist_ok=True)
    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(videos)} videos to '{outputFile}'.")
    return data


if __name__ == "__main__":
    ensure_dirs()
    MAX_VIDEOS = 30
    MAX_TOP_LEVEL_COMMENTS = 500
    MAX_REPLIES_PER_THREAD = 20
    MIN_COMMENTS_COLLECTED_PER_VIDEO = 100

    collections = [
        (PROJECT_SEKAI_LABEL, PROJECT_SEKAI_QUERIES, PROJECT_SEKAI_RAW),
        (BANDORI_LABEL, BANDORI_QUERIES, BANDORI_RAW),
    ]

    for game_label, search_queries, output_file in collections:
        fetchYoutubeData(
            searchQueries=search_queries,
            gameLabel=game_label,
            maxVideos=MAX_VIDEOS,
            maxTopLevelCommentsPerVideo=MAX_TOP_LEVEL_COMMENTS,
            maxRepliesPerThread=MAX_REPLIES_PER_THREAD,
            outputFile=str(output_file),
            minCommentsCollectedPerVideo=MIN_COMMENTS_COLLECTED_PER_VIDEO,
            searchOrder="relevance",
        )
