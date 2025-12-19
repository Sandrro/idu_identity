from __future__ import annotations

import datetime
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests


class VKParser:
    """VK wall posts/comments parser adapted from SOIKA for modern Python."""

    API_VERSION = "5.131"
    COUNT_ITEMS = 100
    TIMEOUT_LIMIT = 15

    @staticmethod
    def _request_json(url: str, params: Dict[str, Any], *, timeout: int = 20) -> Dict[str, Any]:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_group_name(domain: str, access_token: str) -> pd.DataFrame:
        params = {"group_id": domain, "access_token": access_token, "v": VKParser.API_VERSION}
        try:
            data = VKParser._request_json("https://api.vk.com/method/groups.getById", params)
        except Exception as exc:  # pragma: no cover - network
            print("Error while fetching group name:", exc)
            return pd.DataFrame({"group_name": [None]})

        if "response" in data and data["response"]:
            group_name = data["response"][0].get("name")
            return pd.DataFrame({"group_name": [group_name]})

        print("Error while fetching group name:", data)
        return pd.DataFrame({"group_name": [None]})

    @staticmethod
    def get_owner_id_by_domain(domain: str, access_token: str) -> Optional[int]:
        """Get the owner ID of a VK group by its domain."""
        params = {
            "domain": domain,
            "access_token": access_token,
            "v": VKParser.API_VERSION,
            "count": 1,
        }
        try:
            data = VKParser._request_json("https://api.vk.com/method/wall.get", params)
        except Exception:
            return None

        response = data.get("response")
        if not response or not response.get("items"):
            return None
        return response["items"][0].get("owner_id")

    @staticmethod
    def get_group_post_ids(domain: str, access_token: str, post_num_limit: int, step: int) -> List[int]:
        """Retrieve a list of post IDs for a given group."""
        offset = 0
        post_ids: List[int] = []

        while offset < post_num_limit:
            print(offset, " | ", post_num_limit, end="\r")
            data = VKParser._request_json(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": access_token,
                    "v": VKParser.API_VERSION,
                    "domain": domain,
                    "count": step,
                    "offset": offset,
                },
                timeout=VKParser.TIMEOUT_LIMIT,
            ).get("response", {})
            time.sleep(random.random())

            post_ids_new = [k["id"] for k in data.get("items", []) if "id" in k]
            post_ids += post_ids_new
            offset += step

        return post_ids

    @staticmethod
    def get_subcomments(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve subcomments from the VK API."""
        subcomments: List[Dict[str, Any]] = []
        data = VKParser._request_json("https://api.vk.com/method/wall.getComments", params)
        time.sleep(random.random())

        if "response" in data:
            for item in data["response"].get("items", []):
                item["date"] = datetime.datetime.utcfromtimestamp(item["date"]).strftime("%Y-%m-%d %H:%M:%S")
                if "likes" in item:
                    item["likes.count"] = item["likes"].get("count")
                subcomments.append(item)

        return subcomments

    def get_comments(self, owner_id: int, post_id: int, access_token: str) -> List[Dict[str, Any]]:
        """Get comments for a post on VK."""
        params = {
            "owner_id": owner_id,
            "post_id": post_id,
            "access_token": access_token,
            "v": VKParser.API_VERSION,
            "extended": 1,
            "count": 100,
            "need_likes": 1,
        }

        comments: List[Dict[str, Any]] = []
        data = VKParser._request_json("https://api.vk.com/method/wall.getComments", params)
        time.sleep(random.random())

        if "response" in data:
            for item in data["response"].get("items", []):
                if not item.get("text"):
                    continue
                item["date"] = datetime.datetime.utcfromtimestamp(item["date"]).strftime("%Y-%m-%d %H:%M:%S")
                if "likes" in item:
                    item["likes.count"] = item["likes"].get("count")
                comments.append(item)
                if item.get("thread", {}).get("count", 0) > 0:
                    params["comment_id"] = item["id"]
                    subcomments = VKParser.get_subcomments(params)
                    comments.extend(subcomments)
        return comments

    @staticmethod
    def comments_to_dataframe(comments: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        """Convert comments to a DataFrame."""
        df = pd.DataFrame(comments)
        columns = ["id", "from_id", "date", "text", "post_id", "parents_stack", "likes.count"]
        return df[columns]

    @staticmethod
    def run_posts(
        *,
        domain: str,
        access_token: str,
        cutoff_date: str,
        number_of_messages: int | float = float("inf"),
        step: int = 50,
    ) -> pd.DataFrame:
        """Retrieve posts from VK wall."""
        offset = 0
        all_posts: List[pd.DataFrame] = []
        if step > number_of_messages:
            step = int(number_of_messages)
        while offset < number_of_messages:
            print(offset, " | ", number_of_messages, end="\r")

            data = VKParser._request_json(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": access_token,
                    "v": VKParser.API_VERSION,
                    "domain": domain,
                    "count": step,
                    "offset": offset,
                },
                timeout=600,
            )
            if "response" not in data:
                continue

            items = data["response"].get("items", [])
            offset += step
            current_posts = pd.json_normalize(items)
            if current_posts.empty:
                break
            current_posts = current_posts[["date", "id", "text", "views.count", "likes.count", "reposts.count"]]
            current_posts["date"] = [
                datetime.datetime.fromtimestamp(current_posts["date"][i]) for i in range(len(current_posts["date"]))
            ]
            current_posts["type"] = "post"
            all_posts.append(current_posts)
            print(current_posts.date.min())
            if any(current_posts["date"] < datetime.datetime.strptime(cutoff_date, "%Y-%m-%d")):
                print("posts downloaded")
                break
            time.sleep(random.random())

        if not all_posts:
            return pd.DataFrame(columns=["date", "id", "text", "views.count", "likes.count", "reposts.count", "type"])

        df_posts = pd.concat(all_posts).reset_index(drop=True)
        df_posts = df_posts[df_posts.text.map(lambda x: len(x)) > 0]
        df_posts["text"] = df_posts["text"].str.replace(r"\n", "", regex=True)
        df_posts["link"] = df_posts["text"].str.extract(r"(https://\S+)")
        return df_posts

    @staticmethod
    def run_comments(domain: str, post_ids: Iterable[int], access_token: str) -> Optional[pd.DataFrame]:
        owner_id = VKParser.get_owner_id_by_domain(domain, access_token)
        if owner_id is None:
            return None
        all_comments: List[Dict[str, Any]] = []
        for post_id in post_ids:
            comments = VKParser().get_comments(owner_id, post_id, access_token)
            all_comments.extend(comments)
        if not all_comments:
            print("no comments")
            return None

        df = VKParser.comments_to_dataframe(all_comments)
        df["type"] = "comment"
        df = df.reset_index(drop=True)
        print("comments downloaded")
        return df

    @staticmethod
    def run_parser(
        *,
        domain: str,
        access_token: str,
        cutoff_date: str,
        number_of_messages: int | float = float("inf"),
        step: int = 100,
    ) -> pd.DataFrame:
        """Runs the parser and returns a combined DataFrame of posts and comments."""
        df_posts = VKParser.run_posts(
            domain=domain,
            access_token=access_token,
            step=step,
            cutoff_date=cutoff_date,
            number_of_messages=number_of_messages,
        )
        post_ids = df_posts["id"].tolist() if not df_posts.empty else []

        df_comments = VKParser.run_comments(domain=domain, post_ids=post_ids, access_token=access_token)
        if df_comments is not None:
            df_comments.loc[df_comments["parents_stack"].apply(lambda x: len(x) > 0), "type"] = "reply"
            for idx in df_comments.index:
                tmp = df_comments.at[idx, "parents_stack"]
                if tmp is not None:
                    df_comments.at[idx, "parents_stack"] = tmp[0] if len(tmp) > 0 else None

            df_combined = pd.concat([df_posts, df_comments], ignore_index=True)
        else:
            df_combined = df_posts

        df_group_name = VKParser.get_group_name(domain, access_token)
        df_combined["group_name"] = df_group_name["group_name"][0]

        return df_combined
