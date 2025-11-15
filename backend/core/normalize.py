from typing import Dict, List
from .schemas import PostIn

def x_tweets_to_posts(payload: Dict) -> List[PostIn]:
    data = payload.get("data", []) or []
    out: List[PostIn] = []
    for t in data:
        out.append(PostIn(
            post_id=str(t.get("id")),
            author_id=str(t.get("author_id")),
            text=(t.get("text") or "").strip(),
            lang=t.get("lang"),
            created_at=t.get("created_at"),
        ))
    return out
