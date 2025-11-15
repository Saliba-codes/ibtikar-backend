# backend/clients/x_api.py
from typing import Dict, Any
import httpx
from sqlalchemy.orm import Session

from backend.db.models import XToken
from backend.core.crypto import dec, enc
from backend.core.config import settings

API = "https://api.twitter.com/2"

# ----------------------- token helpers -----------------------

def _get_token_pair(user_id: int, db: Session) -> tuple[str, str | None, XToken]:
    row: XToken | None = db.query(XToken).filter(XToken.user_id == user_id).first()
    if not row:
        raise RuntimeError(f"No token for user_id={user_id}")
    access = dec(row.access_token)
    refresh = dec(row.refresh_token) if row.refresh_token else None
    return access, refresh, row

def _client(bearer: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=API,
        headers={"Authorization": f"Bearer {bearer}"},
        timeout=20.0,
    )

async def _refresh_access_token(refresh_token: str) -> dict:
    """
    OAuth2 PKCE refresh: needs client_id, not client_secret.
    """
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": settings.X_CLIENT_ID,
    }
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(
            "https://api.twitter.com/2/oauth2/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        r.raise_for_status()
        return r.json()

def _store_tokens(row: XToken, token_json: dict, db: Session) -> str:
    new_access = token_json.get("access_token")
    new_refresh = token_json.get("refresh_token")  # may be absent depending on X
    if new_access:
        row.access_token = enc(new_access)
    if new_refresh:
        row.refresh_token = enc(new_refresh)
    row.scope = token_json.get("scope", row.scope)
    row.token_type = token_json.get("token_type", row.token_type)
    row.expires_in = token_json.get("expires_in", row.expires_in)
    db.commit()
    return new_access or dec(row.access_token)

# ----------------------- X API wrappers -----------------------

async def get_me(user_id: int, db: Session) -> Dict[str, Any]:
    """
    Calls /users/me, auto-refreshes on 401 once, then retries.
    """
    access, refresh, row = _get_token_pair(user_id, db)

    async def _call(bearer: str):
        async with _client(bearer) as c:
            return await c.get("/users/me", params={"user.fields": "name,username,profile_image_url"})

    r = await _call(access)
    if r.status_code == 401 and refresh:
        tj = await _refresh_access_token(refresh)
        access = _store_tokens(row, tj, db)
        r = await _call(access)

    r.raise_for_status()
    return r.json()

async def get_my_recent_tweets(user_id: int, db: Session, max_results: int = 20) -> Dict[str, Any]:
    me = await get_me(user_id, db)
    uid = me["data"]["id"]
    access, _, _ = _get_token_pair(user_id, db)
    async with _client(access) as c:
        r = await c.get(
            f"/users/{uid}/tweets",
            params={
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,author_id,lang,public_metrics",
                "exclude": "retweets,replies",
            },
        )
        r.raise_for_status()
        return r.json()

async def get_following_feed(user_id: int, db: Session, authors_limit: int = 25, per_batch: int = 20) -> Dict[str, Any]:
    """
    Free-tier friendly fallback: own tweets + mentions.
    Returns {"data": [...]} or {"rate_limited": True, ...} when 429.
    """
    me = await get_me(user_id, db)
    uid = me["data"]["id"]
    access, _, _ = _get_token_pair(user_id, db)

    async with _client(access) as c:
        # 1) own tweets
        r1 = await c.get(
            f"/users/{uid}/tweets",
            params={
                "max_results": min(per_batch, 50),
                "tweet.fields": "created_at,author_id,lang,public_metrics",
                "exclude": "retweets,replies",
            },
        )
        if r1.status_code == 429:
            return {
                "rate_limited": True,
                "resource": "user_tweets",
                "reset": r1.headers.get("x-rate-limit-reset"),
                "limit": r1.headers.get("x-rate-limit-limit"),
                "remaining": r1.headers.get("x-rate-limit-remaining"),
            }
        r1.raise_for_status()
        data = r1.json().get("data", []) or []

        # 2) mentions
        r2 = await c.get(
            f"/users/{uid}/mentions",
            params={
                "max_results": min(per_batch, 50),
                "tweet.fields": "created_at,author_id,lang,public_metrics",
            },
        )
        if r2.status_code == 429:
            return {
                "rate_limited": True,
                "resource": "mentions",
                "reset": r2.headers.get("x-rate-limit-reset"),
                "limit": r2.headers.get("x-rate-limit-limit"),
                "remaining": r2.headers.get("x-rate-limit-remaining"),
            }
        r2.raise_for_status()
        mentions = r2.json().get("data", []) or []

    return {"data": data + mentions}
