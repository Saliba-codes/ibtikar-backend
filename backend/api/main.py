from datetime import datetime
import time

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, case

from ..core.config import settings
from ..db.init_db import init_db
from ..db.session import get_db
from ..db import models
from ..core.crypto import enc
from ..core.memory import new_state, put_state, pop_state
from ..clients.x_client import generate_pkce, build_auth_url, exchange_code_for_token
from ..core.schemas import AnalysisResponse, AnalysisItem
from ..core.normalize import x_tweets_to_posts
from ..clients.ibtikar_client import analyze_texts
from backend.db.models import Prediction  # keep as-is since it already works
from ..clients.x_api import get_me, get_my_recent_tweets, get_following_feed

from typing import List, Optional
from pydantic import BaseModel

# ---------- Analysis schemas ----------

class AnalysisPostItem(BaseModel):
    id: int
    user_id: int
    source: str
    post_id: str
    author_id: str
    lang: Optional[str] = None
    text: str
    label: str
    score: float
    post_created_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        orm_mode = True


class AnalysisPostsResponse(BaseModel):
    total: int
    items: List[AnalysisPostItem]


class AuthorSummaryItem(BaseModel):
    author_id: str
    post_count: int
    harmful_count: int
    safe_count: int
    unknown_count: int
    harmful_ratio: float


class AuthorSummaryResponse(BaseModel):
    total: int
    items: List[AuthorSummaryItem]

# ---------- FastAPI app and endpoints ----------

app = FastAPI(title="IbtikarAI Backend", version="0.2.0")
init_db()  # create tables on startup (local dev)

# ---------- Analysis read endpoints ----------

@app.get("/v1/analysis/posts", response_model=AnalysisPostsResponse)
def list_analysis_posts(
    user_id: int,
    label: Optional[str] = None,
    author_id: Optional[str] = None,
    lang: Optional[str] = None,
    from_created_at: Optional[datetime] = None,
    to_created_at: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    List analyzed posts for a given user, with optional filters.
    Results are ordered by predictions.created_at DESC.
    """
    # basic safety on limit
    if limit > 200:
        limit = 200
    if limit < 1:
        limit = 1
    if offset < 0:
        offset = 0

    query = db.query(models.Prediction).filter(models.Prediction.user_id == user_id)

    if label:
        query = query.filter(models.Prediction.label == label)
    if author_id:
        query = query.filter(models.Prediction.author_id == author_id)
    if lang:
        query = query.filter(models.Prediction.lang == lang)
    if from_created_at:
        query = query.filter(models.Prediction.created_at >= from_created_at)
    if to_created_at:
        query = query.filter(models.Prediction.created_at <= to_created_at)

    total = query.count()

    predictions = (
        query.order_by(models.Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = [
        AnalysisPostItem(
            id=p.id,
            user_id=p.user_id,
            source=p.source,
            post_id=p.post_id,
            author_id=p.author_id,
            lang=p.lang,
            text=p.text,
            label=p.label,
            score=float(p.score),
            post_created_at=p.post_created_at,
            created_at=p.created_at,
        )
        for p in predictions
    ]

    return AnalysisPostsResponse(total=total, items=items)


@app.get("/v1/analysis/authors", response_model=AuthorSummaryResponse)
def list_author_summaries(
    user_id: int,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    Aggregate stats per author for a given user.
    Counts how many harmful/safe/unknown tweets each author has.
    """
    if limit > 200:
        limit = 200
    if limit < 1:
        limit = 1
    if offset < 0:
        offset = 0

    # Build aggregation query
    q = (
        db.query(
            models.Prediction.author_id.label("author_id"),
            func.count(models.Prediction.id).label("post_count"),
            func.sum(
                case((models.Prediction.label == "harmful", 1), else_=0)
            ).label("harmful_count"),
            func.sum(
                case((models.Prediction.label == "safe", 1), else_=0)
            ).label("safe_count"),
            func.sum(
                case(
                    (
                        models.Prediction.label.notin_(["harmful", "safe"]),
                        1,
                    ),
                    else_=0,
                )
            ).label("unknown_count"),
        )
        .filter(models.Prediction.user_id == user_id)
        .group_by(models.Prediction.author_id)
    )

    total = q.count()

    rows = (
        q.order_by(func.sum(case((models.Prediction.label == "harmful", 1), else_=0)).desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items: List[AuthorSummaryItem] = []
    for row in rows:
        post_count = int(row.post_count or 0)
        harmful_count = int(row.harmful_count or 0)
        safe_count = int(row.safe_count or 0)
        unknown_count = int(row.unknown_count or 0)

        harmful_ratio = float(harmful_count / post_count) if post_count > 0 else 0.0

        items.append(
            AuthorSummaryItem(
                author_id=str(row.author_id),
                post_count=post_count,
                harmful_count=harmful_count,
                safe_count=safe_count,
                unknown_count=unknown_count,
                harmful_ratio=harmful_ratio,
            )
        )

    return AuthorSummaryResponse(total=total, items=items)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ibtikar-backend",
        "env": settings.ENV,
        "version": "0.2.0",
    }


def ensure_local_user(db: Session) -> int:
    u = db.query(models.User).filter(models.User.id == 1).first()
    if not u:
        u = models.User(id=1)
        db.add(u)
        db.commit()
    return 1


@app.get("/v1/oauth/x/start")
async def x_oauth_start(user_id: int = 1, db: Session = Depends(get_db)):
    # ensure user exists
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        db.add(models.User(id=user_id))
        db.commit()

    verifier, challenge = generate_pkce()
    state = new_state()
    # store BOTH verifier and user_id
    put_state(state, verifier, user_id)
    return RedirectResponse(build_auth_url(state, challenge))


@app.get("/v1/oauth/x/callback")
async def x_oauth_callback(
    code: str | None = None,
    state: str | None = None,
    db: Session = Depends(get_db),
):
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code/state")

    state_data = pop_state(state)
    if not state_data:
        raise HTTPException(status_code=400, detail="State expired/invalid")

    code_verifier = state_data["verifier"]
    user_id = int(state_data["user_id"])

    token = await exchange_code_for_token(code, code_verifier)

    # ensure this user exists
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        db.add(models.User(id=user_id))
        db.commit()

    existing = db.query(models.XToken).filter(models.XToken.user_id == user_id).first()
    if not existing:
        db.add(
            models.XToken(
                user_id=user_id,
                access_token=enc(token.get("access_token", "")),
                refresh_token=enc(token.get("refresh_token", ""))
                if token.get("refresh_token")
                else None,
                scope=token.get("scope"),
                token_type=token.get("token_type"),
                expires_in=token.get("expires_in"),
            )
        )
    else:
        existing.access_token = enc(token.get("access_token", ""))
        existing.refresh_token = (
            enc(token.get("refresh_token", ""))
            if token.get("refresh_token")
            else None
        )
        existing.scope = token.get("scope")
        existing.token_type = token.get("token_type")
        existing.expires_in = token.get("expires_in")

    db.commit()
    return {"linked": True, "user_id": user_id, "scopes": token.get("scope")}


@app.get("/v1/me/link-status")
def link_status(user_id: int = 1, db: Session = Depends(get_db)):
    xt = db.query(models.XToken).filter(models.XToken.user_id == user_id).first()
    return {
        "user_id": user_id,
        "linked": bool(xt),
        "scopes": xt.scope if xt else None,
    }


@app.get("/v1/x/me")
async def x_me(user_id: int = Query(1), db: Session = Depends(get_db)):
    return await get_me(user_id, db)


@app.get("/v1/x/my-posts")
async def x_my_posts(
    user_id: int = Query(1),
    limit: int = Query(20),
    db: Session = Depends(get_db),
):
    return await get_my_recent_tweets(user_id, db, max_results=limit)


@app.get("/v1/x/feed")
async def x_feed(
    user_id: int = Query(1),
    authors_limit: int = Query(25),
    per_batch: int = Query(20),
    db: Session = Depends(get_db),
):
    """
    “Timeline-lite”: pulls recent posts from accounts the user follows.
    Good enough for Free/Basic tiers. For stricter limits, reduce authors_limit/per_batch.
    """
    return await get_following_feed(
        user_id, db, authors_limit=authors_limit, per_batch=per_batch
    )


@app.get("/v1/x/feed/normalized")
async def x_feed_normalized(
    user_id: int = Query(1),
    authors_limit: int = Query(15),
    per_batch: int = Query(15),
    db: Session = Depends(get_db),
):
    raw = await get_following_feed(
        user_id, db, authors_limit=authors_limit, per_batch=per_batch
    )
    if isinstance(raw, dict) and raw.get("rate_limited"):
        reset = raw.get("reset")
        try:
            reset_human = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(reset)))
                if reset
                else None
            )
        except Exception:
            reset_human = None
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "resource": raw.get("resource"),
                "reset_epoch": reset,
                "reset_time": reset_human,
                "limit": raw.get("limit"),
                "remaining": raw.get("remaining"),
            },
        )

    return {"items": [p.dict() for p in x_tweets_to_posts(raw)]}


@app.post("/v1/analysis/preview", response_model=AnalysisResponse)
async def analysis_preview(
    user_id: int = Query(1),
    authors_limit: int = Query(15),
    per_batch: int = Query(15),
    db: Session = Depends(get_db),
):
    raw = await get_following_feed(
        user_id, db, authors_limit=authors_limit, per_batch=per_batch
    )
    if isinstance(raw, dict) and raw.get("rate_limited"):
        reset = raw.get("reset")
        try:
            reset_human = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(reset)))
                if reset
                else None
            )
        except Exception:
            reset_human = None
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "resource": raw.get("resource"),
                "reset_epoch": reset,
                "reset_time": reset_human,
                "limit": raw.get("limit"),
                "remaining": raw.get("remaining"),
            },
        )

    posts = x_tweets_to_posts(raw)
    if not posts:
        return AnalysisResponse(
            items=[], harmful_count=0, safe_count=0, unknown_count=0
        )

    preds = await analyze_texts([p.text for p in posts])

    items: list[AnalysisItem] = []
    hc = sc = uc = 0

    for p, pr in zip(posts, preds):
        label = pr.get("label", "unknown")
        score = float(pr.get("score", 0.0))

        items.append(AnalysisItem(post=p, label=label, score=score))
        if label == "harmful":
            hc += 1
        elif label == "safe":
            sc += 1
        else:
            uc += 1

        # Key we use to avoid duplicates
        post_id_str = str(p.post_id) if getattr(p, "post_id", None) is not None else None

        # Check if a prediction for this (user, source, post_id) already exists
        existing_pred = (
            db.query(Prediction)
            .filter(
                Prediction.user_id == user_id,
                Prediction.source == "x",
                Prediction.post_id == post_id_str,
            )
            .first()
        )

        if existing_pred:
            # Update existing prediction instead of inserting a duplicate
            existing_pred.author_id = (
                str(p.author_id) if getattr(p, "author_id", None) else None
            )
            existing_pred.lang = getattr(p, "lang", None)
            existing_pred.text = p.text
            existing_pred.label = label
            existing_pred.score = score
            existing_pred.post_created_at = (
                p.created_at if isinstance(p.created_at, datetime) else None
            )
            existing_pred.created_at = datetime.utcnow()
        else:
            # Create a new prediction
            db_obj = Prediction(
                user_id=user_id,
                source="x",
                post_id=post_id_str,
                author_id=(
                    str(p.author_id) if getattr(p, "author_id", None) else None
                ),
                lang=getattr(p, "lang", None),
                text=p.text,
                label=label,
                score=score,
                post_created_at=(
                    p.created_at if isinstance(p.created_at, datetime) else None
                ),
            )
            db.add(db_obj)

    # Save all changes (new + updated) at once
    db.commit()

    return AnalysisResponse(
        items=items,
        harmful_count=hc,
        safe_count=sc,
        unknown_count=uc,
    )
