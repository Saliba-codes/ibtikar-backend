from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,  # NEW
)
from sqlalchemy.orm import relationship

from .session import Base


class User(Base):
    __tablename__ = "users"

    # This id is the same "user_id" you use everywhere (1, 2, …)
    id = Column(Integer, primary_key=True, index=True)

    # Optional – you can store the X handle / username here if you want
    handle = Column(String, nullable=True, unique=False, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tokens = relationship("XToken", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} handle={self.handle!r}>"


class XToken(Base):
    """
    Stores the OAuth tokens you get from X for each user.
    This is what /v1/oauth/x/start and /v1/oauth/x/callback work with.
    """

    __tablename__ = "x_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    access_token = Column(String, nullable=False)
    refresh_token = Column(String, nullable=True)

    scope = Column(String, nullable=True)
    token_type = Column(String, nullable=True)
    expires_in = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to User
    user = relationship("User", back_populates="tokens")

    def __repr__(self) -> str:
        return f"<XToken id={self.id} user_id={self.user_id}>"


class Prediction(Base):
    """
    Stores one IbtikarAI prediction per post.
    This is what /v1/analysis/preview writes to.
    """

    __tablename__ = "predictions"

    # Ensure one prediction per (user, source, post_id)
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source",
            "post_id",
            name="uq_prediction_user_source_post",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)

    # Which NGO user this prediction belongs to
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Where the post came from ("x", etc.)
    source = Column(String, nullable=False)

    # X tweet info
    post_id = Column(String, nullable=True)    # tweet id
    author_id = Column(String, nullable=True)  # tweet author id
    lang = Column(String, nullable=True)

    # Raw text we sent to IbtikarAI
    text = Column(Text, nullable=False)

    # Model output
    label = Column(String, nullable=False)  # "harmful" / "safe" / etc.
    score = Column(Float, nullable=False)

    # When the original post was created (from X), optional
    post_created_at = Column(DateTime, nullable=True)

    # When we ran the analysis (or last updated it)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to User
    user = relationship("User", back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} user_id={self.user_id} "
            f"source={self.source!r} post_id={self.post_id!r} "
            f"label={self.label!r} score={self.score}>"
        )
