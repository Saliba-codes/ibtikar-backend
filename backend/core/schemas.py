from pydantic import BaseModel, Field
from typing import List, Optional

class PostIn(BaseModel):
    source: str = "x"
    post_id: str
    author_id: str
    text: str
    lang: Optional[str] = None
    created_at: Optional[str] = None

class AnalysisItem(BaseModel):
    post: PostIn
    label: str                  # "harmful" | "safe" | "unknown"
    score: float = 0.0          # 0..1

class AnalysisResponse(BaseModel):
    items: List[AnalysisItem]
    harmful_count: int = Field(0)
    safe_count:   int = 0
    unknown_count:int = 0
