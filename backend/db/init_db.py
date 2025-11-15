# backend/db/init_db.py
from .session import engine, Base
from . import models  # 👈 make sure this is imported so Prediction is registered

def init_db() -> None:
    # Any other startup logic…
    Base.metadata.create_all(bind=engine)
