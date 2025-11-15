import time, secrets
from typing import Optional, Dict, Any

# state -> {"verifier": str, "user_id": int, "exp": int}
_state_store: Dict[str, Dict[str, Any]] = {}

def new_state() -> str:
    return secrets.token_urlsafe(24)

def put_state(state: str, verifier: str, user_id: int, ttl_seconds: int = 600) -> None:
    _state_store[state] = {"verifier": verifier, "user_id": int(user_id), "exp": int(time.time()) + ttl_seconds}

def pop_state(state: str) -> Optional[Dict[str, Any]]:
    item = _state_store.pop(state, None)
    if not item:
        return None
    if item["exp"] < time.time():
        return None
    return item
