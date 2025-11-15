from cryptography.fernet import Fernet
from .config import settings

_f = Fernet(settings.FERNET_KEY.encode())

def enc(s: str) -> bytes:
    return _f.encrypt(s.encode())

def dec(b: bytes) -> str:
    return _f.decrypt(b).decode()
