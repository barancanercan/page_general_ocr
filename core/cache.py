import os
import json
import hashlib
from config.settings import CACHE_DIR


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _hash_key(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()


def get_cache(namespace: str, key: str):
    """Cache'den veri oku. Yoksa None döndür."""
    path = os.path.join(CACHE_DIR, namespace, f"{_hash_key(key)}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def set_cache(namespace: str, key: str, value):
    """Cache'e veri yaz."""
    _ensure_cache_dir()
    ns_dir = os.path.join(CACHE_DIR, namespace)
    os.makedirs(ns_dir, exist_ok=True)
    path = os.path.join(ns_dir, f"{_hash_key(key)}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False)


def clear_cache(namespace: str = None):
    """Cache temizle."""
    import shutil
    if namespace:
        path = os.path.join(CACHE_DIR, namespace)
        if os.path.exists(path):
            shutil.rmtree(path)
    elif os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
