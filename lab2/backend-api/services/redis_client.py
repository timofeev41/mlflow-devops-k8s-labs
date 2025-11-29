import redis.asyncio as redis
import json
import os
from typing import List, Optional


class RedisService:
    """Клиент для взаимодействия с Redis."""
    def __init__(self, host=None, port=6379, db=0):
        if host is None:
            host = os.getenv("REDIS_HOST", "localhost")
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    async def ping(self) -> bool:
        try:
            return await self.redis.ping()
        except Exception:
            return False

    async def cache_vector(self, vector_id: int, vector: List[float], ttl: int = 300):
        key = f"vector:{vector_id}"
        value = json.dumps(vector)
        await self.redis.set(key, value, ex=ttl)

    async def get_cached_vector(self, vector_id: int) -> Optional[List[float]]:
        key = f"vector:{vector_id}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def cache_search_result(self, query_hash: str, results: List, ttl: int = 60):
        key = f"search:{query_hash}"
        value = json.dumps(results)
        await self.redis.set(key, value, ex=ttl)

    async def get_cached_search(self, query_hash: str) -> Optional[List]:
        key = f"search:{query_hash}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None