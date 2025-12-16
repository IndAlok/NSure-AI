import asyncio
import asyncpg
import json
import hashlib
import os
from typing import List, Optional, Dict

class PostgresCache:
    def __init__(self):
        self.pool = None

    async def init_pool(self, database_url: str):
        if self.pool:
            return
        try:
            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=5,
                command_timeout=15,
                server_settings={'jit': 'off'}
            )
            await self._create_tables()
        except Exception:
            self.pool = None

    async def initialize_and_clear_cache(self):
        if not self.pool:
            return

        if os.getenv("CLEAR_CACHE_ON_RESTART", "false").lower() == "true":
            try:
                async with self.pool.acquire() as connection:
                    await connection.execute("TRUNCATE TABLE doc_cache, query_cache RESTART IDENTITY CASCADE;")
            except Exception:
                pass

    async def _create_tables(self):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_cache (
                    url_hash TEXT PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    chunks JSONB NOT NULL,
                    embeddings BYTEA,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_doc_cache_created ON doc_cache(created_at);

                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    url_hash TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_query_cache_lookup ON query_cache(url_hash, question);
            """)

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_doc_cache(self, url: str) -> Optional[Dict]:
        if not self.pool: return None
        url_hash = self._get_hash(url)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT text_content, chunks, embeddings FROM doc_cache WHERE url_hash = $1", url_hash)
            if row:
                return {'text': row['text_content'], 'chunks': json.loads(row['chunks']), 'embeddings': row['embeddings']}
        return None

    async def set_doc_cache(self, url: str, text: str, chunks: List[str], embeddings: bytes):
        if not self.pool: return
        url_hash = self._get_hash(url)
        chunks_json = json.dumps(chunks)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO doc_cache (url_hash, text_content, chunks, embeddings) VALUES ($1, $2, $3, $4) ON CONFLICT (url_hash) DO NOTHING",
                url_hash, text, chunks_json, embeddings
            )

    async def get_query_cache(self, url: str, question: str) -> Optional[str]:
        if not self.pool: return None
        query_hash = self._get_hash(f"{url}:{question}")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT answer FROM query_cache WHERE query_hash = $1", query_hash)
            return row['answer'] if row else None

    async def set_query_cache(self, url: str, question: str, answer: str):
        if not self.pool: return
        url_hash = self._get_hash(url)
        query_hash = self._get_hash(f"{url}:{question}")
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO query_cache (query_hash, url_hash, question, answer) VALUES ($1, $2, $3, $4) ON CONFLICT (query_hash) DO NOTHING",
                query_hash, url_hash, question, answer
            )

    async def cleanup_old_cache(self, days: int = 7):
        if not self.pool: return
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM doc_cache WHERE created_at < NOW() - INTERVAL '%s days'", str(days))
            await conn.execute("DELETE FROM query_cache WHERE created_at < NOW() - INTERVAL '%s days'", str(days))

db_cache = PostgresCache()
