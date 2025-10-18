import aiosqlite
from datetime import datetime, timedelta

DB_NAME = "users.db"

async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            premium_until TEXT,
            messages_today INTEGER DEFAULT 0,
            last_message_date TEXT
        )
        """)
        await db.commit()

async def add_user(user_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_id,))
        await db.commit()

async def is_premium(user_id: int) -> bool:
    async with aiosqlite.connect(DB_NAME) as db:
        cur = await db.execute("SELECT premium_until FROM users WHERE id = ?", (user_id,))
        row = await cur.fetchone()
        if row and row[0]:
            try:
                return datetime.fromisoformat(row[0]) > datetime.now()
            except ValueError:
                return False
        return False

async def set_premium(user_id: int, until: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("UPDATE users SET premium_until = ? WHERE id = ?", (until, user_id))
        await db.commit()

async def can_send_message(user_id: int, limit=5) -> bool:
    async with aiosqlite.connect(DB_NAME) as db:
        cur = await db.execute("SELECT messages_today, last_message_date FROM users WHERE id = ?", (user_id,))
        row = await cur.fetchone()
        today = datetime.now().date().isoformat()
        if not row:
            await add_user(user_id)
            return True
        messages_today, last_date = row
        if last_date != today:
            await db.execute("UPDATE users SET messages_today = 1, last_message_date = ? WHERE id = ?", (today, user_id))
            await db.commit()
            return True
        if messages_today < limit:
            await db.execute("UPDATE users SET messages_today = messages_today + 1 WHERE id = ?", (user_id,))
            await db.commit()
            return True
        return False
