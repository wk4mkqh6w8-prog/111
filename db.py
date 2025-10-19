import aiosqlite
from datetime import datetime, timedelta, date

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
        await db.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            amount REAL,
            asset TEXT,
            created_at TEXT NOT NULL
        )
        """)
        await db.commit()

async def add_user(user_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_id,))
        await db.commit()

async def get_user(user_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        cur = await db.execute("SELECT id, premium_until, messages_today, last_message_date FROM users WHERE id = ?", (user_id,))
        return await cur.fetchone()

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

async def set_premium(user_id: int, until_iso: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("UPDATE users SET premium_until = ? WHERE id = ?", (until_iso, user_id))
        await db.commit()

async def set_premium_days(user_id: int, days: int = 30):
    until = (datetime.now() + timedelta(days=days)).isoformat()
    await set_premium(user_id, until)

async def remove_premium(user_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("UPDATE users SET premium_until = NULL WHERE id = ?", (user_id,))
        await db.commit()

async def can_send_message(user_id: int, limit=5) -> bool:
    today = datetime.now().date().isoformat()
    async with aiosqlite.connect(DB_NAME) as db:
        cur = await db.execute("SELECT messages_today, last_message_date FROM users WHERE id = ?", (user_id,))
        row = await cur.fetchone()

        if not row:
            await add_user(user_id)
            await db.execute("UPDATE users SET messages_today = 1, last_message_date = ? WHERE id = ?", (today, user_id))
            await db.commit()
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

# ---------- Статистика пользователей ----------
async def stats():
    today_str = date.today().isoformat()
    async with aiosqlite.connect(DB_NAME) as db:
        total = (await (await db.execute("SELECT COUNT(*) FROM users")).fetchone())[0]
        premium = (await (await db.execute(
            "SELECT COUNT(*) FROM users WHERE premium_until IS NOT NULL AND premium_until > ?",
            (datetime.now().isoformat(),)
        )).fetchone())[0]
        active_today = (await (await db.execute(
            "SELECT COUNT(*) FROM users WHERE last_message_date = ?",
            (today_str,)
        )).fetchone())[0]
        # покупки за сегодня
        purchases_today = (await (await db.execute(
            "SELECT COUNT(*) FROM payments WHERE DATE(created_at) = ?",
            (today_str,)
        )).fetchone())[0]
        return {
            "total": total,
            "premium": premium,
            "active_today": active_today,
            "purchases_today": purchases_today,
        }

# ---------- Учёт оплат ----------
async def log_payment(user_id: int, amount: float | None, asset: str | None):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "INSERT INTO payments (user_id, amount, asset, created_at) VALUES (?, ?, ?, ?)",
            (user_id, amount, asset, datetime.now().isoformat())
        )
        await db.commit()

async def sales_summary():
    """Краткая сводка: сегодня, 7 дней, всего (по количеству)."""
    today_str = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=6)).isoformat()  # включительно 7 дней

    async with aiosqlite.connect(DB_NAME) as db:
        today = (await (await db.execute(
            "SELECT COUNT(*) FROM payments WHERE DATE(created_at) = ?",
            (today_str,)
        )).fetchone())[0]
        week = (await (await db.execute(
            "SELECT COUNT(*) FROM payments WHERE DATE(created_at) >= ?",
            (week_ago,)
        )).fetchone())[0]
        total = (await (await db.execute("SELECT COUNT(*) FROM payments")).fetchone())[0]
        return {"today": today, "week": week, "total": total}

async def daily_breakdown(days: int = 7):
    """Сколько оплат по дням за N последних дней (включая сегодня)."""
    start = (date.today() - timedelta(days=days-1)).isoformat()
    out = []
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT DATE(created_at) d, COUNT(*) c FROM payments "
            "WHERE DATE(created_at) >= ? GROUP BY DATE(created_at) ORDER BY d",
            (start,)
        ) as cur:
            async for d, c in cur:
                out.append((d, c))
    # заполним нулями пропуски
    wanted = [(date.today() - timedelta(days=i)).isoformat() for i in range(days-1, -1, -1)]
    mapping = {d: c for d, c in out}
    return [(d, mapping.get(d, 0)) for d in wanted]
