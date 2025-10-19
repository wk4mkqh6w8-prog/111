import aiosqlite
from datetime import datetime, date

DB_NAME = "users.db"

async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        # пользователи
        await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id INTEGER UNIQUE,
            premium INTEGER DEFAULT 0,
            expires_at TEXT,
            messages_today INTEGER DEFAULT 0,
            last_reset TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            model TEXT
        )
        """)
        # платежи
        await db.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id INTEGER,
            amount REAL,
            asset TEXT,
            paid_at TEXT
        )
        """)

        # миграции на случай старой базы
        try:
            await db.execute("ALTER TABLE users ADD COLUMN model TEXT")
        except Exception:
            pass
        await db.execute("UPDATE users SET model = COALESCE(model, 'openai:gpt-4o-mini')")

        try:
            await db.execute("ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            pass

        await db.commit()

async def add_user(tg_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("""
            INSERT OR IGNORE INTO users (tg_id, messages_today, last_reset, model)
            VALUES (?, 0, ?, 'openai:gpt-4o-mini')
        """, (tg_id, date.today().isoformat()))
        await db.commit()

async def set_premium(tg_id: int, expires_at: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "UPDATE users SET premium = 1, expires_at = ? WHERE tg_id = ?",
            (expires_at, tg_id)
        )
        await db.commit()

async def remove_premium(tg_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "UPDATE users SET premium = 0, expires_at = NULL WHERE tg_id = ?",
            (tg_id,)
        )
        await db.commit()

async def is_premium(tg_id: int) -> bool:
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute("SELECT premium, expires_at FROM users WHERE tg_id = ?", (tg_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return False
            premium, expires_at = row
            if premium and expires_at:
                try:
                    if datetime.now() < datetime.fromisoformat(expires_at):
                        return True
                except Exception:
                    pass
            # истёк — снимем
            await db.execute("UPDATE users SET premium = 0, expires_at = NULL WHERE tg_id = ?", (tg_id,))
            await db.commit()
            return False

async def can_send_message(tg_id: int, limit: int = 5) -> bool:
    today = date.today().isoformat()
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT messages_today, last_reset FROM users WHERE tg_id = ?",
            (tg_id,)
        ) as cur:
            row = await cur.fetchone()

        if not row:
            await add_user(tg_id)
            messages_today, last_reset = 0, today
        else:
            messages_today, last_reset = row

        # новый день — сброс
        if last_reset != today:
            await db.execute(
                "UPDATE users SET messages_today = 0, last_reset = ? WHERE tg_id = ?",
                (today, tg_id)
            )
            await db.commit()
            messages_today = 0

        if messages_today < limit:
            await db.execute(
                "UPDATE users SET messages_today = messages_today + 1 WHERE tg_id = ?",
                (tg_id,)
            )
            await db.commit()
            return True
        return False

# ---- выбор модели ----
async def set_model(tg_id: int, model_code: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("UPDATE users SET model = ? WHERE tg_id = ?", (model_code, tg_id))
        await db.commit()

async def get_model(tg_id: int) -> str:
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute("SELECT model FROM users WHERE tg_id = ?", (tg_id,)) as cur:
            row = await cur.fetchone()
            return (row[0] if row and row[0] else "openai:gpt-4o-mini")

# ---- Статистика / платежи ----
async def record_payment(tg_id: int, amount: float | None, asset: str, paid_at_iso: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "INSERT INTO payments (tg_id, amount, asset, paid_at) VALUES (?, ?, ?, ?)",
            (tg_id, amount, asset, paid_at_iso)
        )
        await db.commit()

async def get_stats_today() -> dict:
    today = date.today().isoformat()
    async with aiosqlite.connect(DB_NAME) as db:
        # оплаты за сегодня
        async with db.execute(
            "SELECT COUNT(*), COALESCE(SUM(amount),0) FROM payments WHERE substr(paid_at,1,10)=?",
            (today,)
        ) as cur:
            cnt, sum_amt = await cur.fetchone()
        # новые пользователи за сегодня
        async with db.execute(
            "SELECT COUNT(*) FROM users WHERE substr(created_at,1,10)=?",
            (today,)
        ) as cur:
            new_users = (await cur.fetchone())[0]
    return {
        "payments": cnt or 0,
        "revenue_usdt": float(sum_amt or 0),
        "new_users": new_users or 0
    }

async def get_totals() -> dict:
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute("SELECT COUNT(*) FROM users") as cur:
            users = (await cur.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM users WHERE premium=1") as cur:
            premium = (await cur.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM payments") as cur:
            payments = (await cur.fetchone())[0]
    return {"users": users or 0, "premium": premium or 0, "payments": payments or 0}
