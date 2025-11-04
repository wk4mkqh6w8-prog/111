import os
from datetime import datetime, timezone, date

import aiosqlite

# Путь к БД. На проде задаём SQLITE_PATH=/var/data/neurobot.db
DB_PATH = os.getenv("SQLITE_PATH", "./data/neurobot.db")
# гарантируем, что папка для файла БД существует
def _ensure_db_dir():
    d = os.path.dirname(DB_PATH)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ========= ВСПОМОГАТЕЛЬНЫЕ =========

def _utcnow_iso() -> str:
    # ISO-8601 в UTC: подходит для лексикографического сравнения строк
    return datetime.now(timezone.utc).isoformat()

def _today_str() -> str:
    # Локальная календарная дата YYYY-MM-DD (для дневного лимита)
    return date.today().isoformat()


# ========= ИНИЦИАЛИЗАЦИЯ =========

async def init_db():
    """Создаём таблицы, если их нет."""
    _ensure_db_dir()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(
            """
            PRAGMA journal_mode=WAL;

            -- Пользователь
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY,
                created_at    TEXT    NOT NULL,
                referrer_id   INTEGER,
                free_credits  INTEGER NOT NULL DEFAULT 0
            );

            -- Сообщения пользователя (для дневного лимита)
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                created_at TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_user_date
              ON messages(user_id, created_at);

            -- Отметки об отправке уведомления об окончании премиума
            CREATE TABLE IF NOT EXISTS premium_notices (
                user_id          INTEGER PRIMARY KEY,
                expired_notified INTEGER NOT NULL DEFAULT 0,
                last_warn_at     TEXT
            );

            -- Премиум статус
            CREATE TABLE IF NOT EXISTS premiums (
                user_id     INTEGER PRIMARY KEY,
                expires_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_premiums_expires
              ON premiums(expires_at);

            -- События активации/продления премиума (для статистики)
            CREATE TABLE IF NOT EXISTS premium_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                activated_at TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_premium_events_day
              ON premium_events(activated_at);
              -- Режим диалога и активный чат
            CREATE TABLE IF NOT EXISTS user_prefs (
                user_id         INTEGER PRIMARY KEY,
                chat_mode       TEXT NOT NULL DEFAULT 'simple', -- 'simple' | 'rooms'
                active_chat_id  INTEGER
            );

            -- Список чатов пользователя
            CREATE TABLE IF NOT EXISTS chats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                title       TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id);

            -- История сообщений по чатам
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     INTEGER NOT NULL,
                role        TEXT NOT NULL,   -- 'system' | 'user' | 'assistant'
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id);
            """
        )
        await db.commit()


# ========= ПОЛЬЗОВАТЕЛИ / РЕФЕРАЛКИ =========

async def add_user(user_id: int):
    """Создаём запись о пользователе, если её ещё нет."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO users(user_id, created_at, referrer_id, free_credits)
            VALUES (?, ?, NULL, 0)
            """,
            (user_id, _utcnow_iso()),
        )
        await db.commit()


async def set_referrer_if_empty(user_id: int, referrer_id: int) -> bool:
    """
    Привязываем реферера, только если ещё не установлен.
    Возвращает True, если привязали впервые.
    """
    if user_id == referrer_id:
        return False
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT referrer_id FROM users WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        await cur.close()

        if row is None:
            await db.execute(
                "INSERT INTO users(user_id, created_at, referrer_id, free_credits) VALUES (?, ?, ?, 0)",
                (user_id, _utcnow_iso(), referrer_id),
            )
            await db.commit()
            return True

        if row[0] is None:
            await db.execute(
                "UPDATE users SET referrer_id = ? WHERE user_id = ?",
                (referrer_id, user_id),
            )
            await db.commit()
            return True

        return False


async def get_referrer_id(user_id: int) -> int | None:
    """Вернёт referrer_id или None."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT referrer_id FROM users WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row and row[0] is not None else None


async def add_free_credits(user_id: int, n: int):
    """Начислить бонусные кредиты (например, за реферала)."""
    if n <= 0:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET free_credits = COALESCE(free_credits,0) + ? WHERE user_id = ?",
            (n, user_id),
        )
        await db.commit()


async def get_free_credits(user_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT COALESCE(free_credits,0) FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row else 0


async def consume_free_credit(user_id: int) -> bool:
    """
    Списать один бонусный кредит, если есть. Возвращает True, если списание выполнено.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("BEGIN IMMEDIATE")
        cur = await db.execute(
            "SELECT COALESCE(free_credits,0) FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
        if not row or int(row[0]) <= 0:
            await db.execute("ROLLBACK")
            await cur.close()
            return False

        await db.execute(
            "UPDATE users SET free_credits = free_credits - 1 WHERE user_id = ?",
            (user_id,),
        )
        await db.execute("COMMIT")
        await cur.close()
        return True


# ========= ДНЕВНОЙ ЛИМИТ =========

async def get_usage_today(user_id: int) -> int:
    """Сколько сообщений пользователь уже отправил сегодня (календарный день)."""
    day = _today_str()  # YYYY-MM-DD
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT COUNT(*)
            FROM messages
            WHERE user_id = ?
              AND substr(created_at, 1, 10) = ?
            """,
            (user_id, day),
        )
        n = (await cur.fetchone())[0]
        await cur.close()
        return int(n)


async def can_send_message(user_id: int, limit: int = 5) -> bool:
    """
    Если пользователь ещё не исчерпал дневной лимит, зачтём сообщение и вернём True.
    Иначе — False.
    """
    used = await get_usage_today(user_id)
    if used >= limit:
        return False
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages(user_id, created_at) VALUES (?, ?)",
            (user_id, _utcnow_iso()),
        )
        await db.commit()
    return True


# ========= ПРЕМИУМ =========

async def is_premium(user_id: int) -> bool:
    """Активен ли премиум сейчас."""
    now = _utcnow_iso()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT expires_at FROM premiums WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return False
        return row[0] > now  # корректно для ISO-8601 UTC строк


async def set_premium(user_id: int, expires_at_iso: str):
    """Выдаёт/продлевает премиум + логирует событие для статистики."""
    now = _utcnow_iso()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO premiums(user_id, expires_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              expires_at = excluded.expires_at,
              updated_at = excluded.updated_at
            """,
            (user_id, expires_at_iso, now),
        )
        await db.execute(
            "INSERT INTO premium_events(user_id, activated_at) VALUES (?, ?)",
            (user_id, now),
        )
        await db.execute(
            """
            INSERT INTO premium_notices(user_id, expired_notified, last_warn_at)
            VALUES (?, 0, NULL)
            ON CONFLICT(user_id) DO UPDATE SET
              expired_notified = 0,
              last_warn_at     = NULL
            """,
            (user_id,),
        )
        await db.commit()


async def revoke_premium(user_id: int):
    """Снять премиум (удалить запись из premiums)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM premiums WHERE user_id = ?", (user_id,))
        await db.execute("DELETE FROM premium_notices WHERE user_id = ?", (user_id,))
        await db.commit()

async def get_premium_expires(user_id: int) -> str | None:
    """Вернёт ISO-дату окончания премиума или None."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT expires_at FROM premiums WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        await cur.close()
        return row[0] if row else None


async def list_expired_unnotified(now_iso: str) -> list[int]:
    """
    Вернёт user_id тех, у кого срок уже истёк (expires_at <= now_iso)
    и кому мы ещё не слали уведомление.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT p.user_id
            FROM premiums p
            LEFT JOIN premium_notices n ON n.user_id = p.user_id
            WHERE p.expires_at <= ?
              AND COALESCE(n.expired_notified, 0) = 0
            """,
            (now_iso,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [int(r[0]) for r in rows]


async def mark_expired_notified(user_id: int, when_iso: str):
    """Пометить, что уведомление об окончании отправлено."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO premium_notices(user_id, expired_notified, last_warn_at)
            VALUES (?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              expired_notified = 1,
              last_warn_at     = excluded.last_warn_at
            """,
            (user_id, when_iso),
        )
        await db.commit()

# ========= СТАТИСТИКА ДЛЯ АДМИНКИ =========

async def count_paid_users_today() -> int:
    """Сколько активаций/продлений премиума сегодня (по событиям)."""
    day = _today_str()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT COUNT(*) FROM premium_events
            WHERE substr(activated_at, 1, 10) = ?
            """,
            (day,),
        )
        n = (await cur.fetchone())[0]
        await cur.close()
        return int(n)


async def count_paid_users_total() -> int:
    """Сколько пользователей сейчас имеют активный премиум."""
    now = _utcnow_iso()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM premiums WHERE expires_at > ?",
            (now,),
        )
        n = (await cur.fetchone())[0]
        await cur.close()
        return int(n)

# ========= ДИАЛОГОВЫЕ РЕЖИМЫ И ЧАТЫ =========

async def get_chat_mode(user_id: int) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT chat_mode FROM user_prefs WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        await cur.close()
        if not row:
            await db.execute("INSERT INTO user_prefs(user_id, chat_mode, active_chat_id) VALUES (?, 'simple', NULL)", (user_id,))
            await db.commit()
            return "simple"
        return row[0] or "simple"

async def set_chat_mode(user_id: int, mode: str):
    mode = "rooms" if mode == "rooms" else "simple"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_prefs(user_id, chat_mode, active_chat_id)
            VALUES (?, ?, NULL)
            ON CONFLICT(user_id) DO UPDATE SET chat_mode = excluded.chat_mode
            """,
            (user_id, mode),
        )
        await db.commit()

async def create_chat(user_id: int, title: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chats(user_id, title, created_at) VALUES (?, ?, ?)",
            (user_id, title.strip() or "Новый чат", _utcnow_iso()),
        )
        chat_id = (await db.execute("SELECT last_insert_rowid()")).fetchone
        cur = await db.execute("SELECT last_insert_rowid()")
        row = await cur.fetchone()
        await cur.close()
        new_id = int(row[0])
        await db.commit()
        return new_id

async def list_chats(user_id: int) -> list[tuple[int, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT id, title FROM chats WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [(int(r[0]), r[1]) for r in rows]

async def set_active_chat(user_id: int, chat_id: int | None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_prefs(user_id, chat_mode, active_chat_id)
            VALUES (?, 'rooms', ?)
            ON CONFLICT(user_id) DO UPDATE SET active_chat_id = excluded.active_chat_id
            """,
            (user_id, chat_id),
        )
        await db.commit()

async def get_active_chat(user_id: int) -> int | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT active_chat_id FROM user_prefs WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row and row[0] is not None else None

async def add_chat_message(chat_id: int, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, _utcnow_iso()),
        )
        await db.commit()

async def get_chat_history(chat_id: int, limit: int = 20) -> list[tuple[str, str]]:
    """
    Возвращает список (role, content) последних сообщений, по возрастанию времени.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        rows = await cur.fetchall()
        await cur.close()
        out = [(r[0], r[1]) for r in rows][::-1]  # перевернём, чтобы было по времени
        return out

async def rename_chat(user_id: int, chat_id: int, new_title: str) -> bool:
    new_title = (new_title or "").strip()[:80] or "Без названия"
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "UPDATE chats SET title = ? WHERE id = ? AND user_id = ?",
            (new_title, chat_id, user_id),
        )
        await db.commit()
        return cur.rowcount > 0

async def delete_chat(user_id: int, chat_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        # Сначала удалим историю
        await db.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
        # Потом сам чат (с проверкой владельца)
        cur = await db.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        await db.commit()
        return cur.rowcount > 0
