import os
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta

import base64
import tempfile
from pathlib import Path
from pypdf import PdfReader  # pip install PyPDF2
from gtts import gTTS  # ← добавь импорт рядом с остальными

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import uvicorn

from openai import OpenAI
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, CallbackQueryHandler, MessageHandler,
    ContextTypes, filters,
)

# =========================
# Конфиг и клиенты
# =========================
load_dotenv()

BOT_TOKEN      = os.getenv("BOT_TOKEN", "")
OPENAI_KEY     = os.getenv("OPENAI_KEY", "")
DEEPSEEK_KEY   = os.getenv("DEEPSEEK_KEY", "")
CRYPTOPAY_KEY  = os.getenv("CRYPTOPAY_KEY", "")
REPLICATE_KEY  = os.getenv("REPLICATE_KEY", "")
ADMIN_ID       = int(os.getenv("ADMIN_ID", "0"))
PORT           = int(os.getenv("PORT", "10000"))
SUPPORT_EMAIL      = os.getenv("SUPPORT_EMAIL", "support@neurobotgpt.ru")
PUBLIC_OFFER_URL   = os.getenv("PUBLIC_OFFER_URL", "https://disk.yandex.ru/i/wdHQVfYcJGjwhw")
SUPPORT_WORK_HOURS = os.getenv("SUPPORT_WORK_HOURS", "10:00–19:00 MSK")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("neurobot")

# Модели (реальные — для движка)
MODEL_OPENAI   = "OpenAI · GPT-4o-mini"
MODEL_DEEPSEEK = "DeepSeek · Chat"
DEFAULT_MODEL  = MODEL_OPENAI

# Выбор пользователя
_user_model_visual: dict[int, str] = {}  # «название модели» которое видит пользователь
_user_model: dict[int, str] = {}         # фактический backend (OpenAI/DeepSeek)
_awaiting_img_prompt: dict[int, bool] = {}
_pending_chat_rename: dict[int, int] = {}  # user_id -> chat_id
_last_answer: dict[int, str] = {}           # последний текстовый ответ для TTS
_long_reply_queue: dict[int, list[str]] = {}  # очереди «показать ещё»

# РЕЖИМЫ (ярлыки): реально влияют на подсказку
TASK_MODES = {
    "default": {
        "label": "Стандарт",
        "system": (
            "You are a helpful, concise assistant. Prefer clear steps and short answers unless "
            "the user asks for depth."
        ),
    },
    "coding": {
        "label": "Кодинг",
        "system": (
            "You are a senior software engineer. Provide runnable code with comments, point out pitfalls, "
            "and show minimal examples. Prefer Python/JS unless the user specifies otherwise."
        ),
    },
    "seo": {
        "label": "SEO",
        "system": (
            "You are an SEO strategist. Produce keyword-rich but natural copy, suggest title/H1/meta, "
            "and include semantic clusters and internal linking ideas when useful."
        ),
    },
    "translate": {
        "label": "Перевод",
        "system": (
            "You are a professional translator (RU↔EN). Preserve meaning, tone, and idioms. "
            "If the source is ambiguous, offer the two best variants."
        ),
    },
    "summarize": {
        "label": "Резюме",
        "system": (
            "You are a world-class summarizer. Output structured bullet points, key facts, and action items. "
            "Keep it brief unless asked to expand."
        ),
    },
    "creative": {
        "label": "Креатив",
        "system": (
            "You are a creative copywriter. Offer punchy hooks, strong voice, and multiple variants when helpful. "
            "Avoid clichés."
        ),
    },
}
_user_task_mode: dict[int, str] = {}  # хранит ключ режима пользователя

# OpenAI клиент
oai = OpenAI(api_key=OPENAI_KEY)

# =========================
# DB helpers
# =========================
from db import (  # noqa
    init_db, add_user, is_premium, can_send_message, set_premium,
    get_usage_today, get_free_credits, consume_free_credit, add_free_credits,
    set_referrer_if_empty, count_paid_users_today, count_paid_users_total,
    get_premium_expires, list_expired_unnotified, mark_expired_notified,
    # новые:
    get_chat_mode, set_chat_mode, create_chat, list_chats,
    set_active_chat, get_active_chat, add_chat_message, get_chat_history,
    rename_chat, delete_chat
)

# =========================
# FastAPI & PTB
# =========================
app = FastAPI(title="NeuroBot API")
application: Application | None = None
_public_url: str | None = None
_keepalive_stop = threading.Event()

REF_BONUS   = 25
DAILY_LIMIT = 5
# --- Цены ---
PRICE_RUB = 500                 # цена для пользователя (в рублях)
PRICE_USDT = "5"                # сумма счёта для Crypto Pay (USDT), строкой как требует API
PRICE_RUB_TEXT = f"{PRICE_RUB} ₽"

# --- Диалоговые режимы ---
DIALOG_SIMPLE = "simple"  # Быстрые ответы (без памяти)
DIALOG_ROOMS  = "rooms"   # Диалоги с контекстом (чаты)

# ---------- LLM ----------
def _compose_prompt(user_id: int, user_text: str) -> list[dict]:
    """Собираем сообщения с учётом выбранного режима."""
    mode_key = _user_task_mode.get(user_id, "default")
    sys_text = TASK_MODES.get(mode_key, TASK_MODES["default"])["system"]
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_text},
    ]

def _ask_openai(user_id: int, prompt: str) -> str:
    msgs = _compose_prompt(user_id, prompt)
    r = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.7,
    )
    return r.choices[0].message.content

def _ask_deepseek(user_id: int, prompt: str) -> str:
    if not DEEPSEEK_KEY:
        return "DeepSeek недоступен: не задан DEEPSEEK_KEY."
    try:
        import httpx
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": _compose_prompt(user_id, prompt),
            "temperature": 0.7,
        }
        with httpx.Client(timeout=30) as s:
            resp = s.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                try:
                    err = resp.json()
                    msg = err.get("error", {}).get("message") or err.get("message") or str(err)
                except Exception:
                    msg = resp.text[:400]
                return f"DeepSeek API error {resp.status_code}: {msg}"
            data = resp.json()
        choice = (data or {}).get("choices", [{}])[0]
        msg = (choice or {}).get("message", {})
        text = msg.get("content") or (choice or {}).get("text") or ""
        return text or "DeepSeek вернул пустой ответ."
    except Exception as e:
        return f"Ошибка DeepSeek: {e!s}"

def ask_llm(user_id: int, prompt: str) -> str:
    real = _user_model.get(user_id, DEFAULT_MODEL)
    if real == MODEL_DEEPSEEK:
        return _ask_deepseek(user_id, prompt)
    return _ask_openai(user_id, prompt)

def ask_llm_context(user_id: int, history: list[tuple[str, str]], user_text: str) -> str:
    """
    history: список (role, content), роли: 'system' | 'user' | 'assistant'
    """
    # системное сообщение — как в обычном режиме (учитываем TASK_MODES):
    sys_text = TASK_MODES.get(_user_task_mode.get(user_id, "default"), TASK_MODES["default"])["system"]
    msgs = [{"role": "system", "content": sys_text}]
    for role, content in history:
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})

    real = _user_model.get(user_id, DEFAULT_MODEL)
    if real == MODEL_DEEPSEEK:
        # DeepSeek
        try:
            import httpx
            url = "https://api.deepseek.com/chat/completions"
            headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": msgs, "temperature": 0.7}
            with httpx.Client(timeout=30) as s:
                resp = s.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    try:
                        err = resp.json()
                        msg = err.get("error", {}).get("message") or err.get("message") or str(err)
                    except Exception:
                        msg = resp.text[:400]
                    return f"DeepSeek API error {resp.status_code}: {msg}"
                data = resp.json()
            choice = (data or {}).get("choices", [{}])[0]
            m = (choice or {}).get("message", {})
            text = m.get("content") or (choice or {}).get("text") or ""
            return text or "DeepSeek вернул пустой ответ."
        except Exception as e:
            return f"Ошибка DeepSeek: {e!s}"
    else:
        # OpenAI
        r = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.7,
        )
        return r.choices[0].message.content

async def tts_and_send(user_id: int, chat_id: int, text: str, bot):
    """
    Озвучивает text через gTTS (бесплатно) и отправляет как аудио (MP3).
    Если хочешь "voice"-сообщение (круглую плашку), позже можно
    добавить конвертацию MP3 -> OGG/OPUS через ffmpeg/pydub.
    """
    try:
        # gTTS ограничений по квоте нет; режем текст на всякий случай
        tts = gTTS(text=text[:4000], lang="ru")
        tmpdir = Path(tempfile.gettempdir())
        fpath = tmpdir / f"tts_{user_id}_{int(time.time())}.mp3"
        tts.save(str(fpath))

        # Telegram «voice» требует OGG/OPUS, поэтому шлём как обычное аудио (MP3)
        with open(fpath, "rb") as f:
            await bot.send_audio(
                chat_id=chat_id,
                audio=f,
                caption="Озвучено 🎧",
                title="TTS",
                filename=fpath.name,
            )

        try:
            fpath.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"Не вышло озвучить: {e}")

async def on_tts_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    uid = q.from_user.id
    text = _last_answer.get(uid)
    if not text:
        await q.message.reply_text("Нет текста для озвучки.")
        return
    await tts_and_send(uid, q.message.chat_id, text, context.bot)

# =========================
# Хелперы для фото/доков
# =========================
async def _download_telegram_file(bot, file_id: str) -> bytes:
    tg_file = await bot.get_file(file_id)
    bio = tempfile.NamedTemporaryFile(delete=False)
    try:
        await tg_file.download_to_drive(custom_path=bio.name)
        with open(bio.name, "rb") as f:
            return f.read()
    finally:
        try:
            Path(bio.name).unlink(missing_ok=True)
        except Exception:
            pass

def _img_b64(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

def _summarize_text_with_llm(user_id: int, title: str, text: str) -> str:
    prompt = (
        f"Мне прислали документ «{title}». Сделай короткое резюме и извлеки ключевые пункты.\n\n"
        f"Текст (обрезан до 8000 символов):\n{text[:8000]}"
    )
    return ask_llm(user_id, prompt)

def _analyze_image_with_llm(user_id: int, hint: str, image_b64: str) -> str:
    """
    hint — что хочет пользователь (если пусто — 'опиши что на фото').
    image_b64 — data:image/jpeg;base64,....
    """
    sys_text = TASK_MODES.get(_user_task_mode.get(user_id, "default"), TASK_MODES["default"])["system"]
    msgs = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": [
            {"type": "text", "text": hint or "Опиши что на фото и дай ключевые детали."},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]},
    ]
    r = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.4,
    )
    return r.choices[0].message.content

# =========================
# Длинные ответы: нарезка и "Показать ещё"
# =========================
def _split_for_telegram(text: str, limit: int = 3500) -> list[str]:
    """Режет длинные ответы по абзацам, чтобы не рвало середину текста."""
    parts, buf = [], []
    total = 0
    for para in (text or "").split("\n"):
        if total + len(para) + 1 > limit and buf:
            parts.append("\n".join(buf))
            buf, total = [], 0
        buf.append(para)
        total += len(para) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts if parts else [text]

async def on_more_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет следующую часть длинного ответа и даёт кнопки 'Показать ещё' + 'Озвучить'."""
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass

    uid = q.from_user.id
    queue = _long_reply_queue.get(uid) or []
    if not queue:
        # нечего слать — просто уберём клавиатуру
        try:
            await q.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        return

    next_part = queue.pop(0)
    _long_reply_queue[uid] = queue

    # важно: озвучиваем именно тот кусок, который сейчас показываем
    _last_answer[uid] = next_part

    # если ещё есть части — две кнопки, иначе оставим только «Озвучить»
    if queue:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Показать ещё ▶️", callback_data="more"),
             InlineKeyboardButton("🎧 Озвучить", callback_data="tts")]
        ])
    else:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🎧 Озвучить", callback_data="tts")]])

    await q.message.reply_text(next_part, reply_markup=kb)

# ---------- Images (Replicate: Flux-1 Schnell) ----------
def _replicate_generate_sync(prompt: str, width: int = 1024, height: int = 1024) -> list[str]:
    """
    Блокирующая генерация через Replicate. Возвращает список URL готовых изображений.
    """
    if not REPLICATE_KEY:
        raise RuntimeError("REPLICATE_KEY пуст — подключите ключ Replicate в .env")

    model = "black-forest-labs/flux-schnell"
    headers = {
        "Authorization": f"Token {REPLICATE_KEY}",
        "Content-Type": "application/json"
    }

    create_payload = {
        "input": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_outputs": 1,
            "go_fast": True
        }
    }

    create = requests.post(
        f"https://api.replicate.com/v1/models/{model}/predictions",
        json=create_payload,
        headers=headers,
        timeout=30
    )
    create.raise_for_status()
    prediction = create.json()
    pred_id = prediction.get("id")
    if not pred_id:
        raise RuntimeError(f"Replicate: не получили id предсказания: {prediction}")

    status = prediction.get("status")
    get_url = f"https://api.replicate.com/v1/predictions/{pred_id}"

    for _ in range(60):
        if status in ("succeeded", "failed", "canceled"):
            break
        poll = requests.get(get_url, headers=headers, timeout=15)
        poll.raise_for_status()
        prediction = poll.json()
        status = prediction.get("status")
        if status == "succeeded":
            break
        time.sleep(1)

    if status != "succeeded":
        err = prediction.get("error") or status
        raise RuntimeError(f"Replicate: задача не удалась: {err}")

    output = prediction.get("output") or []
    if isinstance(output, str):
        output = [output]
    return output


async def generate_image_and_send(user_id: int, chat_id: int, prompt: str, bot) -> None:
    try:
        urls = await asyncio.to_thread(_replicate_generate_sync, prompt)
        if not urls:
            await bot.send_message(chat_id=chat_id, text="Не удалось получить изображение.")
            return
        await bot.send_photo(chat_id=chat_id, photo=urls[0], caption="Готово ✅")
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"Ошибка генерации: {e}")
# ---------- UI ----------
def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Выбрать модель", callback_data="models")],
        [InlineKeyboardButton("🎛 Режимы", callback_data="modes")],
        [InlineKeyboardButton("💬 Диалоги", callback_data="dialog")],
        [InlineKeyboardButton("🖼️ Создать картинку", callback_data="img")],
        [InlineKeyboardButton("👤 Профиль", callback_data="profile")],
        [InlineKeyboardButton("🎁 Реферальная программа", callback_data="ref")],
        [InlineKeyboardButton("❓ Помощь", callback_data="help:how"),
         InlineKeyboardButton("📚 FAQ",    callback_data="help:faq")],
        [InlineKeyboardButton("💳 Купить подписку", callback_data="buy")],
    ])

# ===== Меню моделей =====
def _models_menu_text(mode: str = "short") -> str:
    if mode == "short":
        return (
            "<b>Кратко о моделях</b>\n"
            "• <b>GPT-5</b> — флагман для сложных задач, кодинга и длинных контекстов.\n"
            "• <b>Claude 4.5 Sonnet</b> — силён в анализе, стиле и длинных ответах.\n"
            "• <b>Gemini 2.5 Pro</b> — хороший баланс скорости и качества, мультимодальность.\n"
            "• <b>OpenAI o3</b> — логика и рассуждения, аккуратный тон.\n"
            "• <b>DeepSeek V3.2</b> — быстрые и экономные ответы, отлично для повседневки.\n"
            "• <b>OpenAI o4-mini</b> — быстрые короткие ответы и прототипирование.\n"
            "• <b>GPT-5 mini</b> — лёгкая версия для черновиков и быстрых итераций.\n"
            "• <b>GPT-4o search</b> — модель с упором на поиск/извлечение фактов.\n"
            "• <b>GPT-4o mini</b> — экономичная альтернатива для простых задач.\n"
            "• <b>Claude 3.5 Haiku</b> — очень быстро на коротких запросах.\n"
            "• <b>Gemini 2.5 Flash</b> — быстрые черновики, резюме, списки.\n\n"
            "Выберите модель для работы:"
        )
    else:
        return (
            "<b>Подробно о моделях</b>\n"
            "<b>GPT-5</b> — топ по качеству кода, сложным рассуждениям и длинным контекстам. Рекомендуется для архитектуры, аудитов, сложных SQL и интеграций.\n\n"
            "<b>Claude 4.5 Sonnet</b> — силён в языке и стиле: эссе, стратегии, юридические/деловые тексты, аккуратные объяснения. Хорош на огромных документах.\n\n"
            "<b>Gemini 2.5 Pro</b> — сбалансирован: анализ, идеи, мультимодальные задачи. Подходит для презентаций, маркетинга и быстрых исследований.\n\n"
            "<b>OpenAI o3</b> — фокус на логике/Chain-of-Thought: пошаговые решения, математика, тонкая аргументация, проверка гипотез.\n\n"
            "<b>DeepSeek V3.2</b> — очень быстрый и экономичный: повседневные вопросы, резюме, генерация простых текстов и черновики.\n\n"
            "<b>OpenAI o4-mini</b> — быстрые ответы и прототипирование: черновые спецификации, user stories, наброски кода.\n\n"
            "<b>GPT-5 mini</b> — минимальные задержки: идеи, списки, короткие подсказки, быстрые итерации.\n\n"
            "<b>GPT-4o search</b> — приоритизирует поиск и извлечение: набор фактов, цитаты, обзорные справки.\n\n"
            "<b>GPT-4o mini</b> — экономичный универсал для простых задач/переводов и быстрых советов.\n\n"
            "<b>Claude 3.5 Haiku</b> — молниеносные короткие ответы и рефакторинг текста, подсветка смысла.\n\n"
            "<b>Gemini 2.5 Flash</b> — резюме страниц, TODO-списки, короткие письма, быстрые описания.\n\n"
            "Выберите модель:"
        )

def models_keyboard_visual() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔸 Кратко",  callback_data="mvis:short"),
         InlineKeyboardButton("ℹ️ Подробно", callback_data="mvis:full")],
        [InlineKeyboardButton("Claude 3.5 Haiku", callback_data="mvis:sel:Claude 3.5 Haiku"),
         InlineKeyboardButton("✅ GPT-5",         callback_data="mvis:sel:GPT-5")],
        [InlineKeyboardButton("Claude 4.5 Sonnet", callback_data="mvis:sel:Claude 4.5 Sonnet"),
         InlineKeyboardButton("Gemini 2.5 Pro",    callback_data="mvis:sel:Gemini 2.5 Pro")],
        [InlineKeyboardButton("OpenAI o3",         callback_data="mvis:sel:OpenAI o3"),
         InlineKeyboardButton("DeepSeek V3.2",     callback_data="mvis:sel:DeepSeek V3.2")],
        [InlineKeyboardButton("OpenAI o4-mini",    callback_data="mvis:sel:OpenAI o4-mini"),
         InlineKeyboardButton("GPT-5 mini",        callback_data="mvis:sel:GPT-5 mini")],
        [InlineKeyboardButton("GPT-4o search 🔎",  callback_data="mvis:sel:GPT-4o search"),
         InlineKeyboardButton("GPT-4o mini",       callback_data="mvis:sel:GPT-4o mini")],
        [InlineKeyboardButton("Gemini 2.5 Flash",  callback_data="mvis:sel:Gemini 2.5 Flash")],
        [InlineKeyboardButton("⬅️ Назад",          callback_data="home")],
    ])

# ===== Меню режимов (ярлыки) =====
def modes_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Стандарт", callback_data="mode:default"),
         InlineKeyboardButton("Кодинг",   callback_data="mode:coding")],
        [InlineKeyboardButton("SEO",      callback_data="mode:seo"),
         InlineKeyboardButton("Перевод",  callback_data="mode:translate")],
        [InlineKeyboardButton("Резюме",   callback_data="mode:summarize"),
         InlineKeyboardButton("Креатив",  callback_data="mode:creative")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")],
    ])

def current_mode_label(user_id: int) -> str:
    key = _user_task_mode.get(user_id, "default")
    return TASK_MODES.get(key, TASK_MODES["default"])["label"]

# ===== Диалоговые режимы (simple / rooms) =====

def dialog_menu_text(mode: str) -> str:
    common_note = "\n\n<i>ℹ️ Контекстный режим работает с любой выбранной моделью.</i>"
    if mode == DIALOG_ROOMS:
        return (
            "<b>Диалоги с контекстом</b>\n"
            "Создавайте отдельные чаты по темам: история сообщений сохраняется и учитывается в ответах."
            f"{common_note}"
        )
    else:
        return (
            "<b>Быстрые ответы</b>\n"
            "Каждое сообщение — независимое. История не копится, ответы максимально быстрые."
            f"{common_note}"
        )

def dialog_keyboard(mode_now: str) -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton(
            ("✅ " if mode_now == DIALOG_SIMPLE else "") + "⚡ Быстрые ответы",
            callback_data="dialog:simple"
        )],
        [InlineKeyboardButton(
            ("✅ " if mode_now == DIALOG_ROOMS else "") + "🗂️ Диалоги с контекстом",
            callback_data="dialog:rooms"
        )],
        [InlineKeyboardButton("📂 Мои чаты", callback_data="chats")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")],
    ]
    return InlineKeyboardMarkup(kb)

async def on_dialog_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    mode = await get_chat_mode(q.from_user.id)
    await q.message.edit_text(dialog_menu_text(mode), parse_mode="HTML", reply_markup=dialog_keyboard(mode))

async def on_dialog_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    want = q.data.split("dialog:", 1)[-1]
    want = DIALOG_ROOMS if want == "rooms" else DIALOG_SIMPLE
    await set_chat_mode(q.from_user.id, want)

    # Если включили rooms и нет активного чата — создадим первый
    if want == DIALOG_ROOMS:
        active = await get_active_chat(q.from_user.id)
        if active is None:
            cid = await create_chat(q.from_user.id, "Чат 1")
            await set_active_chat(q.from_user.id, cid)

    await q.message.edit_text(dialog_menu_text(want), parse_mode="HTML", reply_markup=dialog_keyboard(want))

async def on_chats_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    user_id = q.from_user.id
    await set_chat_mode(user_id, DIALOG_ROOMS)  # при открытии списка чатов — сразу режим rooms
    chats = await list_chats(user_id)
    active = await get_active_chat(user_id)

    rows = []
    if not chats:
        rows.append([InlineKeyboardButton("➕ Создать первый чат", callback_data="chat:new")])
    else:
        for cid, title in chats[:10]:
            prefix = "✅ " if active == cid else ""
            rows.append([InlineKeyboardButton(f"{prefix}{title}", callback_data=f"chat:open:{cid}")])
        rows.append([InlineKeyboardButton("➕ Новый чат", callback_data="chat:new")])

    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="dialog")])
    await q.message.edit_text("Ваши чаты:", reply_markup=InlineKeyboardMarkup(rows))

async def on_chat_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    user_id = q.from_user.id
    chats = await list_chats(user_id)
    title = f"Чат {len(chats)+1}"
    cid = await create_chat(user_id, title)
    await set_active_chat(user_id, cid)
    await on_chats_btn(update, context)

async def on_chat_open(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    user_id = q.from_user.id
    cid = int(q.data.split("chat:open:", 1)[-1])
    await set_active_chat(user_id, cid)

    # найдём заголовок чата
    chats = await list_chats(user_id)
    title = next((t for (i, t) in chats if i == cid), f"Чат {cid}")

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✏️ Переименовать", callback_data=f"chat:rename:{cid}")],
        [InlineKeyboardButton("🗑️ Удалить",       callback_data=f"chat:delete:{cid}")],
        [InlineKeyboardButton("⬅️ К списку чатов", callback_data="chats")]
    ])
    await q.message.edit_text(f"Чат: <b>{title}</b>\nВыберите действие:", parse_mode="HTML", reply_markup=kb)

async def on_chat_rename_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    user_id = q.from_user.id
    cid = int(q.data.split("chat:rename:", 1)[-1])
    _pending_chat_rename[user_id] = cid
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Отмена", callback_data="chats")]])
    await q.message.edit_text("Отправьте новое название чата (1–80 символов):", reply_markup=kb)

async def on_chat_delete_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    cid = int(q.data.split("chat:delete:", 1)[-1])
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Да, удалить", callback_data=f"chat:delete:do:{cid}")],
        [InlineKeyboardButton("⬅️ Отмена", callback_data="chats")]
    ])
    await q.message.edit_text("Удалить этот чат? Действие необратимо.", reply_markup=kb)

async def on_chat_delete_do(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    user_id = q.from_user.id
    cid = int(q.data.split("chat:delete:do:", 1)[-1])

    # если удаляем активный — потом сбросим active_chat_id
    active = await get_active_chat(user_id)
    ok = await delete_chat(user_id, cid)
    if ok and active == cid:
        await set_active_chat(user_id, None)

    # если после удаления нет чатов — предложим создать первый
    chats = await list_chats(user_id)
    if not chats:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("➕ Создать чат", callback_data="chat:new")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="dialog")]
        ])
        await q.message.edit_text("Чат удалён. У вас пока нет чатов.", reply_markup=kb)
        return

    # иначе вернёмся к списку
    await on_chats_btn(update, context)

# =========================
# Кнопка/команда генерации изображений
# =========================
async def on_img_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass

    user_id = q.from_user.id
    if not await is_premium(user_id):
        await q.message.reply_text(
            "Доступно в Премиум.\n\n"
            "Премиум даёт:\n"
            "• Безлимитные сообщения\n"
            "• Доступ ко всем моделям\n"
            "• Генерацию изображений\n\n"
            f"Нажмите «Купить подписку», стоимость {PRICE_RUB_TEXT} на 30 дней.",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Купить подписку", callback_data="buy")]]
            )
        )
        return

    _awaiting_img_prompt[user_id] = True
    await q.message.reply_text("Опиши картинку текстом (1–2 предложения). Я сгенерирую изображение.")

async def cmd_img(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not await is_premium(user_id):
        await update.message.reply_text(
            f"Генерация изображений доступна в Премиум ({PRICE_RUB_TEXT} / 30 дней).",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Купить подписку", callback_data="buy")]]
            )
        )
        return
    _awaiting_img_prompt[user_id] = True
    await update.message.reply_text(
        "Опиши картинку текстом. Пример: «синий неоновый город, дождь, стиль киберпанк»."
    )

# =========================
# /start + рефералка
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await add_user(user.id)

    # deep-link параметр: /start ref_<tg_id>
    ref_id = None
    if context.args:
        arg = context.args[0]
        if arg.startswith("ref_"):
            try:
                ref_id = int(arg.split("ref_", 1)[1])
            except Exception:
                ref_id = None

    if ref_id and ref_id != user.id:
        try:
            first_bind = await set_referrer_if_empty(user.id, ref_id)
            if first_bind:
                await add_free_credits(ref_id, REF_BONUS)
                try:
                    await application.bot.send_message(
                        chat_id=ref_id,
                        text=f"🎉 По вашей ссылке зарегистрировался новый пользователь.\n+{REF_BONUS} бесплатных заявок!"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning("ref attach failed: %s", e)

    # НОВОЕ ПРИВЕТСТВИЕ
    text = (
        "Привет! Я <b>НейроБот 🤖</b> — твой умный помощник для текста, идей, кода и перевода.\n\n"
        f"🆓 <b>Бесплатный доступ</b> — {DAILY_LIMIT} сообщений в день + бонусы за приглашённых друзей.\n"
        "💎 <b>Премиум</b> — без ограничений, очередей и лимитов, доступ ко всем моделям и генерации изображений. "
        f"Стоимость — всего <b>{PRICE_RUB_TEXT} на 30 дней</b>.\n\n"
        "🚀 Что я умею:\n"
        "• Отвечать на вопросы и писать тексты любой сложности\n"
        "• Помогать с кодом и объяснять ошибки\n"
        "• Переводить 🇷🇺↔️🇬🇧 тексты и документы\n"
        "• Генерировать идеи, резюме, описания и письма\n"
        "• Создавать изображения по описанию 🖼️\n"
        "• 🎧 <b>Озвучивать ответы голосом</b> — нажми кнопку «Озвучить» под сообщением\n"
        "• 📄 <b>Работать с документами</b> (.txt, .md, .csv, .pdf): краткие выжимки и ключевые пункты\n"
        "• 📷 <b>Понимать фотографии/скриншоты</b>: описание и извлечение важных деталей\n\n"
        "👇 Выбирай, с чего начать:"
    )

    if update.message:
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=main_keyboard())
    else:
        await context.bot.send_message(chat_id=user.id, text=text, parse_mode="HTML", reply_markup=main_keyboard())


# =========================
# Профиль
# =========================
async def _render_profile_html(user_id: int) -> str:
    prem = await is_premium(user_id)
    used_today = await get_usage_today(user_id)
    bonus = await get_free_credits(user_id)

    me = await application.bot.get_me()
    deep_link = f"https://t.me/{me.username}?start=ref_{user_id}"
    visual = _user_model_visual.get(user_id, "GPT-4o mini")
    mode_lbl = current_mode_label(user_id)

    if prem:
        left_text = "∞ (Премиум)"
        status = "Премиум"
        # Покажем до какого числа и сколько осталось
        exp_iso = await get_premium_expires(user_id)
        extra = ""
        if exp_iso:
            try:
                exp_dt = datetime.fromisoformat(exp_iso)
            except Exception:
                exp_dt = None
            if exp_dt:
                now_dt = datetime.utcnow()
                if exp_dt.tzinfo:  # если в expires_at есть tz
                    now_dt = datetime.now(exp_dt.tzinfo)
                remaining = exp_dt - now_dt
                days_left = max(0, remaining.days + (1 if remaining.seconds > 0 else 0))
                extra = f"\nПремиум до: <b>{exp_dt.strftime('%d.%m.%Y %H:%M')}</b> (осталось ~<b>{days_left}</b> дн.)"
        status += extra
    else:
        left_day = max(0, DAILY_LIMIT - used_today)
        total_left = left_day + bonus
        left_text = f"{total_left} (дневной лимит {left_day}, бонусов {bonus})"
        status = "Обычный"

    return (
        f"👤 <b>Профиль</b>\n"
        f"ID: <code>{user_id}</code>\n"
        f"Статус: <b>{status}</b>\n"
        f"Осталось заявок: <b>{left_text}</b>\n"
        f"Модель: <b>{visual}</b>\n"
        f"Режим: <b>{mode_lbl}</b>\n\n"
        f"🔗 <b>Ваша реферальная ссылка:</b>\n{deep_link}\n\n"
        f"За каждого приглашённого: +{REF_BONUS} заявок."
    )

async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    txt = await _render_profile_html(user_id)
    await update.message.reply_text(txt, parse_mode="HTML", reply_markup=main_keyboard())

async def on_profile_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    txt = await _render_profile_html(q.from_user.id)
    try:
        await q.message.edit_text(txt, parse_mode="HTML", reply_markup=main_keyboard())
    except Exception:
        await q.message.reply_text(txt, parse_mode="HTML", reply_markup=main_keyboard())

# =========================
# Рефералка
# =========================
async def _render_referral_html(user_id: int) -> str:
    me = await application.bot.get_me()
    deep_link = f"https://t.me/{me.username}?start=ref_{user_id}"
    return (
        "🎁 <b>Реферальная программа</b>\n\n"
        f"Приглашайте друзей по ссылке и получайте <b>+{REF_BONUS}</b> бесплатных заявок за каждого!\n\n"
        f"🔗 Ваша ссылка:\n{deep_link}\n\n"
        "Как это работает:\n"
        "• Человек нажимает по ссылке и жмёт /start\n"
        f"• Вам автоматически начисляется <b>+{REF_BONUS}</b> заявок\n"
        "• Бонусы суммируются и расходуются после дневного лимита\n"
    )

async def cmd_ref(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    txt = await _render_referral_html(user_id)
    await update.message.reply_text(txt, parse_mode="HTML", reply_markup=main_keyboard())

async def on_ref_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    txt = await _render_referral_html(q.from_user.id)
    try:
        await q.message.edit_text(txt, parse_mode="HTML", reply_markup=main_keyboard())
    except Exception:
        await q.message.reply_text(txt, parse_mode="HTML", reply_markup=main_keyboard())

# =========================
# Визуальный выбор модели
# =========================
async def on_models_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    text = _models_menu_text("short")
    try:
        await q.message.edit_text(text, parse_mode="HTML", reply_markup=models_keyboard_visual())
    except Exception:
        await q.message.reply_text(text, parse_mode="HTML", reply_markup=models_keyboard_visual())

async def on_models_view_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    mode = "short" if q.data == "mvis:short" else "full"
    text = _models_menu_text(mode)
    try:
        await q.message.edit_text(text, parse_mode="HTML", reply_markup=models_keyboard_visual())
    except Exception:
        await q.message.reply_text(text, parse_mode="HTML", reply_markup=models_keyboard_visual())

async def on_model_visual_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    label = (q.data or "").split("mvis:sel:", 1)[-1].strip() or "GPT-4o mini"

    _user_model_visual[q.from_user.id] = label
    # простая логика: всё, что содержит DeepSeek — на DeepSeek, остальное — OpenAI
    if "DeepSeek" in label:
        _user_model[q.from_user.id] = MODEL_DEEPSEEK
    else:
        _user_model[q.from_user.id] = MODEL_OPENAI

    msg = f"✅ Модель «{label}» установлена.\nМожно писать сообщение!"
    try:
        await q.message.edit_text(msg, reply_markup=main_keyboard())
    except Exception:
        await q.message.reply_text(msg, reply_markup=main_keyboard())

# =========================
# Режимы (ярлыки)
# =========================
async def on_modes_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    txt = (
        "Выберите режим ответа:\n"
        "• <b>Стандарт</b> — обычные ответы\n"
        "• <b>Кодинг</b> — больше кода и примеров\n"
        "• <b>SEO</b> — тексты и структура для SEO\n"
        "• <b>Перевод</b> — RU↔EN, аккуратный стиль\n"
        "• <b>Резюме</b> — краткие выжимки\n"
        "• <b>Креатив</b> — идеи, варианты, слоганы"
    )
    try:
        await q.message.edit_text(txt, parse_mode="HTML", reply_markup=modes_keyboard())
    except Exception:
        await q.message.reply_text(txt, parse_mode="HTML", reply_markup=modes_keyboard())

async def on_mode_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    key = (q.data or "").split("mode:", 1)[-1]
    if key not in TASK_MODES:
        key = "default"
    _user_task_mode[q.from_user.id] = key
    lbl = TASK_MODES[key]["label"]
    try:
        await q.message.edit_text(f"✅ Режим «{lbl}» активирован. Готов работать!", reply_markup=main_keyboard())
    except Exception:
        await q.message.reply_text(f"✅ Режим «{lbl}» активирован. Готов работать!", reply_markup=main_keyboard())

# =========================
# Помощь / FAQ / Оферта
# =========================

def _faq_text() -> str:
    return (
        "<b>FAQ — Частые вопросы</b>\n\n"
        "• <b>Что даёт Премиум?</b>\n"
        "  Безлимитные сообщения, доступ ко всем моделям и генерации изображений.\n\n"
        "• <b>Сколько стоит Премиум и на сколько дней?</b>\n"
        f"  {PRICE_RUB_TEXT} за 30 дней. Оплатить можно через кнопку «Купить подписку».\n\n"
        "• <b>Где посмотреть, когда закончится Премиум?</b>\n"
        "  Откройте «👤 Профиль» — там дата окончания и оставшиеся дни.\n\n"
        "• <b>Как работают лимиты без Премиум?</b>\n"
        f"  {DAILY_LIMIT}/день + реферальные бонусы.\n\n"
        "• <b>Как получить бонусы?</b>\n"
        "  Пригласите друзей по вашей реферальной ссылке из Профиля — за каждого +25 заявок.\n\n"
        "• <b>Могу ли я озвучить ответы бота?</b>\n"
        "  Да, просто нажмите кнопку «🎧 Озвучить» под любым сообщением.\n\n"
        "• <b>Можно ли отправлять документы?</b>\n"
        "  Да, бот поддерживает .txt, .md, .csv и .pdf — он сделает краткое резюме и выделит ключевые пункты.\n\n"
        "• <b>Можно ли анализировать фото?</b>\n"
        "  Да, просто отправьте изображение или скриншот — бот опишет, что на нём, и выделит детали.\n\n"
        "• <b>Возвраты и вопросы по оплатам</b>\n"
        f"  Пишите на <a href='mailto:{SUPPORT_EMAIL}'>{SUPPORT_EMAIL}</a> — поможем. "
        "Возврат средств возможен в случаях, предусмотренных законодательством РФ и условиями нашей оферты.\n\n"
        "• <b>Конфиденциальность</b>\n"
        "  Мы обрабатываем персональные данные по ФЗ-152 и используем их только для оказания услуг.\n"
    )

def _support_text() -> str:
    return (
        "<b>Техподдержка</b>\n\n"
        f"• Email: <a href='mailto:{SUPPORT_EMAIL}'>{SUPPORT_EMAIL}</a>\n"
        f"• Время ответа: в рабочие часы {SUPPORT_WORK_HOURS}\n\n"
        "В письме укажите: ID в Telegram (из Профиля), кратко суть вопроса, скрин/ошибку и время события."
    )

async def on_help_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📚 FAQ", callback_data="help:faq"),
         InlineKeyboardButton("🛟 Техподдержка", callback_data="help:support")],
        [InlineKeyboardButton("📄 Публичная оферта", url=PUBLIC_OFFER_URL)],
        [InlineKeyboardButton("✉️ Написать на email", url=f"mailto:{SUPPORT_EMAIL}")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")]
    ])
    await q.message.edit_text("Раздел помощи. Выберите:", reply_markup=kb)

async def on_help_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🛟 Техподдержка", callback_data="help:support")],
        [InlineKeyboardButton("📄 Публичная оферта", url=PUBLIC_OFFER_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")]
    ])
    await q.message.edit_text(_faq_text(), parse_mode="HTML", reply_markup=kb)

async def on_help_support(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✉️ Написать на email", url=f"mailto:{SUPPORT_EMAIL}")],
        [InlineKeyboardButton("📚 FAQ", callback_data="help:faq")],
        [InlineKeyboardButton("📄 Публичная оферта", url=PUBLIC_OFFER_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")]
    ])
    await q.message.edit_text(_support_text(), parse_mode="HTML", reply_markup=kb)

async def on_help_how(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Открывает тот же текст, что и /help, но по кнопке."""
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass

    txt = (
        "<b>Как пользоваться ботом</b>\n\n"
        "• Напишите любое сообщение — я отвечу.\n"
        "• 🎧 Озвучивай ответы: нажми кнопку «Озвучить» под сообщением.\n"
        "• 📄 Отправляй документы (.txt, .md, .csv, .pdf) — сделаю краткое резюме.\n"
        "• 📷 Присылай фото или скриншоты — опишу, что на них.\n"
        "• Нужна картинка? Команда /img.\n"
        "• Выбор модели — /models.\n"
        "• Переключить режим — /mode.\n"
        "• Профиль и рефералка — /profile, /ref.\n"
       f"• Премиум ({PRICE_RUB_TEXT} / 30 дней) — /buy.\n\n"
        "Кнопки ниже — быстрый доступ:"
    )

    # мини-меню помощи с быстрыми ссылками
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📚 FAQ",          callback_data="help:faq"),
         InlineKeyboardButton("🛟 Техподдержка", callback_data="help:support")],
        [InlineKeyboardButton("📄 Публичная оферта", url=PUBLIC_OFFER_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")]
    ])
    await q.message.edit_text(txt, parse_mode="HTML", reply_markup=kb)

# Команды-псевдонимы
async def cmd_support(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_support_text(), parse_mode="HTML", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("✉️ Написать на email", url=f"mailto:{SUPPORT_EMAIL}")],
        [InlineKeyboardButton("📚 FAQ", callback_data="help:faq")],
        [InlineKeyboardButton("📄 Публичная оферта", url=PUBLIC_OFFER_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")]
    ]))

async def cmd_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_faq_text(), parse_mode="HTML", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("🛟 Техподдержка", callback_data="help:support")],
        [InlineKeyboardButton("📄 Публичная оферта", url=PUBLIC_OFFER_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")]
    ]))

# =========================
# Оплата (CryptoPay)
# =========================
async def on_buy_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass

    if not CRYPTOPAY_KEY:
        await q.message.reply_text("Оплата не подключена (нет CRYPTOPAY_KEY).")
        return

    payload = str(q.from_user.id)
    headers = {"Crypto-Pay-API-Token": CRYPTOPAY_KEY}
    data = {
        "asset": "USDT",
        "amount": PRICE_USDT,
        "description": f"Подписка на 30 дней ({PRICE_RUB_TEXT})",
        "payload": payload,
    }
    r = requests.post("https://pay.crypt.bot/api/createInvoice", json=data, headers=headers, timeout=15)
    j = r.json()
    url = j["result"]["pay_url"]

    text = (
    f"Оплата подписки на 30 дней: <b>{PRICE_RUB_TEXT}</b>\n\n"
    "<b>Премиум даёт</b>:\n"
    "• Безлимитные сообщения (без очередей)\n"
    "• Доступ ко <b>всем</b> моделям\n"
    "• <b>Генерацию изображений</b> (Replicate · Flux-1 Schnell)\n"
    "• Приоритетную обработку\n\n"
    f"Ссылка на оплату:\n{url}"
    )
    await q.message.reply_text(text, parse_mode="HTML")

# =========================
# Команды /buy /models /mode /help
# =========================
async def cmd_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /buy — та же логика, что и по кнопке."""
    if not CRYPTOPAY_KEY:
        await update.message.reply_text("Оплата не подключена (нет CRYPTOPAY_KEY).")
        return

    payload = str(update.effective_user.id)
    headers = {"Crypto-Pay-API-Token": CRYPTOPAY_KEY}
    data = {
        "asset": "USDT",
        "amount": PRICE_USDT,
        "description": f"Подписка на 30 дней ({PRICE_RUB_TEXT})",
        "payload": payload,
    }
    r = requests.post("https://pay.crypt.bot/api/createInvoice", json=data, headers=headers, timeout=15)
    j = r.json()
    url = j["result"]["pay_url"]

    text = (
    f"Оплата подписки на 30 дней: <b>{PRICE_RUB_TEXT}</b>\n\n"
    "<b>Премиум даёт</b>:\n"
    "• Безлимитные сообщения (без очередей)\n"
    "• Доступ ко <b>всем</b> моделям\n"
    "• <b>Генерацию изображений</b> (Replicate · Flux-1 Schnell)\n"
    "• Приоритетную обработку\n\n"
    f"Ссылка на оплату:\n{url}"
    )
    await update.message.reply_text(text, parse_mode="HTML")

async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /models — показать меню выбора модели."""
    text = _models_menu_text("short")
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=models_keyboard_visual())

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /mode — показать меню режимов."""
    txt = (
        "Выберите режим ответа:\n"
        "• <b>Стандарт</b> — обычные ответы\n"
        "• <b>Кодинг</b> — больше кода и примеров\n"
        "• <b>SEO</b> — тексты и структура для SEO\n"
        "• <b>Перевод</b> — RU↔EN, аккуратный стиль\n"
        "• <b>Резюме</b> — краткие выжимки\n"
        "• <b>Креатив</b> — идеи, варианты, слоганы"
    )
    await update.message.reply_text(txt, parse_mode="HTML", reply_markup=modes_keyboard())

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help — краткая справка с основными действиями."""
    txt = (
        "<b>Как пользоваться ботом</b>\n\n"
        "• Напишите любое сообщение — я отвечу.\n"
        "• 🎧 Озвучивай ответы: нажми кнопку «Озвучить» под сообщением.\n"
        "• 📄 Отправляй документы (.txt, .md, .csv, .pdf) — я сделаю краткое резюме.\n"
        "• 📷 Присылай фото или скриншоты — расскажу, что на них.\n"
        "• Нужна картинка? Команда /img.\n"
        "• Выбор модели — /models.\n"
        "• Переключить режим — /mode.\n"
        "• Профиль и рефералка — /profile, /ref.\n"
       f"• Премиум ({PRICE_RUB_TEXT} / 30 дней) — /buy.\n\n"
        "Кнопки ниже — быстрый доступ:"
    )
    await update.message.reply_text(txt, parse_mode="HTML", reply_markup=main_keyboard())
# =========================
# Сообщения пользователей
# =========================
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""

    # Переименование чата — если ждём от пользователя новое имя
    if _pending_chat_rename.get(user_id):
        cid = _pending_chat_rename[user_id]
        new_title = (text or "").strip()[:80]
        if not new_title:
            await update.message.reply_text(
                "Название пустое. Отправьте текст от 1 до 80 символов или нажмите «Отмена» в меню."
            )
            return
        ok = await rename_chat(user_id, cid, new_title)
        _pending_chat_rename.pop(user_id, None)
        if ok:
            await update.message.reply_text("Готово: чат переименован ✅")
        else:
            await update.message.reply_text("Не удалось переименовать чат.")
        return

    # Если ждём промпт для изображения — перехватываем
    if _awaiting_img_prompt.get(user_id):
        _awaiting_img_prompt[user_id] = False
        if not await is_premium(user_id):
            await update.message.reply_text(
                "Генерация изображений только для Премиум.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Купить подписку", callback_data="buy")]]
                ),
            )
            return
        await update.message.reply_text("Генерирую изображение…")
        await generate_image_and_send(user_id, update.effective_chat.id, text, context.bot)
        return

    # лимиты как раньше
    if not await is_premium(user_id):
        if await can_send_message(user_id, limit=DAILY_LIMIT):
            pass
        elif await consume_free_credit(user_id):
            pass
        else:
            await update.message.reply_text(
                "🚫 Лимит исчерпан.\n"
                f"— Дневной лимит: {DAILY_LIMIT}/день\n"
                f"— Реферальные бонусы: получите +{REF_BONUS} заявок за каждого приглашённого!\n\n"
                "Купите подписку «💳 Купить подписку» для безлимита."
            )
            return

    # выбор по диалоговому режиму
    mode = await get_chat_mode(user_id)

    # ➊ Поставим временное сообщение
    spinner = await update.message.reply_text("🤖 Генерация ответа…")

    try:
        if mode == DIALOG_ROOMS:
            # нужен активный чат, если нет — создадим
            cid = await get_active_chat(user_id)
            if cid is None:
                cid = await create_chat(user_id, "Чат 1")
                await set_active_chat(user_id, cid)

            # загрузим историю (последние 20 сообщений) + добавим текущий запрос
            history = await get_chat_history(cid, limit=20)
            reply = ask_llm_context(user_id, history, text)

            # сохраним и вопрос, и ответ в историю
            await add_chat_message(cid, "user", text)
            await add_chat_message(cid, "assistant", reply)
        else:
            # быстрый режим
            reply = ask_llm(user_id, text)
    finally:
        # ➋ Удаляем «спиннер» в любом исходе
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=spinner.message_id,
            )
        except Exception:
            pass

    # ➌ Отправляем ответ (кнопки как обсуждали)
    _last_answer[user_id] = reply
    parts = _split_for_telegram(reply)
    if len(parts) == 1:
        _last_answer[user_id] = parts[0]
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🎧 Озвучить", callback_data="tts")]])
        await update.message.reply_text(parts[0], reply_markup=kb)
    else:
        _long_reply_queue[user_id] = parts[1:]
        _last_answer[user_id] = parts[0]
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Показать ещё ▶️", callback_data="more"),
                    InlineKeyboardButton("🎧 Озвучить", callback_data="tts"),
                ]
            ]
        )
        await update.message.reply_text(parts[0], reply_markup=kb)
    return


# =========================
# Обработчик фото
# =========================
async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    # лимиты как в on_message
    if not await is_premium(user_id):
        if await can_send_message(user_id, limit=DAILY_LIMIT):
            pass
        elif await consume_free_credit(user_id):
            pass
        else:
            await update.message.reply_text(
                "🚫 Лимит исчерпан.\n"
                f"— Дневной лимит: {DAILY_LIMIT}/день\n"
                f"— Реферальные бонусы: получите +{REF_BONUS} заявок за каждого приглашённого!\n\n"
                "Купите подписку «💳 Купить подписку» для безлимита."
            )
            return

    spinner = await update.message.reply_text("🤖 Генерация ответа…")
    try:
        # берём самую большую превьюху
        photo = update.message.photo[-1]
        data = await _download_telegram_file(context.bot, photo.file_id)
        img64 = _img_b64(data)
        # если у сообщения есть подпись — используем как hint
        hint = update.message.caption or ""
        reply = _analyze_image_with_llm(user_id, hint, img64)
    except Exception as e:
        await update.message.reply_text(f"Не удалось проанализировать изображение: {e}")
        return
    finally:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=spinner.message_id,
            )
        except Exception:
            pass

    _last_answer[user_id] = reply
    chunks = _split_for_telegram(reply)
    if len(chunks) > 1:
        _long_reply_queue[user_id] = chunks[1:]
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Показать ещё ▶️", callback_data="more"),
                    InlineKeyboardButton("🎧 Озвучить", callback_data="tts"),
                ]
            ]
        )
        await update.message.reply_text(chunks[0], reply_markup=kb)
    else:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🎧 Озвучить", callback_data="tts")]])
        await update.message.reply_text(chunks[0], reply_markup=kb)
        
# =========================
# Обработчик документов (.txt/.md/.csv/.pdf)
# =========================
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    # лимиты
    if not await is_premium(user_id):
        if await can_send_message(user_id, limit=DAILY_LIMIT):
            pass
        elif await consume_free_credit(user_id):
            pass
        else:
            await update.message.reply_text(
                "🚫 Лимит исчерпан.\n"
                f"— Дневной лимит: {DAILY_LIMIT}/день\n"
                f"— Реферальные бонусы: получите +{REF_BONUS} заявок!\n\n"
                "Купите подписку «💳 Купить подписку» для безлимита."
            )
            return

    doc = update.message.document
    title = doc.file_name or "документ"
    spinner = await update.message.reply_text("🤖 Генерация ответа…")
    try:
        data = await _download_telegram_file(context.bot, doc.file_id)
        text_content = ""
        lower = (title or "").lower()

        if lower.endswith((".txt", ".md", ".csv")):
            # простые текстовые — читаем как utf-8
            text_content = data.decode("utf-8", errors="replace")
        elif lower.endswith(".pdf"):
            import io
            reader = PdfReader(io.BytesIO(data))
            pages = min(10, len(reader.pages))  # не больше 10 страниц
            chunks = []
            for i in range(pages):
                try:
                    chunks.append(reader.pages[i].extract_text() or "")
                except Exception:
                    pass
            text_content = "\n\n".join(chunks).strip()
            if not text_content:
                text_content = "[Не удалось извлечь текст из PDF. Попробуйте отправить как изображение/скриншот.]"
        else:
            await update.message.reply_text("Поддерживаю пока .txt, .md, .csv и .pdf. Попробуйте один из этих форматов.")
            return

        reply = _summarize_text_with_llm(user_id, title, text_content)
        try:
            await context.bot.delete_message(chat_id=update.effective_chat.id,
                                            message_id=spinner.message_id)
        except Exception:
            pass
        _last_answer[user_id] = reply
        chunks = _split_for_telegram(reply)
        if len(chunks) > 1:
            _long_reply_queue[user_id] = chunks[1:]
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Показать ещё ▶️", callback_data="more"),
                 InlineKeyboardButton("🎧 Озвучить", callback_data="tts")]
            ])
            await update.message.reply_text(chunks[0], reply_markup=kb)
        else:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🎧 Озвучить", callback_data="tts")]])
            await update.message.reply_text(chunks[0], reply_markup=kb)

    except Exception as e:
        await update.message.reply_text(f"Не удалось обработать документ: {e}")

# =========================
# Админ-команды
# =========================
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return
    paid_today = await count_paid_users_today()
    paid_total = await count_paid_users_total()
    await update.message.reply_text(
        "📊 Админ-панель\n"
        f"Покупок сегодня: {paid_today}\n"
        f"Всего активных премиумов: {paid_total}\n\n"
        "Команды:\n"
        "/add_premium <user_id> <days>\n"
        "/remove_premium <user_id>\n"
        "/broadcast <text>"
    )

async def cmd_add_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    if len(context.args) < 2:
        await update.message.reply_text("Формат: /add_premium <user_id> <days>")
        return
    try:
        uid = int(context.args[0])
        days = int(context.args[1])
        expires_at = (datetime.now() + timedelta(days=days)).isoformat()
        await set_premium(uid, expires_at)
        await update.message.reply_text(f"✅ Премиум выдан {uid} на {days} дн.")
        try:
            await application.bot.send_message(uid, f"🎉 Вам выдали премиум на {days} дней!")
        except Exception:
            pass
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def cmd_remove_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    try:
        if not context.args:
            await update.message.reply_text("Формат: /remove_pремиum <user_id>")
            return
        uid = int(context.args[0])
        await set_premium(uid, (datetime.now() - timedelta(days=1)).isoformat())
        await update.message.reply_text(f"❎ Премиум снят у {uid}.")
        try:
            await application.bot.send_message(uid, "⚠️ Ваш премиум был отключён.")
        except Exception:
            pass
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    if not context.args:
        await update.message.reply_text("Формат: /broadcast <text>")
        return
    text = " ".join(context.args)
    await update.message.reply_text(f"Ок, отправлю: {text}\n(реальную рассылку можно дописать в db.py)")

# =========================
# Webhooks
# =========================
@app.post("/tg")
async def telegram_webhook(request: Request):
    global application
    if application is None:
        return {"ok": False, "error": "bot not initialized"}
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}

@app.post("/cryptopay-webhook")
async def cryptopay_webhook(request: Request):
    """Обработчик вебхуков Crypto Pay (update_type=invoice_paid)."""
    global application
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "bad json"}

    try:
        logger.info("CryptoPay webhook: %s", data)
    except Exception:
        pass

    user_id = None
    paid = False

    # Новый формат
    update_type = data.get("update_type")
    inv_new = data.get("payload") or {}
    if update_type == "invoice_paid" and isinstance(inv_new, dict):
        raw_uid = inv_new.get("payload")
        status_new = inv_new.get("status")
        if raw_uid is not None and (status_new is None or status_new == "paid"):
            try:
                user_id = int(str(raw_uid))
                paid = True
            except Exception:
                user_id = None

    # Совместимость со старым форматом
    if not paid:
        invoice = data.get("invoice") or {}
        status = invoice.get("status")
        raw_uid = invoice.get("payload")
        if status == "paid" and raw_uid is not None:
            try:
                user_id = int(str(raw_uid))
                paid = True
            except Exception:
                user_id = None

    if paid and user_id:
        expires_dt = datetime.now() + timedelta(days=30)
        await set_premium(user_id, expires_dt.isoformat())
        try:
            text = (
                "✅ <b>Оплата получена</b>!\n"
                f"Премиум активирован до <b>{expires_dt.strftime('%d.%m.%Y')}</b>.\n\n"
                "Что дальше?\n"
                "• Откройте профиль — проверить статус и реф. ссылку\n"
                "• Выберите модель — переключиться на нужный режим\n"
                "• Или просто напишите сообщение 🙂"
            )
            await application.bot.send_message(
                chat_id=user_id,
                text=text,
                parse_mode="HTML",
                reply_markup=main_keyboard()
            )
        except Exception:
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text="✅ Оплата получена! Премиум активирован на 30 дней."
                )
            except Exception:
                pass

    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# =========================
# Keep-alive (40s) + авто-починка вебхука
# =========================
def _keepalive_loop():
    if not _public_url:
        return
    url = f"{_public_url.rstrip('/')}/health"
    session = requests.Session()
    while not _keepalive_stop.wait(40):
        try:
            session.get(url, timeout=8)
        except Exception:
            pass

async def _webhook_guard_loop():
    """Раз в 10 минут проверяем webhook и чиним, если он слетел."""
    await asyncio.sleep(8)
    while True:
        try:
            bot = application.bot
            _ = await bot.get_me()
            info = await bot.get_webhook_info()
            needed = f"{_public_url.rstrip('/')}/tg"
            if info.url != needed:
                try:
                    await bot.set_webhook(needed, max_connections=40, drop_pending_updates=False)
                    logger.info("🔧 Webhook восстановлен: %s", needed)
                except Exception as e:
                    logger.warning("Webhook repair failed: %s", e)
        except Exception as e:
            logger.warning("webhook guard error: %s", e)
        await asyncio.sleep(600)  # 10 минут

async def _premium_expiry_notifier_loop():
    """Раз в 15 минут ищем истёкшие премиумы и шлём 1 уведомление."""
    await asyncio.sleep(10)
    while True:
        try:
            now_iso = datetime.utcnow().isoformat()
            user_ids = await list_expired_unnotified(now_iso)
            for uid in user_ids:
                # отправим уведомление
                try:
                    await application.bot.send_message(
                        chat_id=uid,
                        text=(
                            "⛔️ Ваш премиум закончился.\n\n"
                            "Продлите подписку, чтобы сохранить безлимит, доступ ко всем моделям "
                            "и генерацию изображений."
                        ),
                        reply_markup=InlineKeyboardMarkup(
                            [[InlineKeyboardButton("💳 Купить подписку", callback_data="buy")]]
                        )
                    )
                except Exception:
                    pass
                # пометить, что уведомили
                try:
                    await mark_expired_notified(uid, now_iso)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("premium notifier error: %s", e)
        await asyncio.sleep(900)  # 15 минут

# =========================
# Глобальный error-handler PTB (чтобы не падал на 400)
# =========================
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.warning("PTB error: %s", getattr(context, "error", None))

# =========================
# Инициализация
# =========================
def build_application() -> Application:
    app_ = ApplicationBuilder().token(BOT_TOKEN).build()

    # команды
    app_.add_handler(CommandHandler("start",   cmd_start))
    app_.add_handler(CommandHandler("profile", cmd_profile))
    app_.add_handler(CommandHandler("ref",     cmd_ref))
    app_.add_handler(CommandHandler("admin",   cmd_admin))
    app_.add_handler(CommandHandler("add_premium",    cmd_add_premium))
    app_.add_handler(CommandHandler("remove_premium", cmd_remove_premium))
    app_.add_handler(CommandHandler("broadcast",      cmd_broadcast))
    app_.add_handler(CommandHandler("buy",    cmd_buy))
    app_.add_handler(CommandHandler("models", cmd_models))
    app_.add_handler(CommandHandler("mode",   cmd_mode))
    app_.add_handler(CommandHandler("help",   cmd_help))
    app_.add_handler(CommandHandler("support", cmd_support))
    app_.add_handler(CommandHandler("faq",     cmd_faq))

    # кнопка/команда генерации изображений
    app_.add_handler(CallbackQueryHandler(on_img_btn, pattern=r"^img$"))
    app_.add_handler(CommandHandler("img", cmd_img))
    app_.add_handler(CallbackQueryHandler(on_tts_btn, pattern=r"^tts$"))
    app_.add_handler(CallbackQueryHandler(on_more_btn, pattern=r"^more$"))

    # кнопки
    app_.add_handler(CallbackQueryHandler(on_buy_btn,      pattern=r"^buy$"))
    app_.add_handler(CallbackQueryHandler(on_profile_btn,  pattern=r"^profile$"))
    app_.add_handler(CallbackQueryHandler(on_ref_btn,      pattern=r"^ref$"))
    app_.add_handler(CallbackQueryHandler(on_models_btn,   pattern=r"^models$"))
    app_.add_handler(CallbackQueryHandler(on_models_view_toggle, pattern=r"^mvis:(short|full)$"))
    app_.add_handler(CallbackQueryHandler(on_model_visual_select, pattern=r"^mvis:sel:.+$"))
    app_.add_handler(CallbackQueryHandler(on_modes_btn,    pattern=r"^modes$"))
    app_.add_handler(CallbackQueryHandler(on_mode_select,  pattern=r"^mode:(default|coding|seo|translate|summarize|creative)$"))
    app_.add_handler(CallbackQueryHandler(
        lambda u, c: u.callback_query.message.edit_text("Главное меню:", reply_markup=main_keyboard()),
        pattern=r"^home$"
    ))
   
    # помощь / faq
    app_.add_handler(CallbackQueryHandler(on_help_btn,     pattern=r"^help$"))
    app_.add_handler(CallbackQueryHandler(on_help_how,     pattern=r"^help:how$"))   # ← NEW
    app_.add_handler(CallbackQueryHandler(on_help_faq,     pattern=r"^help:faq$"))
    app_.add_handler(CallbackQueryHandler(on_help_support, pattern=r"^help:support$"))

    # сообщения
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    # вложения
    app_.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, on_photo))
    app_.add_handler(MessageHandler(filters.Document.ALL & ~filters.COMMAND, on_document))
    
    # диалоговые режимы и чаты
    app_.add_handler(CallbackQueryHandler(on_dialog_btn,    pattern=r"^dialog$"))
    app_.add_handler(CallbackQueryHandler(on_dialog_select, pattern=r"^dialog:(simple|rooms)$"))
    app_.add_handler(CallbackQueryHandler(on_chats_btn,     pattern=r"^chats$"))
    app_.add_handler(CallbackQueryHandler(on_chat_new,      pattern=r"^chat:new$"))
    app_.add_handler(CallbackQueryHandler(on_chat_open,     pattern=r"^chat:open:\d+$"))
    app_.add_handler(CallbackQueryHandler(on_chat_rename_ask,   pattern=r"^chat:rename:\d+$"))
    app_.add_handler(CallbackQueryHandler(on_chat_delete_confirm, pattern=r"^chat:delete:\d+$"))
    app_.add_handler(CallbackQueryHandler(on_chat_delete_do,    pattern=r"^chat:delete:do:\d+$"))

    # error-handler
    app_.add_error_handler(on_error)

    return app_

@app.on_event("startup")
async def on_startup():
    global application, _public_url
    await init_db()

    application = build_application()
    await application.initialize()
    await application.start()

    _public_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    if not _public_url:
        raise RuntimeError("Не найден PUBLIC_URL/RENDER_EXTERNAL_URL")

    webhook_url = f"{_public_url.rstrip('/')}/tg"
    await application.bot.set_webhook(webhook_url, max_connections=40, drop_pending_updates=False)
    logger.info("✅ Установлен Telegram webhook: %s", webhook_url)

    threading.Thread(target=_keepalive_loop, daemon=True).start()
    asyncio.get_event_loop().create_task(_webhook_guard_loop())
    asyncio.get_event_loop().create_task(_premium_expiry_notifier_loop())

    logger.info("🚀 Startup complete. Listening on port %s", PORT)

@app.on_event("shutdown")
async def on_shutdown():
    _keepalive_stop.set()
    try:
        if application is not None:
            # ВАЖНО: НЕ снимаем webhook — иначе Telegram перестанет будить Render!
            await application.stop()
            await application.shutdown()
    finally:
        logger.info("🛑 Shutdown complete")



# =========================
# Запуск
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
