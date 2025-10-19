import os
import logging
import asyncio
import threading
from datetime import datetime, timedelta

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
# Конфиг
# =========================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY", "")
CRYPTOPAY_KEY = os.getenv("CRYPTOPAY_KEY", "")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("neurobot")

# =========================
# Модели (визуальные)
# =========================
MODEL_OPENAI = "OpenAI · GPT-4o-mini"
MODEL_DEEPSEEK = "DeepSeek · Chat"
DEFAULT_MODEL = MODEL_OPENAI

_user_model_visual: dict[int, str] = {}
_user_model: dict[int, str] = {}

# =========================
# Клиент OpenAI
# =========================
oai = OpenAI(api_key=OPENAI_KEY)

# =========================
# DB helpers
# =========================
from db import (
    init_db, add_user, is_premium, can_send_message, set_premium,
    get_usage_today, get_free_credits, consume_free_credit, add_free_credits,
    set_referrer_if_empty, count_paid_users_today, count_paid_users_total
)

# =========================
# FastAPI & PTB
# =========================
app = FastAPI(title="NeuroBot API")
application: Application | None = None
_public_url: str | None = None
_keepalive_stop = threading.Event()

REF_BONUS = 25
DAILY_LIMIT = 5

# =========================
# LLM
# =========================
def ask_llm(user_id: int, prompt: str) -> str:
    r = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return r.choices[0].message.content


# =========================
# UI
# =========================
def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Выбрать модель", callback_data="models")],
        [InlineKeyboardButton("👤 Профиль", callback_data="profile")],
        [InlineKeyboardButton("🎁 Реферальная программа", callback_data="ref")],
        [InlineKeyboardButton("💳 Купить подписку", callback_data="buy")],
    ])

# ---------- модели ----------
def _models_text_short() -> str:
    return (
        "🔹 <b>Claude 4.5 Sonnet</b> — умный и точный\n"
        "🔸 <b>GPT-5</b> — универсальный и надёжный\n"
        "⚡ <b>Gemini 2.5 Pro</b> — быстрая и логичная\n"
        "🧩 <b>OpenAI o3</b> — оптимален для кода\n"
        "🧠 <b>DeepSeek V3.2</b> — аналитика и факты\n"
        "🚀 <b>GPT-4o mini</b> — лёгкий и мгновенный\n\n"
        "Выберите модель ниже 👇"
    )

def _models_text_full() -> str:
    return (
        "<b>📈 Подробное описание моделей</b>\n\n"
        "🔹 <b>Claude 4.5 Sonnet</b>\n"
        "Интеллектуальная модель с живым стилем речи. Отлично пишет тексты, сценарии, статьи.\n\n"
        "🔸 <b>GPT-5</b>\n"
        "Главная универсальная модель. Лучшая точность, креативность и стабильность. "
        "Подходит для любых задач — от кода до эссе.\n\n"
        "⚡ <b>Gemini 2.5 Pro</b>\n"
        "Отличается скоростью и логичностью. Прекрасно справляется с фактологическими вопросами, короткими ответами и анализом данных.\n\n"
        "🧩 <b>OpenAI o3</b>\n"
        "Точен в технических вопросах и коде. Отлично понимает контекст программирования и API.\n\n"
        "🧠 <b>DeepSeek V3.2</b>\n"
        "Сосредоточен на анализе, вычислениях и структурированных ответах. Лаконичен и строг.\n\n"
        "🚀 <b>GPT-4o mini</b>\n"
        "Самый быстрый и лёгкий вариант для повседневных запросов, чатов и генерации идей.\n\n"
        "Выберите любую модель для начала 👇"
    )

def models_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Кратко", callback_data="models_short"),
         InlineKeyboardButton("📈 Подробнее", callback_data="models_full")],
        [InlineKeyboardButton("Claude 4.5 Sonnet", callback_data="sel:Claude 4.5 Sonnet")],
        [InlineKeyboardButton("GPT-5", callback_data="sel:GPT-5")],
        [InlineKeyboardButton("Gemini 2.5 Pro", callback_data="sel:Gemini 2.5 Pro")],
        [InlineKeyboardButton("OpenAI o3", callback_data="sel:OpenAI o3")],
        [InlineKeyboardButton("DeepSeek V3.2", callback_data="sel:DeepSeek V3.2")],
        [InlineKeyboardButton("GPT-4o mini", callback_data="sel:GPT-4o mini")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="home")],
    ])

# ---------- /start ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await add_user(user.id)

    # проверка рефералки
    ref_id = None
    if context.args and context.args[0].startswith("ref_"):
        try:
            ref_id = int(context.args[0].split("ref_")[1])
        except Exception:
            pass
    if ref_id and ref_id != user.id:
        try:
            first_bind = await set_referrer_if_empty(user.id, ref_id)
            if first_bind:
                await add_free_credits(ref_id, REF_BONUS)
                await application.bot.send_message(
                    chat_id=ref_id,
                    text=f"🎉 По вашей ссылке зарегистрировался новый пользователь.\n+{REF_BONUS} бесплатных заявок!"
                )
        except Exception as e:
            logger.warning("ref attach failed: %s", e)

    text = (
        "Привет! Я нейробот 🤖\n\n"
        f"Бесплатно: {DAILY_LIMIT} сообщений в день (+ реф. бонусы).\n"
        "Премиум — без ограничений и очередей.\n\n"
        "Выбирай действие ниже 👇"
    )
    await update.message.reply_text(text, reply_markup=main_keyboard())

# ---------- профиль ----------
async def _render_profile_html(user_id: int) -> str:
    prem = await is_premium(user_id)
    used_today = await get_usage_today(user_id)
    bonus = await get_free_credits(user_id)
    me = await application.bot.get_me()
    deep_link = f"https://t.me/{me.username}?start=ref_{user_id}"

    if prem:
        status = "Премиум"
        left_text = "∞ (безлимит)"
        async with await aiosqlite.connect(os.getenv("DATABASE_PATH", "/tmp/neurobot.sqlite3")) as db:
            cur = await db.execute("SELECT expires_at FROM premiums WHERE user_id=?", (user_id,))
            row = await cur.fetchone()
            expires = row[0][:10] if row else "-"
    else:
        status = "Обычный"
        left_day = max(0, DAILY_LIMIT - used_today)
        total_left = left_day + bonus
        left_text = f"{total_left} (дневной {left_day} + бонусов {bonus})"
        expires = "-"

    visual = _user_model_visual.get(user_id, "GPT-4o mini")

    return (
        f"👤 <b>Профиль</b>\n"
        f"ID: <code>{user_id}</code>\n"
        f"Статус: <b>{status}</b>\n"
        f"Активен до: <b>{expires}</b>\n"
        f"Осталось заявок: <b>{left_text}</b>\n"
        f"Модель: <b>{visual}</b>\n\n"
        f"🔗 Ваша реферальная ссылка:\n{deep_link}\n"
        f"+{REF_BONUS} заявок за каждого приглашённого!"
    )

async def on_profile_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    txt = await _render_profile_html(q.from_user.id)
    await q.message.edit_text(txt, parse_mode="HTML", reply_markup=main_keyboard())

# ---------- выбор модели ----------
async def on_models_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    text = _models_text_short()
    await q.message.edit_text(text, parse_mode="HTML", reply_markup=models_keyboard())

async def on_models_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if q.data == "models_full":
        text = _models_text_full()
    else:
        text = _models_text_short()
    await q.message.edit_text(text, parse_mode="HTML", reply_markup=models_keyboard())

async def on_model_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    label = q.data.split("sel:")[1]
    _user_model_visual[q.from_user.id] = label
    _user_model[q.from_user.id] = MODEL_DEEPSEEK if "DeepSeek" in label else MODEL_OPENAI
    await q.message.edit_text(f"✅ Модель «{label}» установлена.\nМожно писать сообщение!", reply_markup=main_keyboard())

# ---------- рефералка ----------
async def _render_referral_html(user_id: int) -> str:
    me = await application.bot.get_me()
    link = f"https://t.me/{me.username}?start=ref_{user_id}"
    return (
        "🎁 <b>Реферальная программа</b>\n\n"
        f"Приглашайте друзей и получайте +{REF_BONUS} бесплатных сообщений!\n\n"
        f"🔗 Ваша ссылка:\n{link}"
    )

async def on_ref_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    txt = await _render_referral_html(q.from_user.id)
    await q.message.edit_text(txt, parse_mode="HTML", reply_markup=main_keyboard())

# ---------- оплата ----------
async def on_buy_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not CRYPTOPAY_KEY:
        await q.message.reply_text("💳 Оплата не подключена.")
        return

    payload = str(q.from_user.id)
    headers = {"Crypto-Pay-API-Token": CRYPTOPAY_KEY}
    data = {"asset": "USDT", "amount": "3", "description": "Подписка на 30 дней", "payload": payload}
    r = requests.post("https://pay.crypt.bot/api/createInvoice", json=data, headers=headers, timeout=15)
    url = r.json()["result"]["pay_url"]

    msg = (
        "💳 <b>Премиум подписка</b>\n\n"
        "✅ Безлимитный доступ ко всем моделям\n"
        "⚡ Приоритетная скорость ответов\n"
        "📅 Срок — 30 дней\n"
        "💰 Стоимость — 3 USDT\n\n"
        f"Оплатите по ссылке ниже 👇\n{url}"
    )
    await q.message.reply_text(msg, parse_mode="HTML")

# ---------- сообщения ----------
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""

    if await is_premium(user_id):
        reply = ask_llm(user_id, text)
        await update.message.reply_text(reply)
        return

    if await can_send_message(user_id, DAILY_LIMIT):
        reply = ask_llm(user_id, text)
        await update.message.reply_text(reply)
        return

    if await consume_free_credit(user_id):
        reply = ask_llm(user_id, text)
        await update.message.reply_text(reply)
        return

    await update.message.reply_text(
        f"🚫 Лимит исчерпан.\nВы можете пригласить друзей и получить +{REF_BONUS} сообщений "
        "или купить премиум для безлимита."
    )

# ---------- webhook CryptoPay ----------
@app.post("/cryptopay-webhook")
async def cryptopay_webhook(request: Request):
    global application
    data = await request.json()
    user_id = None

    if data.get("update_type") == "invoice_paid":
        try:
            user_id = int(data["payload"]["payload"])
        except Exception:
            pass

    if user_id:
        expires = (datetime.now() + timedelta(days=30)).isoformat()
        await set_premium(user_id, expires)
        await application.bot.send_message(
            chat_id=user_id,
            text=f"✅ Оплата получена! Премиум активен до {datetime.now().strftime('%d.%m.%Y')}.",
        )

    return {"ok": True}

# ---------- системное ----------
@app.on_event("startup")
async def on_startup():
    global application, _public_url
    await init_db()
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CallbackQueryHandler(on_profile_btn, pattern="^profile$"))
    application.add_handler(CallbackQueryHandler(on_ref_btn, pattern="^ref$"))
    application.add_handler(CallbackQueryHandler(on_models_btn, pattern="^models$"))
    application.add_handler(CallbackQueryHandler(on_models_toggle, pattern="^models_(short|full)$"))
    application.add_handler(CallbackQueryHandler(on_model_select, pattern="^sel:"))
    application.add_handler(CallbackQueryHandler(on_buy_btn, pattern="^buy$"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    await application.initialize()
    await application.start()

    _public_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    webhook_url = f"{_public_url.rstrip('/')}/tg"
    await application.bot.set_webhook(webhook_url)
    logger.info("✅ Webhook установлен: %s", webhook_url)

@app.on_event("shutdown")
async def on_shutdown():
    _keepalive_stop.set()
    if application:
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
