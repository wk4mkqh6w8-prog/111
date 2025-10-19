import os
import logging
from datetime import datetime, timedelta
from threading import Thread

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import uvicorn

from openai import OpenAI

from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --------------------
# Настройки и клиенты
# --------------------
load_dotenv()

BOT_TOKEN      = os.getenv("BOT_TOKEN", "")
OPENAI_KEY     = os.getenv("OPENAI_KEY", "")
CRYPTOPAY_KEY  = os.getenv("CRYPTOPAY_KEY")  # может быть None (пока не добавили)
PORT           = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст. Добавь его в .env")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст. Добавь его в .env")

client = OpenAI(api_key=OPENAI_KEY)

# БД-утилиты
from db import init_db, add_user, is_premium, can_send_message, set_premium  # noqa: E402

# --------------------
# FastAPI (webhook для CryptoPay)
# --------------------
app = FastAPI(title="NeuroBot API")
# === Webhook для Telegram ===
@app.post("/tg")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}

@app.post("/cryptopay-webhook")
async def cryptopay_webhook(request: Request):
    """
    Ожидает JSON от CryptoPay вида:
    {
      "update_id": ...,
      "invoice": {
        "status": "paid",
        "payload": "<telegram_user_id>"
      }
    }
    """
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "bad json"}

    invoice = data.get("invoice") or {}
    status  = invoice.get("status")
    payload = invoice.get("payload")

    if status == "paid" and payload:
        try:
            user_id = int(payload)
        except Exception:
            user_id = None

        if user_id:
            expires_at = (datetime.now() + timedelta(days=30)).isoformat()
            await set_premium(user_id, expires_at)
            # отправим пользователю уведомление о премиуме
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text="✅ Оплата получена! Подписка активирована на 30 дней."
                )
            except Exception:
                pass

    return {"ok": True}

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=PORT)

# --------------------
# Вспомогательные функции
# --------------------
def ask_gpt(prompt: str) -> str:
    """
    Вызов OpenAI (модель можно поменять при желании).
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

# --------------------
# Хэндлеры Telegram
# --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await add_user(user.id)

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Выбрать нейросеть (пока GPT-4o-mini)", callback_data="neuro")],
        [InlineKeyboardButton("💳 Купить подписку", callback_data="buy")]
    ])
    await update.message.reply_text(
        "Привет! Я нейробот 🤖\n\n"
        "Бесплатно: 5 сообщений в день.\n"
        "Премиум — без ограничений и очередей.\n\n"
        "Выбирай действие ниже:",
        reply_markup=kb
    )

async def on_buy_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not CRYPTOPAY_KEY:
        await query.message.reply_text(
            "💳 Оплата ещё не подключена. Добавь CRYPTOPAY_KEY в .env — и я выдам ссылку на оплату."
        )
        return

    payload = str(query.from_user.id)
    headers = {"Crypto-Pay-API-Token": CRYPTOPAY_KEY}
    data = {
        "asset": "USDT",
        "amount": "3",
        "description": "Подписка на 30 дней",
        "payload": payload
    }
    try:
        r = requests.post("https://pay.crypt.bot/api/createInvoice", json=data, headers=headers, timeout=15)
        j = r.json()
        url = j["result"]["pay_url"]
        await query.message.reply_text(f"💳 Оплати подписку по ссылке:\n{url}")
    except Exception as e:
        await query.message.reply_text(f"❌ Не удалось создать счёт: {e}")

async def on_neuro_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Пока доступна только GPT-4o-mini. Выбор других моделей добавим позже 🔧")

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""

    # Премиум-пользователь — без ограничений
    if await is_premium(user_id):
        try:
            reply = ask_gpt(text)
            await update.message.reply_text(reply)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка OpenAI: {e}")
        return

    # Бесплатный лимит (5/день)
    if await can_send_message(user_id, limit=5):
        try:
            reply = ask_gpt(text)
            await update.message.reply_text(reply)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка OpenAI: {e}")
    else:
        await update.message.reply_text(
            "🚫 Сегодня исчерпан лимит из 5 сообщений.\n"
            "Купи подписку через кнопку «💳 Купить подписку», чтобы снять ограничения."
        )

# --------------------
# Инициализация PTB
# --------------------
async def _startup(app):
    # Инициализируем БД и поднимем FastAPI в фоне
    await init_db()
    Thread(target=run_fastapi, daemon=True).start()
    logger.info("✅ База готова и FastAPI запущен на порту %s", PORT)

import asyncio  # добавь вверху, если его ещё нет

def main():
    global application

    # 🩵 Фикс для Python 3.14: создаём event loop вручную
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    application = ApplicationBuilder() \
        .token(BOT_TOKEN) \
        .post_init(_startup) \
        .build()

    # Команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(on_buy_btn,  pattern="^buy$"))
    application.add_handler(CallbackQueryHandler(on_neuro_btn, pattern="^neuro$"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    # === Запуск с webhook (Render) ===
import asyncio
import uvicorn

async def init_bot_and_webhook():
    global application
    await application.initialize()
    await application.start()

    public_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    if not public_url:
        raise RuntimeError(
            "❌ Не найден PUBLIC_URL. На Render он задаётся автоматически. "
            "Локально можно задать вручную."
        )

    webhook_url = f"{public_url}/tg"
    await application.bot.set_webhook(webhook_url)
    logger.info(f"✅ Установлен Telegram webhook: {webhook_url}")

if __name__ == "__main__":
    asyncio.run(init_bot_and_webhook())
    uvicorn.run(app, host="0.0.0.0", port=PORT)
