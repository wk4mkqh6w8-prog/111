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

# --------------------
# Настройки и клиенты
# --------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
CRYPTOPAY_KEY = os.getenv("CRYPTOPAY_KEY")
PORT = int(os.getenv("PORT", "10000"))
ADMIN_ID = int(os.getenv("ADMIN_ID", "0") or 0)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст. Добавь его в переменные окружения.")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст. Добавь его в переменные окружения.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("neurobot")

client = OpenAI(api_key=OPENAI_KEY)

# БД-утилиты
from db import init_db, add_user, is_premium, can_send_message, set_premium  # noqa: E402

# ============================================================================
# FastAPI + Telegram Application (webhook-only)
# ============================================================================
app = FastAPI(title="NeuroBot API")
application: Application | None = None
_public_url: str | None = None
_keepalive_stop = threading.Event()
_webhook_guard_task: asyncio.Task | None = None


# ---------- GPT ----------
def ask_gpt(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ---------- Хэндлеры бота ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await add_user(user.id)

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Выбрать нейросеть (GPT-4o-mini)", callback_data="neuro")],
        [InlineKeyboardButton("💳 Купить подписку", callback_data="buy")],
    ])
    await update.message.reply_text(
        "Привет! Я нейробот 🤖\n\n"
        "Бесплатно: 5 сообщений в день.\n"
        "Премиум — без ограничений и очередей.\n\n"
        "Выбирай действие ниже:",
        reply_markup=kb,
    )


async def on_buy_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not CRYPTOPAY_KEY:
        await query.message.reply_text(
            "💳 Оплата ещё не подключена. Добавь CRYPTOPAY_KEY в переменные окружения."
        )
        return

    payload = str(query.from_user.id)
    headers = {"Crypto-Pay-API-Token": CRYPTOPAY_KEY}
    data = {
        "asset": "USDT",
        "amount": "3",
        "description": "Подписка на 30 дней",
        "payload": payload,
    }
    try:
        r = requests.post(
            "https://pay.crypt.bot/api/createInvoice",
            json=data, headers=headers, timeout=15,
        )
        j = r.json()
        url = j["result"]["pay_url"]
        await query.message.reply_text(f"💳 Оплати подписку по ссылке:\n{url}")
    except Exception as e:
        await query.message.reply_text(f"❌ Не удалось создать счёт: {e}")


async def on_neuro_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Пока доступна только GPT-4o-mini. Другие модели добавим позже 🔧")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""

    if await is_premium(user_id):
        try:
            reply = ask_gpt(text)
            await update.message.reply_text(reply)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка OpenAI: {e}")
        return

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


# ---------- Webhook-эндоинты ----------
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
    global application
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "bad json"}

    invoice = data.get("invoice") or {}
    status = invoice.get("status")
    payload = invoice.get("payload")

    if status == "paid" and payload:
        try:
            user_id = int(payload)
        except Exception:
            user_id = None

        if user_id:
            expires_at = (datetime.now() + timedelta(days=30)).isoformat()
            await set_premium(user_id, expires_at)
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text="✅ Оплата получена! Подписка активирована на 30 дней.",
                )
            except Exception:
                pass

            if ADMIN_ID:
                try:
                    await application.bot.send_message(
                        chat_id=ADMIN_ID,
                        text=f"💰 Оплата получена от {user_id}. Премиум до {expires_at}"
                    )
                except Exception:
                    pass

    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ---------- Keep-alive пинг (чтобы Render не усыплял) ----------
def _keepalive_loop():
    """
    Раз в 40 секунд дёргаем /health, чтобы Render не усыплял инстанс.
    Первый пинг — сразу.
    """
    if not _public_url:
        return
    url = f"{_public_url.rstrip('/')}/health"
    session = requests.Session()
    try:
        session.get(url, timeout=8)
        logger.info("keepalive: first ping %s", url)
    except Exception:
        pass

    while not _keepalive_stop.wait(40):  # 40 секунд
        try:
            session.get(url, timeout=8)
            logger.info("keepalive: ping %s", url)
        except Exception:
            pass


# ---------- Конструктор PTB-приложения ----------
def build_application() -> Application:
    app_ = ApplicationBuilder().token(BOT_TOKEN).build()
    app_.add_handler(CommandHandler("start", cmd_start))
    app_.add_handler(CallbackQueryHandler(on_buy_btn, pattern=r"^buy$"))
    app_.add_handler(CallbackQueryHandler(on_neuro_btn, pattern=r"^neuro$"))
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    return app_


# ---------- Утилиты для вебхука ----------
async def _set_webhook():
    assert application is not None
    assert _public_url
    try:
        await application.bot.delete_webhook(drop_pending_updates=False)
    except Exception:
        pass

    webhook_url = f"{_public_url.rstrip('/')}/tg"
    await application.bot.set_webhook(
        url=webhook_url,
        max_connections=40,
        drop_pending_updates=False,
    )
    logger.info("bot:✅ Установлен Telegram webhook: %s", webhook_url)


async def _ensure_webhook_forever():
    assert application is not None
    assert _public_url
    target = f"{_public_url.rstrip('/')}/tg"
    while True:
        try:
            info = await application.bot.get_webhook_info()
            current = info.url or ""
            if current.rstrip("/") != target.rstrip("/"):
                logger.warning("webhook guard: mismatch (%s) -> fixing to %s", current, target)
                await _set_webhook()
        except Exception as e:
            logger.warning("webhook guard: error: %s (resetting)", e)
            try:
                await _set_webhook()
            except Exception:
                pass
        await asyncio.sleep(120)


# ---------- Жизненный цикл FastAPI ----------
@app.on_event("startup")
async def on_startup():
    global application, _public_url, _webhook_guard_task

    await init_db()
    application = build_application()
    await application.initialize()
    await application.start()

    _public_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    if not _public_url:
        raise RuntimeError("Не найден PUBLIC_URL/RENDER_EXTERNAL_URL")

    await _set_webhook()

    threading.Thread(target=_keepalive_loop, daemon=True).start()
    _webhook_guard_task = asyncio.create_task(_ensure_webhook_forever())

    logger.info("bot:🚀 Startup complete. Listening on port %s", PORT)


@app.on_event("shutdown")
async def on_shutdown():
    _keepalive_stop.set()
    try:
        if _webhook_guard_task:
            _webhook_guard_task.cancel()
            try:
                await _webhook_guard_task
            except Exception:
                pass
        if application is not None:
            try:
                await application.bot.delete_webhook(drop_pending_updates=False)
            except Exception:
                pass
            await application.stop()
            await application.shutdown()
    finally:
        logger.info("bot:🛑 Shutdown complete")


# ============================================================================
# Запуск локально / на Render
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
