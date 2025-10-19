import os
import logging
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

# ===============================
# Настройки и клиенты
# ===============================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
CRYPTOPAY_KEY = os.getenv("CRYPTOPAY_KEY")  # может быть None
PORT = int(os.getenv("PORT", "10000"))

# Администраторы бота: "123456789,987654321"
ADMIN_IDS_ENV = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(x) for x in ADMIN_IDS_ENV.replace(" ", "").split(",") if x.isdigit()}

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст. Добавь его в переменные окружения Render.")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст. Добавь его в переменные окружения Render.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurobot")

client = OpenAI(api_key=OPENAI_KEY)

# БД-утилиты
from db import (  # noqa: E402
    init_db, add_user, is_premium, can_send_message, set_premium,
    set_premium_days, remove_premium, get_user, stats,
    log_payment, sales_summary, daily_breakdown
)

# ============================================================================
# FastAPI + Telegram Application (webhook-only)
# ============================================================================
app = FastAPI(title="NeuroBot API")
application: Application | None = None
_public_url: str | None = None
_keepalive_stop = threading.Event()


# ===============================
# GPT
# ===============================
def ask_gpt(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ===============================
# Хэндлеры пользователя
# ===============================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await add_user(user.id)

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Выбрать нейросеть (пока GPT-4o-mini)", callback_data="neuro")],
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
            "💳 Оплата ещё не подключена. Добавь CRYPTOPAY_KEY в переменные окружения Render."
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
        await query.message.reply_text(
            "💳 Оплати подписку по ссылке:\n"
            f"{url}\n\n"
            "После оплаты доступ откроется автоматически ✅"
        )
    except Exception as e:
        await query.message.reply_text(f"❌ Не удалось создать счёт: {e}")


async def on_neuro_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Пока доступна только GPT-4o-mini. Другие модели добавим позже 🔧")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""

    # Премиум — без ограничений
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


# ===============================
# Админ-панель
# ===============================
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


ADMIN_HELP = (
    "🔐 Админ-панель:\n"
    "/whoami — показать твой id\n"
    "/stats — пользователи и покупки за сегодня\n"
    "/sales — продажи: сегодня / 7 дней / всего + разбивка по дням\n"
    "/status <tg_id> — статус пользователя\n"
    "/grant <tg_id> [дней] — выдать премиум (по умолчанию 30)\n"
    "/revoke <tg_id> — снять премиум\n"
)


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"🆔 Ваш Telegram ID: {update.effective_user.id}")


async def admin_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("⛔ Нет доступа.")
    await update.message.reply_text(ADMIN_HELP)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("⛔ Нет доступа.")
    s = await stats()
    await update.message.reply_text(
        "📊 Статистика за сегодня:\n"
        f"• Пользователей всего: {s['total']}\n"
        f"• С премиумом: {s['premium']}\n"
        f"• Активных сегодня: {s['active_today']}\n"
        f"• Покупок сегодня: {s['purchases_today']}"
    )


async def cmd_sales(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("⛔ Нет доступа.")
    s = await sales_summary()
    days = await daily_breakdown(days=7)
    hist = "\n".join([f"• {d}: {c}" for d, c in days])
    await update.message.reply_text(
        "💳 Продажи:\n"
        f"• Сегодня: {s['today']}\n"
        f"• За 7 дней: {s['week']}\n"
        f"• Всего: {s['total']}\n\n"
        f"📅 По дням (последние 7):\n{hist}"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("⛔ Нет доступа.")
    if not context.args:
        return await update.message.reply_text("Использование: /status <tg_id>")
    try:
        uid = int(context.args[0])
    except Exception:
        return await update.message.reply_text("tg_id должен быть числом.")
    u = await get_user(uid)
    if not u:
        return await update.message.reply_text("Пользователь не найден.")
    _, premium_until, messages_today, last_date = u
    await update.message.reply_text(
        "👤 Пользователь: {uid}\nПремиум до: {pu}\nСообщений сегодня: {m}\nПоследняя дата: {d}".format(
            uid=uid, pu=premium_until or "—", m=messages_today or 0, d=last_date or "—"
        )
    )


async def cmd_grant(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("⛔ Нет доступа.")
    if not context.args:
        return await update.message.reply_text("Использование: /grant <tg_id> [дней]")
    try:
        uid = int(context.args[0])
        days = int(context.args[1]) if len(context.args) > 1 else 30
    except Exception:
        return await update.message.reply_text("tg_id и дни должны быть числами.")
    await set_premium_days(uid, days)
    await update.message.reply_text(f"✅ Выдан премиум на {days} дн. пользователю {uid}")


async def cmd_revoke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("⛔ Нет доступа.")
    if not context.args:
        return await update.message.reply_text("Использование: /revoke <tg_id>")
    try:
        uid = int(context.args[0])
    except Exception:
        return await update.message.reply_text("tg_id должен быть числом.")
    await remove_premium(uid)
    await update.message.reply_text(f"✅ Премиум снят у {uid}")


# ===============================
# Webhook-эндоинты
# ===============================
@app.post("/tg")
async def telegram_webhook(request: Request):
    """
    Принимаем апдейты от Telegram и прокидываем в PTB.
    """
    global application
    if application is None:
        return {"ok": False, "error": "bot not initialized"}

    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}


@app.post("/cryptopay-webhook")
async def cryptopay_webhook(request: Request):
    """
    Принимаем уведомления от CryptoBot:
    {
      "invoice": {
         "status": "paid",
         "payload": "<telegram_user_id>",
         "amount": "3",
         "asset": "USDT"
      }
    }
    """
    global application
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "bad json"}

    invoice = data.get("invoice") or {}
    status = invoice.get("status")
    payload = invoice.get("payload")

    # пробуем извлечь сумму/валюту (если есть)
    amount = None
    try:
        if "amount" in invoice and invoice["amount"] is not None:
            amount = float(invoice["amount"])
    except Exception:
        amount = None
    asset = invoice.get("asset")

    if status == "paid" and payload:
        try:
            user_id = int(payload)
        except Exception:
            user_id = None

        if user_id:
            # логируем оплату
            try:
                await log_payment(user_id, amount, asset)
            except Exception:
                pass

            # выдаём премиум на 30 дней
            expires_at = (datetime.now() + timedelta(days=30)).isoformat()
            await set_premium(user_id, expires_at)

            # уведомляем пользователя
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text="✅ Оплата получена! Подписка активирована на 30 дней.",
                )
            except Exception:
                pass

    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ===============================
# Keep-alive пинг
# ===============================
def _keepalive_loop():
    """
    Фоновый поток: раз в 9 минут дергает /health, чтобы держать соединения теплыми,
    пока инстанс активен (на Free Render всё равно может уснуть при длительном простое).
    """
    if not _public_url:
        return
    url = f"{_public_url.rstrip('/')}/health"
    session = requests.Session()
    while not _keepalive_stop.wait(540):  # 9 минут
        try:
            session.get(url, timeout=8)
            logger.debug("keep-alive ping %s", url)
        except Exception:
            pass


# ===============================
# Жизненный цикл FastAPI
# ===============================
def build_application() -> Application:
    app_ = ApplicationBuilder().token(BOT_TOKEN).build()

    # Пользовательские
    app_.add_handler(CommandHandler("start", cmd_start))
    app_.add_handler(CallbackQueryHandler(on_buy_btn, pattern=r"^buy$"))
    app_.add_handler(CallbackQueryHandler(on_neuro_btn, pattern=r"^neuro$"))
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    # Админские
    app_.add_handler(CommandHandler("whoami", whoami))
    app_.add_handler(CommandHandler("admin", admin_menu))
    app_.add_handler(CommandHandler("stats", cmd_stats))
    app_.add_handler(CommandHandler("sales", cmd_sales))
    app_.add_handler(CommandHandler("status", cmd_status))
    app_.add_handler(CommandHandler("grant", cmd_grant))
    app_.add_handler(CommandHandler("revoke", cmd_revoke))

    return app_


@app.on_event("startup")
async def on_startup():
    global application, _public_url

    # 1) DB
    await init_db()

    # 2) Telegram Application
    application = build_application()
    await application.initialize()
    await application.start()

    # 3) Webhook
    _public_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    if not _public_url:
        # на Render RENDER_EXTERNAL_URL подставляется автоматически
        raise RuntimeError("Не найден PUBLIC_URL/RENDER_EXTERNAL_URL")

    webhook_url = f"{_public_url.rstrip('/')}/tg"
    await application.bot.set_webhook(webhook_url)
    logger.info("✅ Установлен Telegram webhook: %s", webhook_url)

    # 4) Keep-alive
    threading.Thread(target=_keepalive_loop, daemon=True).start()

    logger.info("🚀 Startup complete. Listening on port %s", PORT)


@app.on_event("shutdown")
async def on_shutdown():
    _keepalive_stop.set()
    try:
        if application is not None:
            try:
                await application.bot.delete_webhook(drop_pending_updates=False)
            except Exception:
                pass
            await application.stop()
            await application.shutdown()
    finally:
        logger.info("🛑 Shutdown complete")


# ============================================================================
# Запуск локально / на Render
# ============================================================================
if __name__ == "__main__":
    # На Render просто используем Start Command:  python bot.py
    uvicorn.run(app, host="0.0.0.0", port=PORT)
