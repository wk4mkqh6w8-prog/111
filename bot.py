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

# ========================
# Настройки и инициализация
# ========================
load_dotenv()

BOT_TOKEN     = os.getenv("BOT_TOKEN", "")
OPENAI_KEY    = os.getenv("OPENAI_KEY", "")
CRYPTOPAY_KEY = os.getenv("CRYPTOPAY_KEY")
ADMIN_ID      = os.getenv("ADMIN_ID")  # строкой, можно int(ADMIN_ID) где надо
PORT          = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст.")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurobot")

client = OpenAI(api_key=OPENAI_KEY)

# БД-утилиты (часть функций может отсутствовать — обрабатываем мягко)
from db import init_db, add_user, is_premium, can_send_message, set_premium  # noqa: E402
try:
    from db import record_payment  # async def record_payment(user_id, amount, asset, at: iso str)
except Exception:
    record_payment = None
try:
    from db import get_stats_today, get_totals  # обе async
except Exception:
    get_stats_today = None
    get_totals = None

# ========================
# Приложения
# ========================
app = FastAPI(title="NeuroBot API")
application: Application | None = None
_public_url: str | None = None
_keepalive_stop = threading.Event()


# ========================
# GPT
# ========================
def ask_gpt(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ========================
# Хэндлеры бота
# ========================
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


async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # доступ только админу
    if not ADMIN_ID or str(update.effective_user.id) != str(ADMIN_ID):
        return

    lines = ["🛡 Админ-панель"]
    # сегодняшняя/общая статистика — если функции есть в db.py
    try:
        if get_stats_today:
            s = await get_stats_today()
            lines.append(
                f"📊 Сегодня: оплаты={s.get('payments', 0)}, новые пользователи={s.get('new_users', 0)}"
            )
        if get_totals:
            t = await get_totals()
            lines.append(
                f"📈 Всего: пользователи={t.get('users', 0)}, премиум={t.get('premium', 0)}, оплаты={t.get('payments', 0)}"
            )
    except Exception as e:
        lines.append(f"⚠️ Ошибка чтения статистики: {e}")

    # текущие переменные окружения (укороченно)
    lines.append(f"WEBHOOK: {(_public_url or '')[:80]}")
    await update.message.reply_text("\n".join(lines))


async def on_buy_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not CRYPTOPAY_KEY:
        await query.message.reply_text(
            "💳 Оплата ещё не подключена. Добавь CRYPTOPAY_KEY в переменные окружения."
        )
        return

    payload = str(query.from_user.id)  # важно: это мы затем получим в вебхуке
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


# ========================
# HTTP-эндпоинты (webhooks)
# ========================
@app.post("/tg")
async def telegram_webhook(request: Request):
    """
    Эндпоинт для Telegram webhook.
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
    Эндпоинт для уведомлений от CryptoBot.
    Ловим разные варианты структуры, подробно логируем, выдаём премиум и пишем статистику.
    """
    global application

    raw = await request.body()
    try:
        raw_text = raw.decode("utf-8", "ignore")
    except Exception:
        raw_text = str(raw)
    logger.info("CRYPTO WEBHOOK RAW: %s", raw_text)

    # Разбор JSON
    try:
        data = await request.json()
    except Exception:
        logger.exception("CRYPTO WEBHOOK: bad json")
        return {"ok": False, "reason": "bad json"}

    # Извлекаем invoice/status/payload из разных возможных обёрток
    invoice = (
        data.get("invoice")
        or (data.get("payload") or {}).get("invoice")
        or (data.get("update") or {}).get("invoice")
        or {}
    )
    status = (invoice.get("status") or data.get("status") or "").lower().strip()
    payload = (
        invoice.get("payload")
        or (invoice.get("custom_data") or {}).get("payload")
        or (data.get("payload") if isinstance(data.get("payload"), str) else None)
    )
    amount = invoice.get("amount") or invoice.get("paid_amount")
    asset  = invoice.get("asset") or invoice.get("paid_asset") or "USDT"

    if status == "paid" and payload:
        try:
            user_id = int(str(payload).strip())
        except Exception:
            user_id = None

        if user_id:
            try:
                # 1) Премиум на 30 дней
                expires_at = (datetime.now() + timedelta(days=30)).isoformat()
                await set_premium(user_id, expires_at)

                # 2) Запись платежа в статистику (если функция есть)
                try:
                    if record_payment:
                        # нормализуем amount к строке/float
                        val = None
                        try:
                            val = float(str(amount)) if amount is not None else None
                        except Exception:
                            pass
                        await record_payment(user_id, val, asset, datetime.utcnow().isoformat())
                except Exception:
                    logger.exception("CRYPTO WEBHOOK: record_payment failed")

                # 3) Уведомления
                try:
                    await application.bot.send_message(
                        chat_id=user_id,
                        text="✅ Оплата получена! Подписка активирована на 30 дней.",
                    )
                except Exception:
                    logger.exception("CRYPTO WEBHOOK: can't notify user %s", user_id)

                if ADMIN_ID:
                    try:
                        await application.bot.send_message(
                            chat_id=int(ADMIN_ID),
                            text=f"💰 Оплата: user={user_id}, amount={amount} {asset}",
                        )
                    except Exception:
                        pass

                logger.info("CRYPTO WEBHOOK: premium set for user %s until %s", user_id, expires_at)
                return {"ok": True}
            except Exception:
                logger.exception("CRYPTO WEBHOOK: set_premium failed")
                return {"ok": False, "reason": "set_premium failed"}

    # если не смогли обработать — предупредим админа
    try:
        if ADMIN_ID:
            await application.bot.send_message(
                chat_id=int(ADMIN_ID),
                text=f"⚠️ Crypto webhook не обработан.\nstatus={status}\npayload={payload}\nraw={raw_text[:1000]}",
            )
    except Exception:
        pass

    logger.warning("CRYPTO WEBHOOK: unhandled payload. status=%s payload=%s data=%s", status, payload, data)
    return {"ok": True, "handled": False}


@app.get("/health")
async def health():
    # Используем UTC, чтобы не ловить DeprecationWarning от utcnow()
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ========================
# Keep-alive
# ========================
def _keepalive_loop():
    if not _public_url:
        return
    url = f"{_public_url.rstrip('/')}/health"
    session = requests.Session()
    while not _keepalive_stop.wait(540):  # ~9 минут
        try:
            session.get(url, timeout=8)
            logger.debug("keep-alive ping %s", url)
        except Exception:
            pass


# ========================
# Жизненный цикл FastAPI
# ========================
def build_application() -> Application:
    app_ = ApplicationBuilder().token(BOT_TOKEN).build()
    app_.add_handler(CommandHandler("start", cmd_start))
    app_.add_handler(CommandHandler("admin", cmd_admin))
    app_.add_handler(CallbackQueryHandler(on_buy_btn, pattern=r"^buy$"))
    app_.add_handler(CallbackQueryHandler(on_neuro_btn, pattern=r"^neuro$"))
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
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
        raise RuntimeError("Не найден PUBLIC_URL/RENDER_EXTERNAL_URL")

    webhook_url = f"{_public_url.rstrip('/')}/tg"
    await application.bot.set_webhook(webhook_url, drop_pending_updates=True)
    logger.info("✅ Установлен Telegram webhook: %s", webhook_url)

    # 4) Keep-alive
    threading.Thread(target=_keepalive_loop, daemon=True).start()

    logger.info("🚀 Startup complete. Listening on port %s", PORT)


@app.on_event("shutdown")
async def on_shutdown():
    _keepalive_stop.set()
    try:
        if application is not None:
            await application.stop()
            await application.shutdown()
    finally:
        logger.info("🛑 Shutdown complete")


# ========================
# Запуск локально/Render
# ========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
