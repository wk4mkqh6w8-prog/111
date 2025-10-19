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

BOT_TOKEN     = os.getenv("BOT_TOKEN", "")
OPENAI_KEY    = os.getenv("OPENAI_KEY", "")
CRYPTOPAY_KEY = os.getenv("CRYPTOPAY_KEY")          # опционально (для оплаты)
DEEPSEEK_KEY  = os.getenv("DEEPSEEK_KEY", "")       # опционально (для DeepSeek)
PORT          = int(os.getenv("PORT", "10000"))
ADMIN_ID      = int(os.getenv("ADMIN_ID", "0") or 0)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст. Добавь его в переменные окружения.")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст. Добавь его в переменные окружения.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("neurobot")

client = OpenAI(api_key=OPENAI_KEY)

# БД-утилиты
from db import (
    init_db, add_user, is_premium, can_send_message, set_premium,
    record_payment, get_stats_today, get_totals, remove_premium,
    set_model, get_model
)  # noqa: E402

# ============================================================================
# FastAPI + Telegram Application (webhook-only)
# ============================================================================
app = FastAPI(title="NeuroBot API")
application: Application | None = None
_public_url: str | None = None
_keepalive_stop = threading.Event()
_webhook_guard_task: asyncio.Task | None = None

# ---------- доступные модели ----------
MODELS = {
    "openai:gpt-4o-mini": {
        "title": "OpenAI · GPT-4o mini",
        "provider": "openai",
        "model": "gpt-4o-mini",
    },
    "deepseek:chat": {
        "title": "DeepSeek · Chat",
        "provider": "deepseek",
        "model": "deepseek-chat",
    },
}

def list_model_buttons() -> InlineKeyboardMarkup:
    rows = []
    for code, meta in MODELS.items():
        rows.append([InlineKeyboardButton(meta["title"], callback_data=f"model|{code}")])
    return InlineKeyboardMarkup(rows)

# ---------- вызовы провайдеров ----------
def ask_openai(model_name: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

def ask_deepseek(model_name: str, prompt: str) -> str:
    if not DEEPSEEK_KEY:
        return "DeepSeek недоступен: не задан DEEPSEEK_KEY."

    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,  # "deepseek-chat"
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
            timeout=30,
        )

        # HTTP-ошибка (401, 403, 429, 5xx и т.д.)
        if r.status_code != 200:
            try:
                err_json = r.json()
                msg = err_json.get("error", {}).get("message") or err_json.get("message") or str(err_json)
            except Exception:
                msg = r.text[:400]
            return f"DeepSeek API error {r.status_code}: {msg}"

        j = r.json()

        # нормальный ответ: choices[0].message.content (или .text как запасной вариант)
        if isinstance(j, dict) and "choices" in j and j["choices"]:
            choice = j["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message") or {}
                content = msg.get("content")
                if content:
                    return content
                text = choice.get("text")
                if text:
                    return text

        # ошибка внутри JSON
        if isinstance(j, dict) and "error" in j:
            e = j["error"]
            return f"DeepSeek error: {e.get('message', str(e))}"

        # непредвиденный формат
        return f"DeepSeek: unexpected response: {str(j)[:400]}"

    except Exception as e:
        return f"Ошибка DeepSeek: {e}"

# ---------- хелперы ----------
def is_admin(update: Update) -> bool:
    return ADMIN_ID and update.effective_user and int(update.effective_user.id) == int(ADMIN_ID)

# ---------- хэндлеры бота ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await add_user(user.id)

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Выбрать модель", callback_data="model|open")],
        [InlineKeyboardButton("💳 Купить подписку", callback_data="buy")],
    ])
    await update.message.reply_text(
        "Привет! Я нейробот 🤖\n\n"
        "Бесплатно: 5 сообщений в день.\n"
        "Премиум — без ограничений.\n\n"
        "Выбирай действие ниже:",
        reply_markup=kb,
    )

async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(f"Ваш Telegram ID: {uid}\nADMIN_ID в боте: {ADMIN_ID}")

async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа. Установите ADMIN_ID или используйте свой админ-аккаунт.")
        return
    lines = ["🛡 Админ-панель"]
    try:
        totals = await get_totals()
        today = await get_stats_today()
        lines += [
            f"📈 Всего: пользователи={totals.get('users',0)}, премиум={totals.get('premium',0)}, оплат={totals.get('payments',0)}",
            f"📊 Сегодня: оплаты={today.get('payments',0)}, новые пользователи={today.get('new_users',0)}, доход={today.get('revenue_usdt',0)} USDT",
        ]
    except Exception as e:
        lines.append(f"⚠️ Не удалось получить статистику: {e}")

    lines.append("\nКоманды:")
    lines.append("• /model — выбрать модель")
    lines.append("• /stats — показать статистику")
    lines.append("• /grant_premium <tg_id> [days=30]")
    lines.append("• /revoke_premium <tg_id>")
    await update.message.reply_text("\n".join(lines))

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    totals = await get_totals()
    today = await get_stats_today()
    text = (
        "📊 Статистика\n"
        f"Сегодня: оплаты={today.get('payments',0)}, новые пользователи={today.get('new_users',0)}, доход={today.get('revenue_usdt',0)} USDT\n"
        f"Всего: пользователи={totals.get('users',0)}, премиум={totals.get('premium',0)}, оплат={totals.get('payments',0)}"
    )
    await update.message.reply_text(text)

async def cmd_grant_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    args = context.args
    if not args:
        await update.message.reply_text("Использование: /grant_premium <tg_id> [days=30]")
        return
    try:
        tg_id = int(args[0])
        days = int(args[1]) if len(args) > 1 else 30
    except Exception:
        await update.message.reply_text("Неверные аргументы. Пример: /grant_premium 123456789 30")
        return
    expires_at = (datetime.now() + timedelta(days=days)).isoformat()
    await set_premium(tg_id, expires_at)
    await update.message.reply_text(f"✅ Выдал премиум {tg_id} до {expires_at}")

async def cmd_revoke_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    args = context.args
    if not args:
        await update.message.reply_text("Использование: /revoke_premium <tg_id>")
        return
    try:
        tg_id = int(args[0])
    except Exception:
        await update.message.reply_text("Неверный tg_id.")
        return
    await remove_premium(tg_id)
    await update.message.reply_text(f"✅ Снял премиум у {tg_id}")

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выберите модель:", reply_markup=list_model_buttons())

async def on_model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Выберите модель:", reply_markup=list_model_buttons())

async def on_model_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        _, code = (query.data or "").split("|", 1)
    except ValueError:
        await query.message.reply_text("Некорректный выбор модели.")
        return

    meta = MODELS.get(code)
    if not meta:
        await query.message.reply_text("Неизвестная модель.")
        return

    await set_model(query.from_user.id, code)
    await query.message.reply_text(f"✅ Модель установлена: {meta['title']}")

async def on_buy_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not CRYPTOPAY_KEY:
        await query.message.reply_text("💳 Оплата ещё не подключена. Добавь CRYPTOPAY_KEY в переменные окружения.")
        return
    payload = str(query.from_user.id)
    headers = {"Crypto-Pay-API-Token": CRYPTOPAY_KEY}
    data = {"asset": "USDT", "amount": "3", "description": "Подписка на 30 дней", "payload": payload}
    try:
        r = requests.post("https://pay.crypt.bot/api/createInvoice", json=data, headers=headers, timeout=15)
        j = r.json()
        url = j["result"]["pay_url"]
        await query.message.reply_text(f"💳 Оплати подписку по ссылке:\n{url}")
    except Exception as e:
        await query.message.reply_text(f"❌ Не удалось создать счёт: {e}")

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""

    # модель пользователя
    code = await get_model(user_id)
    meta = MODELS.get(code, MODELS["openai:gpt-4o-mini"])
    provider, model_name = meta["provider"], meta["model"]

    # премиум — без ограничений
    if await is_premium(user_id):
        try:
            if provider == "openai":
                reply = ask_openai(model_name, text)
            elif provider == "deepseek":
                reply = ask_deepseek(model_name, text)
            else:
                reply = "Эта модель недоступна. Выберите другую через /model."
            await update.message.reply_text(reply)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка ИИ: {e}")
        return

    # бесплатный лимит (5/день)
    if await can_send_message(user_id, limit=5):
        try:
            if provider == "openai":
                reply = ask_openai(model_name, text)
            elif provider == "deepseek":
                reply = ask_deepseek(model_name, text)
            else:
                reply = "Эта модель недоступна. Выберите другую через /model."
            await update.message.reply_text(reply)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка ИИ: {e}")
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
    status  = invoice.get("status")
    payload = invoice.get("payload")
    amount  = invoice.get("amount") or invoice.get("paid_amount")
    asset   = invoice.get("asset")  or invoice.get("paid_asset") or "USDT"

    if status == "paid" and payload:
        try:
            user_id = int(payload)
        except Exception:
            user_id = None

        if user_id:
            expires_at = (datetime.now() + timedelta(days=30)).isoformat()
            await set_premium(user_id, expires_at)

            # запись платежа
            try:
                val = None
                if amount is not None:
                    try:
                        val = float(str(amount))
                    except Exception:
                        val = None
                await record_payment(user_id, val, asset, datetime.utcnow().isoformat())
            except Exception:
                pass

            # уведомления
            try:
                await application.bot.send_message(chat_id=user_id, text="✅ Оплата получена! Подписка активирована на 30 дней.")
            except Exception:
                pass
            if ADMIN_ID:
                try:
                    pretty_amt = f"{(val if val is not None else 0):g} {asset}" if val is not None else asset
                    await application.bot.send_message(chat_id=ADMIN_ID, text=f"💰 Оплата: user={user_id}, amount={pretty_amt}")
                except Exception:
                    pass

    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# ---------- Keep-alive пинг (каждые 40 секунд) ----------
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
    while not _keepalive_stop.wait(40):
        try:
            session.get(url, timeout=8)
            logger.info("keepalive: ping %s", url)
        except Exception:
            pass

# ---------- Конструктор PTB-приложения ----------
def build_application() -> Application:
    app_ = ApplicationBuilder().token(BOT_TOKEN).build()
    app_.add_handler(CommandHandler("start",  cmd_start))
    app_.add_handler(CommandHandler("whoami", cmd_whoami))
    app_.add_handler(CommandHandler("admin",  cmd_admin))
    app_.add_handler(CommandHandler("stats",  cmd_stats))
    app_.add_handler(CommandHandler("grant_premium",  cmd_grant_premium))
    app_.add_handler(CommandHandler("revoke_premium", cmd_revoke_premium))
    app_.add_handler(CommandHandler("model",  cmd_model))
    app_.add_handler(CallbackQueryHandler(on_model_menu, pattern=r"^model\|open$"))
    app_.add_handler(CallbackQueryHandler(on_model_pick, pattern=r"^model\|"))
    app_.add_handler(CallbackQueryHandler(on_buy_btn,   pattern=r"^buy$"))
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
            current = (info.url or "").rstrip("/")
            if current != target.rstrip("/"):
                logger.warning("webhook guard: mismatch (%s) -> fixing to %s", current, target)
                await _set_webhook()
        except Exception as e:
            logger.warning("webhook guard: error: %s (resetting) %s", type(e).__name__, e)
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
