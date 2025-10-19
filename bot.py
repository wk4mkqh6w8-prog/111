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
# Конфиг и клиенты
# =========================
load_dotenv()

BOT_TOKEN      = os.getenv("BOT_TOKEN", "")
OPENAI_KEY     = os.getenv("OPENAI_KEY", "")
DEEPSEEK_KEY   = os.getenv("DEEPSEEK_KEY", "")
CRYPTOPAY_KEY  = os.getenv("CRYPTOPAY_KEY", "")
ADMIN_ID       = int(os.getenv("ADMIN_ID", "0"))
PORT           = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN пуст")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY пуст")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("neurobot")

# Модели
MODEL_OPENAI   = "OpenAI · GPT-4o-mini"
MODEL_DEEPSEEK = "DeepSeek · Chat"
DEFAULT_MODEL  = MODEL_OPENAI
_user_model: dict[int, str] = {}

# OpenAI клиент
oai = OpenAI(api_key=OPENAI_KEY)

# =========================
# DB helpers
# =========================
from db import (  # noqa
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

REF_BONUS   = 25
DAILY_LIMIT = 5

# ---------- LLM ----------
def _ask_openai(prompt: str) -> str:
    r = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return r.choices[0].message.content

def _ask_deepseek(prompt: str) -> str:
    if not DEEPSEEK_KEY:
        return "DeepSeek недоступен: не задан DEEPSEEK_KEY."
    try:
        import httpx
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
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
    model = _user_model.get(user_id, DEFAULT_MODEL)
    if model == MODEL_DEEPSEEK:
        return _ask_deepseek(prompt)
    return _ask_openai(prompt)

# ---------- UI ----------
def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Выбрать модель", callback_data="models")],
        [InlineKeyboardButton("👤 Профиль", callback_data="profile")],
        [InlineKeyboardButton("🎁 Реферальная программа", callback_data="ref")],
        [InlineKeyboardButton("💳 Купить подписку", callback_data="buy")],
    ])

def models_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("OpenAI · GPT-4o-mini", callback_data="m:oai")],
        [InlineKeyboardButton("DeepSeek · Chat",     callback_data="m:ds")],
        [InlineKeyboardButton("⬅️ Назад",            callback_data="home")],
    ])

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

    text = (
        "Привет! Я нейробот 🤖\n\n"
        f"Бесплатно: {DAILY_LIMIT} сообщений в день (+ реф. бонусы).\n"
        "Премиум — без ограничений и очередей.\n\n"
        "Выбирай действие ниже:"
    )
    if update.message:
        await update.message.reply_text(text, reply_markup=main_keyboard())
    else:
        # на всякий случай, если старт по кнопке меню
        await context.bot.send_message(chat_id=user.id, text=text, reply_markup=main_keyboard())

# =========================
# Профиль
# =========================
async def _render_profile_html(user_id: int) -> str:
    prem = await is_premium(user_id)
    used_today = await get_usage_today(user_id)
    bonus = await get_free_credits(user_id)

    if prem:
        left_text = "∞ (Премиум)"
        status = "Премиум"
    else:
        left_day = max(0, DAILY_LIMIT - used_today)
        total_left = left_day + bonus
        left_text = f"{total_left} (дневной лимит {left_day}, бонусов {bonus})"
        status = "Обычный"

    me = await application.bot.get_me()
    deep_link = f"https://t.me/{me.username}?start=ref_{user_id}"

    return (
        f"👤 <b>Профиль</b>\n"
        f"ID: <code>{user_id}</code>\n"
        f"Статус: <b>{status}</b>\n"
        f"Осталось заявок: <b>{left_text}</b>\n\n"
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
        try:
            await q.answer()
        except Exception:
            pass
        txt = await _render_profile_html(q.from_user.id)
        try:
            await q.message.edit_text(txt, parse_mode="HTML", reply_markup=main_keyboard())
        except Exception:
            await q.message.reply_text(txt, parse_mode="HTML", reply_markup=main_keyboard())
    except Exception as e:
        try:
            await q.answer(f"Ошибка: {e}", show_alert=True)
        except Exception:
            pass

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
        try:
            await q.answer()
        except Exception:
            pass
        txt = await _render_referral_html(q.from_user.id)
        try:
            await q.message.edit_text(txt, parse_mode="HTML", reply_markup=main_keyboard())
        except Exception:
            await q.message.reply_text(txt, parse_mode="HTML", reply_markup=main_keyboard())
    except Exception as e:
        try:
            await q.answer(f"Ошибка: {e}", show_alert=True)
        except Exception:
            pass

# =========================
# Выбор модели
# =========================
async def on_models_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        try:
            await q.answer()
        except Exception:
            pass
        try:
            await q.message.edit_text("Выбери модель:", reply_markup=models_keyboard())
        except Exception:
            await q.message.reply_text("Выбери модель:", reply_markup=models_keyboard())
    except Exception:
        pass

async def on_model_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        try:
            await q.answer()
        except Exception:
            pass
        if q.data == "m:oai":
            _user_model[q.from_user.id] = MODEL_OPENAI
            msg = "✅ Модель установлена: OpenAI · GPT-4o-mini"
        elif q.data == "m:ds":
            _user_model[q.from_user.id] = MODEL_DEEPSEEK
            msg = "✅ Модель установлена: DeepSeek · Chat"
        else:
            msg = "Неизвестная модель."
        try:
            await q.message.edit_text(msg, reply_markup=main_keyboard())
        except Exception:
            await q.message.reply_text(msg, reply_markup=main_keyboard())
    except Exception:
        pass

# =========================
# Оплата (CryptoPay)
# =========================
async def on_buy_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        try:
            await q.answer()
        except Exception:
            pass

        if not CRYPTOPAY_KEY:
            await q.message.reply_text("💳 Оплата не подключена (нет CRYPTOPAY_KEY).")
            return

        payload = str(q.from_user.id)
        headers = {"Crypto-Pay-API-Token": CRYPTOPAY_KEY}
        data = {
            "asset": "USDT",
            "amount": "3",
            "description": "Подписка на 30 дней",
            "payload": payload,
        }
        r = requests.post("https://pay.crypt.bot/api/createInvoice", json=data, headers=headers, timeout=15)
        j = r.json()
        url = j["result"]["pay_url"]
        await q.message.reply_text(f"💳 Оплати подписку по ссылке:\n{url}")
    except Exception as e:
        await q.message.reply_text(f"❌ Не удалось создать счёт: {e}")

# =========================
# Сообщения пользователей
# =========================
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or ""

    if await is_premium(user_id):
        reply = ask_llm(user_id, text)
        await update.message.reply_text(reply)
        return

    if await can_send_message(user_id, limit=DAILY_LIMIT):
        reply = ask_llm(user_id, text)
        await update.message.reply_text(reply)
        return

    if await consume_free_credit(user_id):
        reply = ask_llm(user_id, text)
        await update.message.reply_text(reply)
        return

    await update.message.reply_text(
        "🚫 Лимит исчерпан.\n"
        f"— Дневной лимит: {DAILY_LIMIT}/день\n"
        f"— Реферальные бонусы: получите +{REF_BONUS} заявок за каждого приглашённого!\n\n"
        "Купите подписку «💳 Купить подписку» для безлимита."
    )

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
            await update.message.reply_text("Формат: /remove_premium <user_id>")
            return
        uid = int(context.args[0])
        # снять премиум — ставим истёкшую дату
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
    # простая рассылка: всем платникам (демо) — можно расширить
    # здесь для краткости просто подтверждаем
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
    global application
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
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text="✅ Оплата получена! Подписка активирована на 30 дней."
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
    """
    Раз в 10 минут проверяем webhook и чинем, если он слетел.
    """
    await asyncio.sleep(8)
    while True:
        try:
            bot = application.bot
            me = await bot.get_me()
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

    # кнопки
    app_.add_handler(CallbackQueryHandler(on_buy_btn,      pattern=r"^buy$"))
    app_.add_handler(CallbackQueryHandler(on_profile_btn,  pattern=r"^profile$"))
    app_.add_handler(CallbackQueryHandler(on_ref_btn,      pattern=r"^ref$"))
    app_.add_handler(CallbackQueryHandler(on_models_btn,   pattern=r"^models$"))
    app_.add_handler(CallbackQueryHandler(on_model_select, pattern=r"^m:(oai|ds)$"))
    app_.add_handler(CallbackQueryHandler(
        lambda u, c: u.callback_query.message.edit_text("Главное меню:", reply_markup=main_keyboard()),
        pattern=r"^home$"
    ))

    # сообщения
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

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
    # guard-корутина
    asyncio.get_event_loop().create_task(_webhook_guard_loop())

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
