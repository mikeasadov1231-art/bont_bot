import logging
import re
import os
import threading
import requests
from io import BytesIO
from flask import Flask, request, jsonify

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

import cv2
import numpy as np
from PIL import Image
import pytesseract

# ---TOKEN = "7941821777:AAFqurGA-6lAx6JqOyuAy0gYO2hf-Wc93jA"

PORT = int(os.environ.get("PORT", 10000))

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- ВЕБ-СЕРВЕР ДЛЯ ПИНГОВ -------------------
app_flask = Flask(__name__)

@app_flask.route('/')
def index():
    return "Бот работает", 200

@app_flask.route('/health')
def health():
    return "OK", 200

# ------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -------------------

def get_central_rate():
    return 16.0

def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary

def extract_bonds_from_ocr(text):
    bonds = []
    lines = text.split('\n')
    for line in lines:
        if not any(key in line for key in ['ОФЗ', 'SU', 'RU', 'БО', 'ПБ']):
            continue
        numbers = re.findall(r'\d+[,.]\d+', line)
        if len(numbers) >= 4:
            try:
                price = float(numbers[0].replace(',', '.'))
                ytm = float(numbers[1].replace(',', '.'))
                coupon = float(numbers[2].replace(',', '.'))
                duration = float(numbers[3].replace(',', '.'))
                rating_match = re.search(r'[AА][+-]?|BBB|BB|B|C', line)
                rating = rating_match.group(0) if rating_match else None
                is_ofz = 'ОФЗ' in line or 'SU' in line
                bonds.append({
                    'name': line.strip(),
                    'price': price,
                    'ytm': ytm,
                    'coupon': coupon,
                    'duration': duration,
                    'rating': rating,
                    'is_ofz': is_ofz
                })
            except:
                continue
    return bonds

# ------------------- ЛОГИКА ВЫБОРА (ВАШ АЛГОРИТМ) -------------------

def select_best_ofz(bonds):
    filtered = [b for b in bonds if b.get('is_ofz', False) and b['price'] <= 98 and b['duration'] >= 2.5]
    if not filtered:
        return None
    best = max(filtered, key=lambda x: (x['ytm'], -x['price'], x['coupon'], -x['duration']))
    equal = [b for b in filtered if b['ytm'] == best['ytm'] and b['price'] == best['price'] and b['coupon'] == best['coupon'] and b['duration'] == best['duration']]
    return equal if len(equal)>1 else best

def select_best_corp(bonds):
    filtered = []
    for b in bonds:
        if b.get('is_ofz', False):
            continue
        if b['duration'] < 2.5:
            continue
        rating = b.get('rating', '')
        if rating and re.match(r'^A[+-]?$|^AA[+-]?$|^AAA$', rating):
            filtered.append(b)
    if not filtered:
        return None
    best = max(filtered, key=lambda x: (x['ytm'], -x['price'], x['coupon'], -x['duration']))
    equal = [b for b in filtered if b['ytm'] == best['ytm'] and b['price'] == best['price'] and b['coupon'] == best['coupon'] and b['duration'] == best['duration']]
    return equal if len(equal)>1 else best

def decide(bonds, key_rate):
    ofz_candidates = select_best_ofz(bonds)
    if ofz_candidates is None:
        corp = select_best_corp(bonds)
        if corp is None:
            return None, "Нет подходящих бумаг"
        else:
            return corp, "Корпоративная облигация (нет ОФЗ)"
    if isinstance(ofz_candidates, list):
        best_ofz = ofz_candidates[0]
    else:
        best_ofz = ofz_candidates
    if best_ofz['ytm'] >= key_rate:
        return ofz_candidates, f"ОФЗ (YTM >= ставка {key_rate}%)"
    else:
        corp = select_best_corp(bonds)
        if corp is None:
            return ofz_candidates, f"ОФЗ (корпоратов нет)"
        if isinstance(corp, list):
            best_corp = corp[0]
        else:
            best_corp = corp
        spread = round(best_corp['ytm'] - best_ofz['ytm'], 2)
        if spread >= 2.5:
            return corp, f"Корпорат (спред {spread} >= 2.5)"
        elif 1.5 <= spread < 2.5:
            if best_corp['price'] < best_ofz['price']:
                return corp, f"Корпорат (спред {spread}, цена корпората ниже)"
            else:
                return ofz_candidates, f"ОФЗ (спред {spread}, цена корпората не ниже)"
        else:
            return ofz_candidates, f"ОФЗ (спред {spread} < 1.5)"

# ------------------- ОБРАБОТЧИКИ TELEGRAM -------------------

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Привет! Отправьте мне скриншоты с таблицами облигаций (ОФЗ и корпоративные).\n"
        "Я распознаю данные и выберу лучшую бумагу по вашему алгоритму.\n"
        "Можно отправить несколько фото. Для расчёта введите /calculate."
    )

async def handle_image(update: Update, context: CallbackContext):
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте фото.")
        return
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()
    processed = preprocess_image(image_bytes)
    custom_config = r'--oem 3 --psm 6 -l rus'
    text = pytesseract.image_to_string(processed, config=custom_config)
    logger.info(f"OCR распознал:\n{text}")
    bonds = extract_bonds_from_ocr(text)
    if not bonds:
        await update.message.reply_text("Не удалось распознать данные. Попробуйте другое фото или введите данные вручную (команда /manual).")
        return
    if 'bonds' not in context.user_data:
        context.user_data['bonds'] = []
    context.user_data['bonds'].extend(bonds)
    await update.message.reply_text(f"Распознано {len(bonds)} облигаций. Отправьте ещё фото или введите /calculate.")

async def calculate(update: Update, context: CallbackContext):
    bonds = context.user_data.get('bonds', [])
    if not bonds:
        await update.message.reply_text("Нет данных. Сначала отправьте скриншоты.")
        return
    key_rate = get_central_rate()
    if key_rate is None:
        await update.message.reply_text("Не удалось получить ключевую ставку. Введите её вручную командой /setrate 16.0")
        return
    result, reason = decide(bonds, key_rate)
    if result is None:
        await update.message.reply_text("Ни одна бумага не подходит по условиям.")
        return
    if isinstance(result, list):
        names = [b['name'] for b in result]
        msg = "Покупаем равными долями:\n" + "\n".join(names)
        msg += f"\n\nОбоснование: {reason}"
        msg += f"\nПараметры (первой): цена {result[0]['price']}%, YTM {result[0]['ytm']}%, дюрация {result[0]['duration']} лет"
    else:
        msg = f"Рекомендация: {result['name']}\n"
        msg += f"Цена: {result['price']}%\nYTM: {result['ytm']}%\nКупон: {result['coupon']}%\nДюрация: {result['duration']} лет\n"
        msg += f"Обоснование: {reason}"
    await update.message.reply_text(msg)
    context.user_data['bonds'] = []

async def setrate(update: Update, context: CallbackContext):
    try:
        rate = float(context.args[0])
        context.user_data['key_rate'] = rate
        await update.message.reply_text(f"Ключевая ставка установлена: {rate}%")
    except:
        await update.message.reply_text("Используйте: /setrate 16.0")

async def manual(update: Update, context: CallbackContext):
    await update.message.reply_text("Ручной ввод пока не реализован. Отправьте скриншот.")

# ------------------- ЗАПУСК БОТА И ВЕБ-СЕРВЕРА -------------------

def run_bot():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("calculate", calculate))
    app.add_handler(CommandHandler("manual", manual))
    app.add_handler(CommandHandler("setrate", setrate))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    app_flask.run(host="0.0.0.0", port=PORT)
