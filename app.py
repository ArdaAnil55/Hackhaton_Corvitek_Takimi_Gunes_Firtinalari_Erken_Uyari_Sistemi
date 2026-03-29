"""
app.py  —  v2.3 (Chat Proxy Eklendi)
======================================

DÜZELTMELER v2.2 → v2.3:
  [BUG-12] Frontend doğrudan Anthropic API çağırıyordu → güvenlik açığı.
           Çözüm: /api/chat endpoint'i eklendi; tüm chat trafiği Gemini üzerinden
           backend'e yönlendirildi. API anahtarı asla tarayıcıya açılmıyor.

  [BUG-2 / BUG-9] Zaten düzeltilmiş: render_template + 18_19.html template adı.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import asyncio
import time
import logging
from dotenv import load_dotenv

load_dotenv()

from data_solar_fetcher import SolarDataService
from ai_analyst import AIAnalystService

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("SolarApp")

app = Flask(__name__, template_folder="templates")
CORS(app)

solar_service = SolarDataService()
ai_service    = AIAnalystService()

CACHE     = {"data": None, "timestamp": 0}
CACHE_TTL = 60   # saniye

# ── Ana Sayfa ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('18_19.html')

# ── Dashboard API ─────────────────────────────────────────────────────────────
@app.route('/api/dashboard', methods=['GET'])
def dashboard_api():
    current_time = time.time()

    if CACHE["data"] and (current_time - CACHE["timestamp"] < CACHE_TTL):
        logger.info("Önbellekten yanıt.")
        return jsonify(CACHE["data"])

    try:
        solar_data = asyncio.run(solar_service.fetch_and_fuse())
    except Exception as e:
        logger.error(f"Veri çekme hatası: {e}")
        if CACHE["data"]:
            stale = dict(CACHE["data"])
            stale["warning"] = "Canlı veri alınamadı; son önbellek gösteriliyor."
            return jsonify(stale), 206
        return jsonify({"error": "Veri servisleri kullanılamıyor."}), 503

    try:
        solar_data["ai_interpretation"] = ai_service.get_latest_interpretation(solar_data)
    except Exception as e:
        logger.error(f"AI yorum hatası: {e}")
        solar_data["ai_interpretation"] = "Yapay zeka modülü geçici olarak devre dışı."

    CACHE["data"]      = solar_data
    CACHE["timestamp"] = current_time
    logger.info(f"Önbellek güncellendi. Kp={solar_data.get('fused_kp')}")
    return jsonify(solar_data)

# ── Chat Proxy API ─────────────────────────────────────────────────────────────
# [BUG-12 DÜZELTİLDİ]: Tarayıcıdan direkt Anthropic çağrısı yerine,
# bu endpoint tüm chat mesajlarını Gemini üzerinden yönetir.
# API anahtarı sadece sunucu tarafında .env'de saklanır.
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json() or {}
    history      = data.get('history', [])
    solar_context = data.get('solar_context', 'Veri yok')

    if not history:
        return jsonify({"error": "Boş mesaj geçmişi"}), 400

    # Mevcut dashboard verisini solar_context'e ekleyerek zenginleştir
    if CACHE["data"]:
        d = CACHE["data"]
        cached_ctx = (
            f"Kp={d.get('fused_kp','?')} | "
            f"Tahmin={d.get('predicted_kp_3h','?')} | "
            f"Trend={d.get('trend','?')} | "
            f"Alarm={d.get('alert_level','?')}"
        )
        # Frontend'den gelen context ile birleştir (frontend daha güncel olabilir)
        solar_context = solar_context + " || Backend: " + cached_ctx

    try:
        reply = ai_service.chat(history, solar_context)
        return jsonify({"reply": reply})
    except Exception as e:
        logger.error(f"Chat API hatası: {e}")
        return jsonify({"reply": f"Sistem hatası: {str(e)[:120]}"}), 500

# ── Sağlık Kontrolü ───────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status":       "ok",
        "cache_age_s":  round(time.time() - CACHE["timestamp"], 1),
        "cache_valid":  CACHE["data"] is not None,
        "model_loaded": True,
    })

if __name__ == '__main__':
    logger.info("Solar Sentinel v2.3 başlatılıyor...")
    app.run(debug=True, port=5000)