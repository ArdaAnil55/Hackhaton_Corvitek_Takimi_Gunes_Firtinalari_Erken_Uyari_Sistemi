# ai_analyst.py  —  v3.1 (Quota Defender Sürümü)
# ================================================
# DÜZELTMELER:
# - Model Tipi: Yorumlarda belirtildiği gibi gerçekten gemini-1.5-flash yapıldı.
# - Rate Limiter (Hard Cooldown): Kp dalgalanmalarının API kotasını eritmesini engellemek için eklendi.

import google.generativeai as genai
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("Sunny")

# ── Tek Satır Yorum Önbelleği & Rate Limiter ──────────────────────────────────
_AI_CACHE        = {"response": None, "timestamp": 0, "last_kp": None}
_CACHE_TTL_S     = 300   # Normal şartlarda 5 dakika önbellek
_KP_CHANGE_MIN   = 1.0   # Bu kadar Kp değişirse önbelleği geç
_HARD_COOLDOWN_S = 20    # GÜVENLİK ZIRHI: İki API çağrısı arasında en az 20 saniye geçmek ZORUNDA.


class AIAnalystService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        # Son API çağrı zamanını takip etmek için (Rate Limiting)
        self.last_api_call = 0 
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # DÜZELTİLDİ: 2.0-flash yerine ücretsiz kotası geniş olan 1.5-flash
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            logger.info("Sunny AI: Gemini 2.5 Flash bağlandı.")
        else:
            logger.warning("Sunny AI: GEMINI_API_KEY yok → Demo Modu.")

    # ── Yorum Önbelleği Kontrolü ──────────────────────────────────────────────
    def _should_refresh(self, kp: float) -> bool:
        current_time = time.time()
        
        # HARD COOLDOWN: Eğer son API çağrısından bu yana 20 saniye geçmediyse, 
        # Kp ne kadar değişirse değişsin API'ye gitmeyi REDDET.
        if current_time - self.last_api_call < _HARD_COOLDOWN_S:
            return False

        if _AI_CACHE["response"] is None:
            return True
        if current_time - _AI_CACHE["timestamp"] > _CACHE_TTL_S:
            return True
        if _AI_CACHE["last_kp"] is not None and abs(kp - _AI_CACHE["last_kp"]) >= _KP_CHANGE_MIN:
            return True
            
        return False

    def _call_gemini(self, prompt: str) -> str:
        """Gemini'ye istek atar; hata durumunda açıklayıcı mesaj döner."""
        try:
            self.last_api_call = time.time() # Çağrı zamanını kaydet
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            logger.error(f"Gemini hatası: {err}")
            if "429" in err or "quota" in err.lower():
                if _AI_CACHE["response"]:
                    return _AI_CACHE["response"] + " *(Kota sınırı — önbellekten gösteriliyor)*"
                return "Sunny: Sistem geçici olarak kota sınırında. Birkaç saniye bekleyin."
            return f"AI modülü geçici devre dışı: {err[:120]}"

    # ── Anlık Dashboard Yorumu ────────────────────────────────────────────────
    def get_latest_interpretation(self, solar_data: dict) -> str:
        kp           = solar_data.get("fused_kp")
        predicted_kp = solar_data.get("predicted_kp_3h")
        trend        = solar_data.get("trend", "→ Stabil")
        wind         = solar_data.get("solar_wind", {}) or {}
        sources      = solar_data.get("sources", {}) or {}

        if kp is None:
            return "Veri füzyonu tamamlanamadı."

        wind_ok = bool(wind) and wind.get("bz") is not None

        # Demo Modu
        if not self.api_key:
            pred_str = f" | Tahmin: Kp {predicted_kp:.1f}" if predicted_kp is not None else ""
            if kp >= 7 or (predicted_kp and predicted_kp >= 7):
                return f"[DEMO] Sunny: Kritik radyasyon! Kp={kp:.1f}{pred_str}. Elektrik şebekelerinde %85 risk. {trend}"
            elif kp >= 5 or (predicted_kp and predicted_kp >= 5):
                return f"[DEMO] Sunny: Güneş rüzgarlarında artış — Kp={kp:.1f}{pred_str}. Uydu parazit riski. {trend}"
            else:
                return f"[DEMO] Sunny: Uzay havası sakin — Kp={kp:.1f}{pred_str}. Tüm sistemler güvenli. {trend}"

        # Önbellek & Rate Limit kontrolü
        if not self._should_refresh(kp):
            logger.info(f"Sunny: Önbellekten yanıt veya Cooldown aktif.")
            return _AI_CACHE["response"] or "Sunny: Veriler analiz ediliyor..."

        # Prompt oluştur
        if predicted_kp is not None and wind_ok:
            block = (
                f"- Anlık Kp      : {kp:.2f}\n"
                f"- 3h Tahmin     : {predicted_kp:.2f} (XGBoost)\n"
                f"- Bz_GSM        : {wind.get('bz', 'N/A')} nT\n"
                f"- Hız           : {wind.get('speed', 'N/A')} km/s\n"
                f"- Yoğunluk      : {wind.get('density', 'N/A')} p/cc\n"
                f"- Trend         : {trend}\n"
                f"- Kaynaklar     : {sources}"
            )
        elif predicted_kp is not None:
            block = f"- Anlık Kp: {kp:.2f} | 3h Tahmin: {predicted_kp:.2f} | Trend: {trend}"
        else:
            block = f"- Anlık Kp: {kp:.2f} (XGBoost modeli veri biriktiriyor)"

        prompt = (
            "Sen SolarSentinel erken uyarı ağının asistanı Sunny'sin.\n"
            f"Güncel veriler:\n{block}\n\n"
            "Görev: Mevcut ile 3h tahmin farkını, Bz negatif+yüksek hız tehlikesini değerlendir. "
            "Max 3 kısa cümle, Türkçe, kontrol merkezi ciddiyetinde."
        )

        result = self._call_gemini(prompt)
        if "kota" not in result.lower() and "devre dışı" not in result.lower():
            _AI_CACHE["response"]  = result
            _AI_CACHE["timestamp"] = time.time()
            _AI_CACHE["last_kp"]   = kp
        return result

    # ── chat() metodu ──
    def chat(self, history: list, solar_context: str = "") -> str:
        if not history:
            return "Merhaba! Ben Sunny. Size nasıl yardımcı olabilirim?"

        if not self.api_key:
            last_msg = history[-1].get("content", "")
            return f"[DEMO] Sunny: '{last_msg}' sorunuzu aldım. Gerçek yanıt için API KEY ekleyin."

        system_parts = [
            "Sen SolarSentinel uzay havası erken uyarı sisteminin AI asistanı Sunny'sin.",
            "Kısa, net, kontrol merkezi ciddiyetinde Türkçe yanıtlar ver.",
        ]
        if solar_context:
            system_parts.append(f"Güncel sistem durumu: {solar_context}")
        system_prompt = " ".join(system_parts)

        gemini_history = []
        for msg in history[:-1]:
            role    = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "")
            if content:
                gemini_history.append({"role": role, "parts": [content]})

        last_user_msg = history[-1].get("content", "")

        try:
            self.last_api_call = time.time() # Chat için de çağrı zamanını kaydet
            if gemini_history:
                chat_session = self.model.start_chat(history=gemini_history)
                full_prompt  = f"{system_prompt}\n\nKullanıcı: {last_user_msg}"
                response     = chat_session.send_message(full_prompt)
            else:
                full_prompt = f"{system_prompt}\n\nKullanıcı: {last_user_msg}"
                response = self.model.generate_content(full_prompt)

            return response.text.strip()

        except Exception as e:
            err = str(e)
            logger.error(f"Chat Gemini hatası: {err}")
            if "429" in err or "quota" in err.lower():
                return "Sunny: Şu an çok fazla soru sordunuz, sistem kendini korumaya aldı. Lütfen 15-20 saniye bekleyip tekrar yazın."
            return f"Bağlantı hatası: {err[:100]}"