"""
data_solar_fetcher.py  —  v4.0 (62 Özellikli Delta XGBoost Sürümü)
================================================================
"""

import asyncio
import httpx
import logging
import pickle
import os
import collections
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("SolarFetcher")

# ─── MODEL YÜKLEME ───────────────────────────────────────────────────────────
_model      = None
_feat_cols  = None
_model_meta = {}

try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solar_xgboost_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            _model_meta = pickle.load(f)

        if isinstance(_model_meta, dict) and "model" in _model_meta:
            _model     = _model_meta["model"]
            _feat_cols = _model_meta.get("feature_cols")
            logger.info(f"Model yüklendi | {_model_meta.get('n_features', len(_feat_cols) if _feat_cols else '?')} özellik")
        else:
            _model     = _model_meta
            _feat_cols = None
            logger.warning("Eski model formatı.")
    else:
        logger.warning(f"Model bulunamadı: {model_path}")
except Exception as e:
    logger.error(f"Model yüklenemedi: {e}")

# ─── ROLLING CACHE ───────────────────────────────────────────────────────────
_CACHE_MINUTES = 1440   # 24 saat
_wind_cache = collections.deque(maxlen=_CACHE_MINUTES)

def _push_cache(bx: float, by: float, bz: float, bt: float, speed: float, density: float, kp: float | None):
    _wind_cache.append({
        "ts": datetime.now(timezone.utc),
        "bx_gsm": bx, "by_gsm": by, "bz_gsm": bz, "bt": bt,
        "speed": speed, "density": density, "kp": kp or 0.0,
    })

def _rolling_stat(key: str, hours: int, stat: str = "mean") -> float | None:
    if not _wind_cache:
        return None
    now  = datetime.now(timezone.utc)
    vals = [
        r[key] for r in _wind_cache
        if (now - r["ts"]) <= timedelta(hours=hours) and r[key] is not None and not np.isnan(r[key])
    ]
    if not vals:
        return None
    return float({"mean": np.mean, "std": np.std, "min": np.min}[stat](vals))

def _lag_val(key: str, lag_h: int, fallback: float) -> float:
    if not _wind_cache:
        return fallback
    now = datetime.now(timezone.utc)
    candidates = [
        r for r in _wind_cache
        if abs((now - r["ts"]).total_seconds() / 3600 - lag_h) < 1.5
    ]
    return candidates[-1][key] if candidates else fallback

# ─── 62 FEATURE BUILDER ──────────────────────────────────────────────────────
def build_realtime_features(bx: float, by: float, bz: float, bt: float, speed: float, density: float, kp_now: float) -> dict:
    
    def roll(key, hours, stat="mean"):
        fallback = {"bx_gsm": bx, "by_gsm": by, "bz_gsm": bz, "bt": bt, "speed": speed, "density": density, "kp": kp_now}[key]
        v = _rolling_stat(key, hours, stat)
        return v if v is not None else fallback

    def lag(key, hours):
        fallback = {"bx_gsm": bx, "by_gsm": by, "bz_gsm": bz, "bt": bt, "speed": speed, "density": density, "kp": kp_now}[key]
        return _lag_val(key, hours, fallback)

    ram = 1.67e-6 * density * (speed ** 2)
    VBs = speed * max(-bz, 0)
    
    now = datetime.now(timezone.utc)
    hour = now.hour
    doy = now.timetuple().tm_yday

    feat = {
        "kp": kp_now, "bt": bt, "bx_gsm": bx, "by_gsm": by, "bz_gsm": bz, "speed": speed, "density": density,

        "kp_1step_mean": roll("kp", 3), "kp_2step_mean": roll("kp", 6), "kp_4step_mean": roll("kp", 12), "kp_8step_mean": roll("kp", 24),
        "bz_1step_mean": roll("bz_gsm", 3), "bz_2step_mean": roll("bz_gsm", 6), "bz_4step_mean": roll("bz_gsm", 12), "bz_8step_mean": roll("bz_gsm", 24),
        "bz_1step_std": roll("bz_gsm", 3, "std") or 0.0, "bz_2step_std": roll("bz_gsm", 6, "std") or 0.0, "bz_4step_std": roll("bz_gsm", 12, "std") or 0.0,
        "bz_1step_min": roll("bz_gsm", 3, "min"), "bz_2step_min": roll("bz_gsm", 6, "min"), "bz_4step_min": roll("bz_gsm", 12, "min"),
        
        "speed_1step_mean": roll("speed", 3), "speed_2step_mean": roll("speed", 6), "speed_4step_mean": roll("speed", 12), "speed_8step_mean": roll("speed", 24),
        "density_1step_mean": roll("density", 3), "density_2step_mean": roll("density", 6), "density_4step_mean": roll("density", 12),
        "bt_1step_mean": roll("bt", 3), "bt_2step_mean": roll("bt", 6), "bt_4step_mean": roll("bt", 12),

        "dbz_1step": bz - lag("bz_gsm", 3), "dbz_2step": bz - lag("bz_gsm", 6), "dbz_4step": bz - lag("bz_gsm", 12),
        "dby_1step": by - lag("by_gsm", 3), "dbt_1step": bt - lag("bt", 3),
        "dkp_1step": kp_now - lag("kp", 3), "dkp_2step": kp_now - lag("kp", 6),
        "kp_vol_4step": roll("kp", 12, "std") or 0.0,

        "ram_pressure": ram, "VBs": VBs,
        "VBs_2step_mean": float(np.mean([r["speed"] * max(-r["bz_gsm"], 0) for r in _wind_cache if (now - r["ts"]).total_seconds()/3600 <= 6] or [VBs])),
        "VBs_4step_mean": float(np.mean([r["speed"] * max(-r["bz_gsm"], 0) for r in _wind_cache if (now - r["ts"]).total_seconds()/3600 <= 12] or [VBs])),
        "bz_sq": bz ** 2, "bt_sq": bt ** 2,

        "kp_lag_1": lag("kp", 3), "kp_lag_2": lag("kp", 6), "kp_lag_4": lag("kp", 12), "kp_lag_8": lag("kp", 24),
        "bz_lag_1": lag("bz_gsm", 3), "bz_lag_2": lag("bz_gsm", 6), "bz_lag_4": lag("bz_gsm", 12), "bz_lag_8": lag("bz_gsm", 24),
        "by_lag_1": lag("by_gsm", 3), "by_lag_2": lag("by_gsm", 6),
        "speed_lag_1": lag("speed", 3), "speed_lag_2": lag("speed", 6), "speed_lag_4": lag("speed", 12),

        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }
    return feat

def _classify_xray(flux: float) -> str:
    if   flux >= 1e-4: return "X"
    elif flux >= 1e-5: return "M"
    elif flux >= 1e-6: return "C"
    elif flux >= 1e-7: return "B"
    else:              return "A"

# ─── NOAA / NASA API Sınıfı ──────────────────────────────────────────────────
class SolarDataService:
    async def _get_noaa_kp(self) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", timeout=8.0)
                resp.raise_for_status()
                return {"value": float(resp.json()[-1][1]), "status": "success"}
        except Exception as e:
            return {"value": None, "status": f"error:{e}"}

    async def _get_solar_wind(self) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                mag_r, pla_r = await asyncio.gather(
                    client.get("https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json", timeout=8.0),
                    client.get("https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json", timeout=8.0),
                )
                mag_r.raise_for_status()
                pla_r.raise_for_status()
                mag_row = mag_r.json()[-1]
                pla_row = pla_r.json()[-1]
                # API Düzeltildi: 1=bx, 2=by, 3=bz, 6=bt
                return {"bx": float(mag_row[1]), "by": float(mag_row[2]), "bz": float(mag_row[3]), "bt": float(mag_row[6]), "density": float(pla_row[1]), "speed": float(pla_row[2]), "status": "success"}
        except Exception as e:
            return {"bx": None, "by": None, "bz": None, "bt": None, "speed": None, "density": None, "status": "fallback"}

    async def _get_xray(self) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json", timeout=8.0)
                resp.raise_for_status()
            long_ch = [r for r in resp.json() if isinstance(r, dict) and r.get("energy") == "0.1-0.8nm" and r.get("flux", 0) > 0]
            if not long_ch: return {"cls": None, "flux": None, "status": "no_data"}
            flux = float(long_ch[-1]["flux"])
            return {"cls": _classify_xray(flux), "flux": flux, "status": "success"}
        except Exception as e:
            return {"cls": None, "flux": None, "status": f"error:{e}"}

    async def _get_cme(self) -> dict:
        try:
            now = datetime.now(timezone.utc)
            url = f"https://api.nasa.gov/DONKI/CME?startDate={(now - timedelta(days=10)).strftime('%Y-%m-%d')}&endDate={now.strftime('%Y-%m-%d')}&api_key=KSmuczA4ikoefpbLMdw2XtYNekXtCe3UUooL98B2"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=15.0)
                resp.raise_for_status()
            return {"count": len(resp.json()) if isinstance(resp.json(), list) else 0, "status": "success"}
        except Exception as e:
            return {"count": 0, "status": f"error:{e}"}

    async def fetch_and_fuse(self) -> dict:
        noaa_data, wind_data, xray_data, cme_data = await asyncio.gather(
            self._get_noaa_kp(), self._get_solar_wind(), self._get_xray(), self._get_cme()
        )

        noaa_kp = noaa_data.get("value", 4.0)
        bx      = wind_data.get("bx", 0.0)
        by      = wind_data.get("by", 0.0)
        bz      = wind_data.get("bz", 0.0)
        bt      = wind_data.get("bt", 0.0)
        speed   = wind_data.get("speed", 400.0)
        density = wind_data.get("density", 5.0)
        fused_kp = round(noaa_kp, 2) if noaa_kp is not None else 4.0

        if bz is not None and wind_data.get("status") != "fallback":
            _push_cache(bx, by, bz, bt, speed, density, noaa_kp)

        # ── XGBoost Tahmini (DELTA MODELİ) ──
        predicted_kp = None
        prediction_reliable = False
        raw_delta = 0.0

        if _model is not None and _feat_cols is not None:
            try:
                safe_bx = bx if bx is not None else 0.0
                safe_by = by if by is not None else 0.0
                safe_bz = bz if bz is not None else 0.0
                safe_bt = bt if bt is not None else 0.0
                safe_speed = speed if speed is not None else 400.0
                safe_density = density if density is not None else 5.0
                
                feat_dict = build_realtime_features(safe_bx, safe_by, safe_bz, safe_bt, safe_speed, safe_density, fused_kp)

                input_df = pd.DataFrame([feat_dict], columns=_feat_cols).astype(float)
                raw_delta = float(_model.predict(input_df)[0])
                
                # Hedef Değişim (Delta) olduğu için mevcut Kp'nin üzerine ekliyoruz!
                predicted_kp = round(float(np.clip(fused_kp + raw_delta, 0.0, 9.0)), 2)
                print(f"\n--- DEBUG EKRANI ---")
                print(f"NOAA'dan Gelen Mevcut Kp : {fused_kp}")
                print(f"Modelin Tahmin Ettiği Delta : {raw_delta}")
                print(f"Toplam Tahmin (Kp + Delta)  : {predicted_kp}")
                print(f"--------------------\n")
                prediction_reliable = True 

            except Exception as e:
                logger.error(f"XGBoost Hatası: {e}")

        diff = (predicted_kp if predicted_kp is not None else fused_kp) - fused_kp
        if diff > 0.3:    trend = "↑ Artış"
        elif diff < -0.3: trend = "↓ Düşüş"
        else:             trend = "→ Stabil"

        alert_val = predicted_kp if predicted_kp is not None else fused_kp
        if alert_val >= 7:   alert = "🔴 KRİTİK"
        elif alert_val >= 5: alert = "🟠 UYARI"
        else:                alert = "🟢 SAKİN"

        return {
            "timestamp":           datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "fused_kp":            fused_kp,
            "noaa_kp":             noaa_kp,
            "solar_wind": {
                "bx":      bx,
                "by":      by,
                "bz":      bz,
                "bt":      bt,
                "speed":   speed,
                "density": density,
            },
            "predicted_delta":     round(raw_delta, 3) if prediction_reliable else None,
            "predicted_kp_3h":     predicted_kp,
            "prediction_reliable": prediction_reliable,
            "trend":               trend,
            "alert_level":         alert,
            "cache_size":          len(_wind_cache),
            "xray_cls":            xray_data.get("cls"),
            "xray_flux":           xray_data.get("flux"),
            "cme_count":           cme_data.get("count", 0),
            "sources": {
                "noaa_kp":    noaa_data["status"],
                "solar_wind": wind_data["status"],
                "xray":       xray_data.get("status", "unknown"),
                "cme":        cme_data.get("status", "unknown"),
            },
        }

if __name__ == "__main__":
    print(asyncio.run(SolarDataService().fetch_and_fuse()))