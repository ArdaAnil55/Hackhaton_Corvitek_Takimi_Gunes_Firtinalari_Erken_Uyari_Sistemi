import asyncio
import httpx
import io
import pickle
import shutil
import warnings
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("SolarDeltaTrainer")

CFG = {
    "years": list(range(2017, 2027)),
    "omni_url_tpl": "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
    "cache_dir": Path("omni_cache"),
    "model_output": "solar_xgboost_model.pkl",
    "forecast_horizon_steps": 1,   # 3 saatlik çözünürlükte 1 step = +3h
    "resample_freq": "3h",
    "roll_windows": [1, 2, 4, 8],  # 3h, 6h, 12h, 24h
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "param_grid": [
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 650, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 6, "learning_rate": 0.02, "n_estimators": 800, "subsample": 0.85, "colsample_bytree": 0.85},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 450, "subsample": 0.9, "colsample_bytree": 0.8},
    ],
}

OMNI_COLS = {
    "year":    0,
    "doy":     1,
    "hour":    2,
    "bt":      9,
    "bx_gsm": 14,
    "by_gsm": 15,
    "bz_gsm": 16,
    "density": 23,
    "speed":   24,
    "kp10":    38,
}

OMNI_FILL = {
    "bt":      9999.9,
    "bx_gsm":  9999.9,
    "by_gsm":  9999.9,
    "bz_gsm":  9999.9,
    "density":  999.9,
    "speed":   9999.0,
    "kp10":      990,
}

FEATURE_COLS = [
    "kp",
    "bt", "bx_gsm", "by_gsm", "bz_gsm", "speed", "density",

    "kp_1step_mean", "kp_2step_mean", "kp_4step_mean", "kp_8step_mean",
    "bz_1step_mean", "bz_2step_mean", "bz_4step_mean", "bz_8step_mean",
    "bz_1step_std", "bz_2step_std", "bz_4step_std",
    "bz_1step_min", "bz_2step_min", "bz_4step_min",
    "speed_1step_mean", "speed_2step_mean", "speed_4step_mean", "speed_8step_mean",
    "density_1step_mean", "density_2step_mean", "density_4step_mean",
    "bt_1step_mean", "bt_2step_mean", "bt_4step_mean",

    "dbz_1step", "dbz_2step", "dbz_4step",
    "dby_1step", "dbt_1step",
    "dkp_1step", "dkp_2step",
    "kp_vol_4step",

    "ram_pressure", "VBs", "VBs_2step_mean", "VBs_4step_mean",
    "bz_sq", "bt_sq",

    "kp_lag_1", "kp_lag_2", "kp_lag_4", "kp_lag_8",
    "bz_lag_1", "bz_lag_2", "bz_lag_4", "bz_lag_8",
    "by_lag_1", "by_lag_2", "speed_lag_1", "speed_lag_2", "speed_lag_4",

    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

TARGET_COL = "target_delta_kp_3h"


async def _fetch_year(client: httpx.AsyncClient, year: int) -> bytes | None:
    url = CFG["omni_url_tpl"].format(year=year)
    try:
        resp = await client.get(url, timeout=90.0)
        resp.raise_for_status()
        log.info(f"✔ {year} indirildi")
        return resp.content
    except Exception as e:
        log.error(f"✘ {year} indirilemedi: {e}")
        return None


async def download_omni_data() -> dict[int, bytes]:
    CFG["cache_dir"].mkdir(exist_ok=True)
    results = {}
    to_download = []

    for year in CFG["years"]:
        cache_file = CFG["cache_dir"] / f"omni2_{year}.dat"
        root_file = Path(f"omni2_{year}.dat")

        if cache_file.exists():
            results[year] = cache_file.read_bytes()
        elif root_file.exists():
            shutil.copy(root_file, cache_file)
            results[year] = cache_file.read_bytes()
        else:
            to_download.append(year)

    if to_download:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            raw_list = await asyncio.gather(*[_fetch_year(client, y) for y in to_download])

        for year, raw in zip(to_download, raw_list):
            if raw:
                cache_file = CFG["cache_dir"] / f"omni2_{year}.dat"
                cache_file.write_bytes(raw)
                results[year] = raw

    return results


def parse_omni_bytes(raw: bytes) -> pd.DataFrame:
    idx = list(OMNI_COLS.values())
    names = list(OMNI_COLS.keys())

    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", engine="python", header=None)
    df = df[idx].copy()
    df.columns = names

    df["timestamp"] = pd.to_datetime(
        df["year"].astype(str)
        + df["doy"].astype(str).str.zfill(3)
        + df["hour"].astype(str).str.zfill(2),
        format="%Y%j%H",
        utc=True,
    )

    df = df.drop(columns=["year", "doy", "hour"]).set_index("timestamp").sort_index()
    df["kp"] = df["kp10"] / 10.0
    df = df.drop(columns=["kp10"])

    for col, threshold in OMNI_FILL.items():
        if col == "kp10":
            df.loc[df["kp"] >= threshold / 10, "kp"] = np.nan
        else:
            df.loc[df[col].abs() >= threshold, col] = np.nan

    return df


def load_all_years(raw_data: dict[int, bytes]) -> pd.DataFrame:
    frames = []
    for year in sorted(raw_data.keys()):
        try:
            frames.append(parse_omni_bytes(raw_data[year]))
        except Exception as e:
            log.error(f"{year} parse hatası: {e}")
    return pd.concat(frames).sort_index()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    df = df.reindex(full_range)

    for col in ["bt", "bx_gsm", "by_gsm", "bz_gsm"]:
        df.loc[df[col].abs() > 100, col] = np.nan

    df.loc[(df["speed"] < 200) | (df["speed"] > 3000), "speed"] = np.nan
    df.loc[(df["density"] < 0) | (df["density"] > 200), "density"] = np.nan
    df.loc[(df["kp"] < 0) | (df["kp"] > 9), "kp"] = np.nan

    cols = ["bt", "bx_gsm", "by_gsm", "bz_gsm", "speed", "density", "kp"]
    df[cols] = df[cols].interpolate(method="time", limit=3)

    return df


def to_3hour_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.resample(CFG["resample_freq"]).mean().index)
    for col in ["bt", "bx_gsm", "by_gsm", "bz_gsm", "speed", "density", "kp"]:
        out[col] = df[col].resample(CFG["resample_freq"]).mean()
    return out.dropna(subset=["bt", "by_gsm", "bz_gsm", "speed", "density", "kp"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    for w in CFG["roll_windows"]:
        label = f"{w}step"
        feat[f"kp_{label}_mean"] = feat["kp"].rolling(w, min_periods=1).mean()
        feat[f"bz_{label}_mean"] = feat["bz_gsm"].rolling(w, min_periods=1).mean()
        feat[f"bz_{label}_std"] = feat["bz_gsm"].rolling(w, min_periods=1).std().fillna(0.0)
        feat[f"bz_{label}_min"] = feat["bz_gsm"].rolling(w, min_periods=1).min()
        feat[f"speed_{label}_mean"] = feat["speed"].rolling(w, min_periods=1).mean()
        feat[f"density_{label}_mean"] = feat["density"].rolling(w, min_periods=1).mean()
        feat[f"bt_{label}_mean"] = feat["bt"].rolling(w, min_periods=1).mean()

    feat["dbz_1step"] = feat["bz_gsm"].diff(1)
    feat["dbz_2step"] = feat["bz_gsm"].diff(2)
    feat["dbz_4step"] = feat["bz_gsm"].diff(4)
    feat["dby_1step"] = feat["by_gsm"].diff(1)
    feat["dbt_1step"] = feat["bt"].diff(1)

    feat["dkp_1step"] = feat["kp"].diff(1)
    feat["dkp_2step"] = feat["kp"].diff(2)
    feat["kp_vol_4step"] = feat["kp"].rolling(4, min_periods=1).std().fillna(0.0)

    feat["ram_pressure"] = 1.67e-6 * feat["density"] * (feat["speed"] ** 2)
    feat["VBs"] = feat["speed"] * np.maximum(-feat["bz_gsm"], 0)
    feat["VBs_2step_mean"] = feat["VBs"].rolling(2, min_periods=1).mean()
    feat["VBs_4step_mean"] = feat["VBs"].rolling(4, min_periods=1).mean()
    feat["bz_sq"] = feat["bz_gsm"] ** 2
    feat["bt_sq"] = feat["bt"] ** 2

    for lag in [1, 2, 4, 8]:
        feat[f"kp_lag_{lag}"] = feat["kp"].shift(lag)
        feat[f"bz_lag_{lag}"] = feat["bz_gsm"].shift(lag)

    for lag in [1, 2]:
        feat[f"by_lag_{lag}"] = feat["by_gsm"].shift(lag)

    for lag in [1, 2, 4]:
        feat[f"speed_lag_{lag}"] = feat["speed"].shift(lag)

    hour = feat.index.hour
    doy = feat.index.dayofyear
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 3 saatlik kısa vadeli tahmin için uç delta'ları sınırlı hedef uzayında öğret
    raw_delta = feat["kp"].shift(-CFG["forecast_horizon_steps"]) - feat["kp"]
    feat[TARGET_COL] = raw_delta.clip(lower=-0.8, upper=0.8)

    return feat


def prepare_dataset(feat: pd.DataFrame):
    cols_needed = FEATURE_COLS + [TARGET_COL]
    df_model = feat[cols_needed].dropna().copy()
    return df_model[FEATURE_COLS], df_model[TARGET_COL]


def split_dataset(X: pd.DataFrame, y: pd.Series):
    n = len(X)
    train_end = int(n * CFG["train_ratio"])
    val_end = train_end + int(n * CFG["val_ratio"])
    return (
        X.iloc[:train_end], y.iloc[:train_end],
        X.iloc[train_end:val_end], y.iloc[train_end:val_end],
        X.iloc[val_end:], y.iloc[val_end:]
    )


def make_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Küçük ve orta değişimleri daha iyi öğret.
    Amaç: 3 saatlik tahminde aşırı uçuk delta'ları azaltmak.
    """
    abs_y = np.abs(y.values)
    w = np.ones(len(y), dtype=float)

    w[abs_y <= 0.20] = 2.4
    w[(abs_y > 0.20) & (abs_y <= 0.50)] = 1.8
    w[(abs_y > 0.50) & (abs_y <= 0.80)] = 1.1

    return w


def train_and_select_model(X_train, y_train, X_val, y_val):
    best_model = None
    best_score = np.inf
    best_params = None

    train_weights = make_sample_weights(y_train)
    val_weights = make_sample_weights(y_val)

    for params in CFG["param_grid"]:
        log.info(f"Parametre deneniyor: {params}")

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=50,
            reg_alpha=0.2,
            reg_lambda=2.0,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[val_weights],
            verbose=False,
        )

        pred_delta = model.predict(X_val)
        mae_delta = mean_absolute_error(y_val, pred_delta)
        rmse_delta = np.sqrt(mean_squared_error(y_val, pred_delta))

        big_pred_penalty = float(np.mean(np.abs(pred_delta) > 0.7))
        score = 0.55 * mae_delta + 0.35 * rmse_delta + 0.10 * big_pred_penalty

        log.info(
            f"Val MAE(delta): {mae_delta:.4f} | "
            f"Val RMSE(delta): {rmse_delta:.4f} | "
            f"LargeΔPenalty: {big_pred_penalty:.4f}"
        )

        if score < best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params


def evaluate_model(model, X_test, y_test):
    pred_delta = model.predict(X_test)

    mae_delta = mean_absolute_error(y_test, pred_delta)
    rmse_delta = np.sqrt(mean_squared_error(y_test, pred_delta))
    abs_err = np.abs(pred_delta - y_test.values)

    acc_pm_025 = float(np.mean(abs_err <= 0.25))
    acc_pm_050 = float(np.mean(abs_err <= 0.50))
    large_delta_rate = float(np.mean(np.abs(pred_delta) > 0.7))

    log.info("=" * 60)
    log.info(f"MAE(delta)   : {mae_delta:.4f}")
    log.info(f"RMSE(delta)  : {rmse_delta:.4f}")
    log.info(f"±0.25 acc    : {acc_pm_025:.3f}")
    log.info(f"±0.50 acc    : {acc_pm_050:.3f}")
    log.info(f"|predΔ|>0.7  : {large_delta_rate:.3f}")
    log.info("=" * 60)

    return {
        "mae_delta": mae_delta,
        "rmse_delta": rmse_delta,
        "acc_pm_025": acc_pm_025,
        "acc_pm_050": acc_pm_050,
        "large_delta_rate": large_delta_rate,
    }


def save_model(model, metrics, best_params):
    meta = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "forecast_horizon_hours": 3,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "version": "6.2_DELTA_XGB_CLIPPED",
        "target_type": "delta_kp_clipped",
        "best_params": best_params,
        "metrics": metrics,
        "resample_freq": CFG["resample_freq"],
    }

    with open(CFG["model_output"], "wb") as f:
        pickle.dump(meta, f)

    log.info(f"Model kaydedildi → {CFG['model_output']}")


async def main():
    raw_data = await download_omni_data()
    df_raw = load_all_years(raw_data)
    df_clean = clean_data(df_raw)
    df_3h = to_3hour_dataframe(df_clean)
    df_feat = build_features(df_3h)
    X, y = prepare_dataset(df_feat)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    model, best_params = train_and_select_model(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, metrics, best_params)


if __name__ == "__main__":
    asyncio.run(main())