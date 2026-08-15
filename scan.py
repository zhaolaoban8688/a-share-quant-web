#!/usr/bin/env python3
"""A股沪深主板四买点结构扫描器 V5.0.

核心规则：近期在MA20之上出现放量破前高上涨；随后缩量回调。
买点一=回踩MA20且收盘不破MA20；买点二=跌破MA20后回踩邻近前高；
买点三=继续回踩邻近前低；买点四=前低失效后逐级寻找更低历史波谷。
所有买点仅认可最新K线：收阳、实体大于上一根、成交量大于上一根。
输出 data/latest.json、data/candidates.csv 与候选SVG日K图。

仅用于研究与复盘，不构成投资建议。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests

SH_TZ = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/125.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}
_thread = threading.local()


def session() -> requests.Session:
    if not hasattr(_thread, "session"):
        s = requests.Session()
        s.headers.update(UA)
        _thread.session = s
    return _thread.session


def num(v: Any, default: float = math.nan) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pct(x: float) -> float:
    return round(x * 100.0, 2) if math.isfinite(x) else math.nan


def safe_round(x: Any, n: int = 2) -> Any:
    x = num(x)
    return round(x, n) if math.isfinite(x) else None


def retry_call(fn, attempts: int = 3, base_sleep: float = 1.2):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i + 1 < attempts:
                time.sleep(base_sleep * (2**i) + random.random())
    raise last  # type: ignore[misc]


def normalize_spot(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Eastmoney/Sina AKShare spot columns and unify volume to hands.

    新浪成交量通常以“股”为单位，东财/腾讯日K多以“手”为单位。旧版仅按
    全市场成交量中位数判断，遇到缩量日或数据分布变化时可能整批误判，导致
    当日成交量被放大约100倍，所有股票都被“放量回调”条件淘汰。
    V3.4改为逐股用 成交额≈价格×成交量×100 自动判断单位。
    """
    df = df.copy()
    rename = {
        "代码": "code",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "change_pct",
        "成交量": "volume",
        "成交额": "amount",
        "最高": "high",
        "最低": "low",
        "今开": "open",
        "昨收": "prev_close",
        "量比": "volume_ratio",
        "换手率": "turnover",
        "60日涨跌幅": "r60_spot",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = ["code", "name", "price", "amount", "high", "low", "open", "prev_close"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df["code"] = (
        df["code"].astype(str).str.lower().str.replace("sh", "", regex=False)
        .str.replace("sz", "", regex=False).str.replace("bj", "", regex=False)
        .str.extract(r"(\d{6})", expand=False)
    )
    numeric_cols = [
        "price", "change_pct", "volume", "amount", "high", "low", "open",
        "prev_close", "volume_ratio", "turnover", "r60_spot",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (df["price"] > 0) & (df["volume"] > 0) & (df["amount"] > 0)
    # amount / (price * volume): 若volume为股，约等于1；若volume为手，约等于100。
    unit_ratio = df.loc[valid, "amount"] / (
        df.loc[valid, "price"] * df.loc[valid, "volume"]
    ).replace(0, np.nan)
    share_unit = unit_ratio.between(0.15, 8.0, inclusive="both")
    df.loc[unit_ratio.index[share_unit], "volume"] = (
        df.loc[unit_ratio.index[share_unit], "volume"] / 100.0
    )
    return df


def get_spot() -> tuple[pd.DataFrame, str, list[str]]:
    warnings: list[str] = []
    try:
        df = retry_call(lambda: ak.stock_zh_a_spot_em(), attempts=3)
        return normalize_spot(df), "AKShare-东方财富", warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"东方财富实时快照失败：{type(exc).__name__}")
    try:
        df = retry_call(lambda: ak.stock_zh_a_spot(), attempts=2, base_sleep=2)
        return normalize_spot(df), "AKShare-新浪", warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"新浪实时快照失败：{type(exc).__name__}")
        raise RuntimeError("两个免费实时快照源均失败，请稍后重试") from exc


def market_symbol(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def parse_tencent_payload(payload: dict[str, Any], sym: str) -> pd.DataFrame:
    node = payload.get("data", {}).get(sym, {})
    arr = node.get("qfqday") or node.get("day") or node.get("hfqday") or []
    rows = []
    for r in arr:
        if len(r) < 6:
            continue
        rows.append(
            {
                "date": str(r[0]),
                "open": num(r[1]),
                "close": num(r[2]),
                "high": num(r[3]),
                "low": num(r[4]),
                "volume": num(r[5]),
                "amount": num(r[6]) if len(r) > 6 else math.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("腾讯日K为空")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "open", "close", "high", "low", "volume"]).sort_values("date")


def fetch_tencent_history(code: str, bars: int = 280) -> pd.DataFrame:
    sym = market_symbol(code)
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{sym},day,,,{bars},qfq"}
    r = session().get(url, params=params, timeout=12)
    r.raise_for_status()
    return parse_tencent_payload(r.json(), sym)


def fetch_ak_history(code: str, bars: int = 280) -> pd.DataFrame:
    end = datetime.now(SH_TZ).strftime("%Y%m%d")
    start = (datetime.now(SH_TZ) - timedelta(days=max(500, bars * 2))).strftime("%Y%m%d")
    raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if raw is None or raw.empty:
        raise ValueError("AKShare历史日K为空")
    ren = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
    df = raw.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "open", "close", "high", "low", "volume"]).sort_values("date").tail(bars)


def fetch_history(code: str, bars: int = 280) -> tuple[str, pd.DataFrame | None, str | None]:
    try:
        df = retry_call(lambda: fetch_tencent_history(code, bars), attempts=3, base_sleep=0.5)
        return code, df, None
    except Exception as t_exc:  # noqa: BLE001
        try:
            df = retry_call(lambda: fetch_ak_history(code, bars), attempts=2, base_sleep=1.2)
            return code, df, f"腾讯失败，已回退AKShare：{type(t_exc).__name__}"
        except Exception as a_exc:  # noqa: BLE001
            return code, None, f"历史行情失败：{type(t_exc).__name__}/{type(a_exc).__name__}"


def merge_spot_bar(hist: pd.DataFrame, row: pd.Series, scan_time: datetime) -> pd.DataFrame:
    """Merge snapshot into current date bar without corrupting volume units.

    若腾讯/AKShare历史日K已经含有当天K线，优先保留历史源成交量；仅用实时
    快照更新收盘价及高低价。这样即使新浪快照偶发单位异常，也不会让当天量能
    被放大100倍。盘中尚无当日历史K线时，再追加已经自动换算为“手”的快照。
    """
    h = hist.copy().reset_index(drop=True)
    today = pd.Timestamp(scan_time.date())
    # 周末不把周五快照伪造成周末K线。法定休市日若快照与历史末价几乎一致，也保留历史末K。
    if scan_time.weekday() >= 5:
        return h.tail(360).reset_index(drop=True)
    price, opn, high, low = [num(row.get(c)) for c in ("price", "open", "high", "low")]
    vol, amount = num(row.get("volume"), 0), num(row.get("amount"), 0)
    if not all(math.isfinite(x) and x > 0 for x in (price, opn, high, low)):
        return h.tail(360).reset_index(drop=True)

    if not h.empty and pd.Timestamp(h.iloc[-1]["date"]).normalize() == today:
        i = h.index[-1]
        old_open = num(h.at[i, "open"])
        old_high = num(h.at[i, "high"])
        old_low = num(h.at[i, "low"])
        old_vol = num(h.at[i, "volume"], 0)
        old_amount = num(h.at[i, "amount"], 0)
        h.at[i, "open"] = old_open if math.isfinite(old_open) and old_open > 0 else opn
        h.at[i, "close"] = price
        h.at[i, "high"] = max(high, old_high if math.isfinite(old_high) else high)
        h.at[i, "low"] = min(low, old_low if math.isfinite(old_low) and old_low > 0 else low)
        # 两源量能相差20倍以上时视为单位异常，保留历史源；否则取较完整者。
        if old_vol > 0 and vol > 0 and max(old_vol, vol) / max(min(old_vol, vol), 1e-9) > 20:
            h.at[i, "volume"] = old_vol
        else:
            h.at[i, "volume"] = max(old_vol, vol)
        h.at[i, "amount"] = max(old_amount, amount)
    else:
        if not h.empty:
            last_close = num(h.iloc[-1]["close"])
            last_date = pd.Timestamp(h.iloc[-1]["date"]).normalize()
            if last_date < today and last_close > 0 and abs(price / last_close - 1) < 0.0005:
                return h.tail(360).reset_index(drop=True)
        new = {
            "date": today, "open": opn, "close": price, "high": high, "low": low,
            "volume": max(vol, 0), "amount": max(amount, 0),
        }
        h = pd.concat([h, pd.DataFrame([new])], ignore_index=True)
    return h.tail(360).reset_index(drop=True)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    for n in (5, 10, 20, 60):
        d[f"ma{n}"] = d["close"].rolling(n).mean()
    d["vma5"] = d["volume"].rolling(5).mean()
    d["vma10"] = d["volume"].rolling(10).mean()
    d["vma20"] = d["volume"].rolling(20).mean()
    prev = d["close"].shift(1)
    tr = pd.concat([(d["high"] - d["low"]), (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    dif = ema(d["close"], 12) - ema(d["close"], 26)
    dea = ema(dif, 9)
    d["dif"], d["dea"], d["macd_hist"] = dif, dea, 2 * (dif - dea)
    d["r5"] = d["close"].pct_change(5)
    d["r10"] = d["close"].pct_change(10)
    d["r20"] = d["close"].pct_change(20)
    d["r60"] = d["close"].pct_change(60)
    return d





@dataclass
class Impulse:
    start_idx: int
    breakout_idx: int
    peak_idx: int
    prior_peak_idx: int
    prior_trough_idx: int
    prior_high: float
    prior_trough: float
    start_low: float
    peak_high: float
    gain: float
    breakout_volume_ratio: float
    impulse_volume_ratio: float
    above_ma20_ratio: float
    impulse_volume_median: float


@dataclass
class StructuralLevel:
    idx: int
    kind: str
    price: float
    move_before: float
    move_after: float
    validations: int = 1


@dataclass
class BuySetup:
    buy_point: int
    state: str
    signal: str
    impulse: Impulse
    pullback_days: int
    drawdown: float
    contraction: float
    support_name: str
    support_price: float
    support_distance: float
    support_idx: int
    touch_idx: int
    confirmation_idx: int
    confirmation_body_ratio: float
    confirmation_body_pct: float
    confirmation_volume_ratio: float
    confirmation_return: float
    broke_ma20: bool
    support_validations: int
    confirmation_above_ma20: bool


def local_peak(d: pd.DataFrame, idx: int, radius: int = 3) -> bool:
    if idx < radius or idx + radius >= len(d):
        return False
    window = d.iloc[idx - radius:idx + radius + 1]
    v = num(d.iloc[idx]["high"])
    return math.isfinite(v) and v >= num(window["high"].max()) * 0.998


def local_low(d: pd.DataFrame, idx: int, radius: int = 3) -> bool:
    if idx < radius or idx + radius >= len(d):
        return False
    window = d.iloc[idx - radius:idx + radius + 1]
    v = num(d.iloc[idx]["low"])
    return math.isfinite(v) and v <= num(window["low"].min()) * 1.002


def body_size(row: pd.Series) -> float:
    return abs(num(row["close"]) - num(row["open"]))


def is_confirmation_candle(d: pd.DataFrame, idx: int) -> tuple[bool, float, float, float, float]:
    """最新K线必须：阳线、实体大于上一根、成交量大于上一根。"""
    if idx < 1:
        return False, 0.0, 0.0, 0.0, 0.0
    cur, prev = d.iloc[idx], d.iloc[idx - 1]
    opn, close, vol = num(cur["open"]), num(cur["close"]), num(cur["volume"])
    prev_open, prev_close, prev_vol = num(prev["open"]), num(prev["close"]), num(prev["volume"])
    if not all(math.isfinite(x) and x > 0 for x in (opn, close, vol, prev_open, prev_close, prev_vol)):
        return False, 0.0, 0.0, 0.0, 0.0
    cur_body = close - opn
    prev_body = abs(prev_close - prev_open)
    body_ratio = cur_body / max(prev_body, opn * 0.0005)
    body_pct = cur_body / opn
    vol_ratio = vol / prev_vol
    ret = close / prev_close - 1
    ok = cur_body > 0 and cur_body > prev_body and vol > prev_vol
    return ok, body_ratio, body_pct, vol_ratio, ret


def structural_peak_quality(d: pd.DataFrame, idx: int, breakout_idx: int) -> tuple[bool, float, float, int, float]:
    """判断被突破的前高是否足够明显，并返回峰前涨幅、峰后回调及前低。"""
    if not local_peak(d, idx, 3) or breakout_idx - idx < 4:
        return False, 0.0, 0.0, -1, math.nan
    left = d.iloc[max(0, idx - 30):idx]
    right = d.iloc[idx + 1:breakout_idx]
    if len(left) < 8 or len(right) < 3:
        return False, 0.0, 0.0, -1, math.nan
    peak = num(d.iloc[idx]["high"])
    left_low = num(left["low"].min())
    trough_idx = int(right["low"].idxmin())
    trough = num(d.iloc[trough_idx]["low"])
    rise_before = peak / max(left_low, 1e-9) - 1
    pullback = 1 - trough / max(peak, 1e-9)
    ok = rise_before >= 0.05 and pullback >= 0.03
    return ok, rise_before, pullback, trough_idx, trough


def find_impulses(d: pd.DataFrame, confirm_idx: int) -> list[Impulse]:
    """近期必须存在：MA20之上、放量收盘突破明显前高的一段上涨。"""
    out: list[Impulse] = []
    left_bound = max(35, confirm_idx - 120)
    right_bound = confirm_idx - 3
    if right_bound <= left_bound:
        return out

    for breakout_idx in range(left_bound, right_bound + 1):
        row = d.iloc[breakout_idx]
        close, opn, ma20, vol = [num(row[c]) for c in ("close", "open", "ma20", "volume")]
        if not all(math.isfinite(x) and x > 0 for x in (close, opn, ma20, vol)):
            continue
        if close <= ma20 or close <= opn:
            continue

        prior_slice = d.iloc[max(25, breakout_idx - 55):breakout_idx - 3]
        if len(prior_slice) < 18:
            continue
        peak_candidates = [
            i for i in range(prior_slice.index.min(), prior_slice.index.max() + 1)
            if local_peak(d, i, 3)
        ]
        peak_candidates.sort(reverse=True)
        chosen = None
        for prior_peak_idx in peak_candidates:
            ok_peak, _, _, prior_trough_idx, prior_trough = structural_peak_quality(d, prior_peak_idx, breakout_idx)
            if not ok_peak:
                continue
            prior_high = num(d.iloc[prior_peak_idx]["high"])
            if close < prior_high * 1.002:
                continue
            chosen = (prior_peak_idx, prior_trough_idx, prior_high, prior_trough)
            break
        if chosen is None:
            continue
        prior_peak_idx, prior_trough_idx, prior_high, prior_trough = chosen

        base_vol = num(d.iloc[max(20, breakout_idx - 12):breakout_idx]["volume"].median())
        cluster = d.iloc[breakout_idx:min(confirm_idx, breakout_idx + 3)]
        breakout_ratio = max(
            vol / max(base_vol, 1e-9),
            num(cluster["volume"].mean()) / max(base_vol, 1e-9) if len(cluster) else 0,
        )
        if not math.isfinite(base_vol) or base_vol <= 0 or breakout_ratio < 1.15:
            continue

        peak_right = min(confirm_idx - 1, breakout_idx + 45)
        if peak_right <= breakout_idx:
            continue
        peak_idx = int(d.iloc[breakout_idx:peak_right + 1]["high"].idxmax())
        peak_high = num(d.iloc[peak_idx]["high"])
        gain = peak_high / max(prior_trough, 1e-9) - 1
        if gain < 0.08 or peak_high < prior_high * 1.025:
            continue

        impulse_bars = d.iloc[breakout_idx:peak_idx + 1]
        if len(impulse_bars) < 3:
            continue
        above_ratio = float((impulse_bars["close"] >= impulse_bars["ma20"]).mean())
        up_days = int((impulse_bars["close"] > impulse_bars["open"]).sum())
        if above_ratio < 0.60 or up_days < 2:
            continue
        impulse_vol_median = num(impulse_bars["volume"].median())
        impulse_vol_ratio = impulse_vol_median / max(base_vol, 1e-9)
        start_idx = prior_trough_idx
        out.append(Impulse(
            start_idx=start_idx,
            breakout_idx=breakout_idx,
            peak_idx=peak_idx,
            prior_peak_idx=prior_peak_idx,
            prior_trough_idx=prior_trough_idx,
            prior_high=prior_high,
            prior_trough=prior_trough,
            start_low=prior_trough,
            peak_high=peak_high,
            gain=gain,
            breakout_volume_ratio=breakout_ratio,
            impulse_volume_ratio=impulse_vol_ratio,
            above_ma20_ratio=above_ratio,
            impulse_volume_median=impulse_vol_median,
        ))

    out.sort(key=lambda x: (x.peak_idx, x.gain, x.breakout_volume_ratio), reverse=True)
    unique: list[Impulse] = []
    for item in out:
        if any(abs(item.peak_idx - old.peak_idx) <= 3 and abs(item.prior_peak_idx - old.prior_peak_idx) <= 5 for old in unique):
            continue
        unique.append(item)
        if len(unique) >= 12:
            break
    return unique


def find_structural_troughs(d: pd.DataFrame, end_idx: int, lookback: int = 320) -> list[StructuralLevel]:
    """识别结构明显的历史波谷：前有下跌、后有反弹，非单日噪声。"""
    out: list[StructuralLevel] = []
    start = max(20, end_idx - lookback)
    stop = min(end_idx - 4, len(d) - 5)
    for i in range(start, stop + 1):
        if not local_low(d, i, 3):
            continue
        left = d.iloc[max(0, i - 30):i]
        right = d.iloc[i + 1:min(len(d), i + 31)]
        if len(left) < 8 or len(right) < 6:
            continue
        price = num(d.iloc[i]["low"])
        decline = 1 - price / max(num(left["high"].max()), 1e-9)
        rebound = num(right["high"].max()) / max(price, 1e-9) - 1
        if decline >= 0.05 and rebound >= 0.06:
            out.append(StructuralLevel(i, "历史波谷", price, decline, rebound, 1))
    out.sort(key=lambda x: x.idx, reverse=True)
    return out


def count_support_validations(d: pd.DataFrame, level: float, start_idx: int, end_idx: int, tol: float = 0.03) -> int:
    """统计支撑位被结构性低点验证的次数；相邻触点至少间隔5日。"""
    hits: list[int] = []
    for i in range(max(start_idx, 3), min(end_idx, len(d) - 4)):
        if not local_low(d, i, 2):
            continue
        low = num(d.iloc[i]["low"])
        if abs(low / max(level, 1e-9) - 1) > tol:
            continue
        future = d.iloc[i + 1:min(len(d), i + 10)]
        if future.empty or num(future["high"].max()) / max(low, 1e-9) - 1 < 0.04:
            continue
        if not hits or i - hits[-1] >= 5:
            hits.append(i)
    return len(hits)


def horizontal_support_clear(d: pd.DataFrame, level: float, start_idx: int, end_idx: int) -> bool:
    """买点四：水平支撑线不可穿越实体，且形成后不得有收盘价跌破。"""
    if end_idx <= start_idx + 1:
        return True
    mid = d.iloc[start_idx + 1:end_idx]
    if mid.empty:
        return True
    body_low = mid[["open", "close"]].min(axis=1)
    body_high = mid[["open", "close"]].max(axis=1)
    crosses_body = (body_low < level) & (body_high > level)
    closes_below = mid["close"] < level
    return not bool(crosses_body.any() or closes_below.any())


def make_setup(
    *, d: pd.DataFrame, impulse: Impulse, buy_point: int, support_name: str,
    support_price: float, support_distance: float, support_idx: int, touch_idx: int,
    confirm_idx: int, pullback_days: int, drawdown: float, contraction: float,
    body_ratio: float, body_pct: float, volume_ratio: float, confirm_ret: float,
    broke_ma20: bool, validations: int = 1,
) -> BuySetup:
    labels = {
        1: "买点一·MA20回踩确认",
        2: "买点二·前高支撑确认",
        3: "买点三·前低支撑确认",
        4: "买点四·历史波谷支撑确认",
    }
    return BuySetup(
        buy_point=buy_point,
        state=f"B{buy_point}",
        signal=labels[buy_point],
        impulse=impulse,
        pullback_days=pullback_days,
        drawdown=drawdown,
        contraction=contraction,
        support_name=support_name,
        support_price=support_price,
        support_distance=support_distance,
        support_idx=support_idx,
        touch_idx=touch_idx,
        confirmation_idx=confirm_idx,
        confirmation_body_ratio=body_ratio,
        confirmation_body_pct=body_pct,
        confirmation_volume_ratio=volume_ratio,
        confirmation_return=confirm_ret,
        broke_ma20=broke_ma20,
        support_validations=validations,
        confirmation_above_ma20=num(d.iloc[confirm_idx]["close"]) >= num(d.iloc[confirm_idx]["ma20"]),
    )


def evaluate_setup_at(
    d: pd.DataFrame,
    confirm_idx: int,
    ma_tolerance: float,
    structure_tolerance: float,
    max_contraction: float,
) -> tuple[BuySetup | None, str]:
    """严格按买点一→二→三→四评估最新K线；不允许3日回看确认。"""
    ok, body_ratio, body_pct, volume_ratio, confirm_ret = is_confirmation_candle(d, confirm_idx)
    if not ok:
        return None, "最新K线未同时满足阳线+实体放大+量能放大"

    impulses = find_impulses(d, confirm_idx)
    if not impulses:
        return None, "近期无MA20上方放量破前高上涨段"

    best: BuySetup | None = None
    best_quality = -1e9
    for impulse in impulses:
        pullback_days = confirm_idx - impulse.peak_idx
        if not 2 <= pullback_days <= 45:
            continue
        pull = d.iloc[impulse.peak_idx + 1:confirm_idx + 1]
        pull_core = d.iloc[impulse.peak_idx + 1:confirm_idx]
        if pull_core.empty:
            continue

        min_close_idx = int(pull["close"].idxmin())
        min_close = num(d.iloc[min_close_idx]["close"])
        drawdown = 1 - min_close / max(impulse.peak_high, 1e-9)
        if drawdown < 0.01 or drawdown > 0.60:
            continue

        pull_vol = num(pull_core["volume"].median())
        contraction = pull_vol / max(impulse.impulse_volume_median, 1e-9)
        if not math.isfinite(contraction) or contraction > max_contraction:
            continue

        broke_ma20 = bool((pull["close"] < pull["ma20"]).any())
        latest = d.iloc[confirm_idx]
        latest_close = num(latest["close"])
        latest_ma20 = num(latest["ma20"])
        setup: BuySetup | None = None

        # 买点一：整个回调收盘不能跌破MA20；最低收盘距离当时MA20不超10%。
        min_ma20 = num(d.iloc[min_close_idx]["ma20"])
        ma_dist = abs(min_close / max(min_ma20, 1e-9) - 1) if min_ma20 > 0 else math.inf
        if not broke_ma20 and latest_close >= latest_ma20 and ma_dist <= ma_tolerance:
            setup = make_setup(
                d=d, impulse=impulse, buy_point=1, support_name="MA20",
                support_price=min_ma20, support_distance=ma_dist,
                support_idx=min_close_idx, touch_idx=min_close_idx, confirm_idx=confirm_idx,
                pullback_days=pullback_days, drawdown=drawdown, contraction=contraction,
                body_ratio=body_ratio, body_pct=body_pct, volume_ratio=volume_ratio,
                confirm_ret=confirm_ret, broke_ma20=False,
            )

        # 买点二：已跌破MA20，继续回踩本轮上涨突破的前高；支撑偏差≤5%。
        if setup is None and broke_ma20:
            level = impulse.prior_high
            dist = abs(min_close / max(level, 1e-9) - 1)
            if dist <= structure_tolerance and latest_close >= level * 0.95:
                setup = make_setup(
                    d=d, impulse=impulse, buy_point=2, support_name="邻近前高",
                    support_price=level, support_distance=dist,
                    support_idx=impulse.prior_peak_idx, touch_idx=min_close_idx, confirm_idx=confirm_idx,
                    pullback_days=pullback_days, drawdown=drawdown, contraction=contraction,
                    body_ratio=body_ratio, body_pct=body_pct, volume_ratio=volume_ratio,
                    confirm_ret=confirm_ret, broke_ma20=True,
                )

        # 买点三：已跌破MA20，继续回踩邻近上一波回调前低；偏差≤5%。
        if setup is None and broke_ma20:
            level = impulse.prior_trough
            dist = abs(min_close / max(level, 1e-9) - 1)
            if dist <= structure_tolerance and latest_close >= level * 0.95:
                setup = make_setup(
                    d=d, impulse=impulse, buy_point=3, support_name="邻近前低",
                    support_price=level, support_distance=dist,
                    support_idx=impulse.prior_trough_idx, touch_idx=min_close_idx, confirm_idx=confirm_idx,
                    pullback_days=pullback_days, drawdown=drawdown, contraction=contraction,
                    body_ratio=body_ratio, body_pct=body_pct, volume_ratio=volume_ratio,
                    confirm_ret=confirm_ret, broke_ma20=True,
                )

        # 买点四：邻近前低已失效后，逐级向前找更低历史波谷；至少两次验证，水平线不穿实体。
        if setup is None and broke_ma20 and min_close < impulse.prior_trough * 0.95:
            troughs = find_structural_troughs(d, impulse.prior_trough_idx)
            lower_chain: list[StructuralLevel] = []
            ceiling = impulse.prior_trough
            for lv in troughs:
                if lv.price < ceiling * 0.995:
                    lower_chain.append(lv)
                    ceiling = lv.price
            for lv in lower_chain:
                dist = abs(min_close / max(lv.price, 1e-9) - 1)
                if dist > structure_tolerance:
                    continue
                if latest_close < lv.price:  # 买点四收盘跌破支撑即失效
                    continue
                validations = count_support_validations(d, lv.price, lv.idx, min_close_idx + 1, tol=0.03)
                if validations < 2:
                    continue
                if not horizontal_support_clear(d, lv.price, lv.idx, min_close_idx):
                    continue
                setup = make_setup(
                    d=d, impulse=impulse, buy_point=4, support_name="历史波谷",
                    support_price=lv.price, support_distance=dist,
                    support_idx=lv.idx, touch_idx=min_close_idx, confirm_idx=confirm_idx,
                    pullback_days=pullback_days, drawdown=drawdown, contraction=contraction,
                    body_ratio=body_ratio, body_pct=body_pct, volume_ratio=volume_ratio,
                    confirm_ret=confirm_ret, broke_ma20=True, validations=validations,
                )
                break

        if setup is None:
            continue

        tol = ma_tolerance if setup.buy_point == 1 else structure_tolerance
        proximity = 1 - min(setup.support_distance / max(tol, 1e-9), 1)
        quality = (
            clamp((impulse.gain - 0.08) / 0.45, 0, 1) * 22
            + clamp((impulse.breakout_volume_ratio - 1.15) / 1.85, 0, 1) * 15
            + clamp((impulse.above_ma20_ratio - 0.60) / 0.40, 0, 1) * 10
            + clamp((max_contraction - contraction) / max(max_contraction - 0.30, 0.01), 0, 1) * 20
            + proximity * 15
            + clamp((body_ratio - 1.0) / 2.5, 0, 1) * 8
            + clamp((volume_ratio - 1.0) / 1.5, 0, 1) * 6
            + {1: 8, 2: 6, 3: 4, 4: 2}[setup.buy_point]
            + (4 if setup.buy_point == 2 and setup.confirmation_above_ma20 else 0)
        )
        if quality > best_quality:
            best_quality = quality
            best = setup

    if best is not None:
        return best, "已确认"
    return None, "有上涨段，但回调尚未同时满足缩量、支撑和失效规则"


def find_buy_setup(
    d: pd.DataFrame,
    ma_tolerance: float = 0.10,
    structure_tolerance: float = 0.05,
    max_contraction: float = 0.85,
) -> tuple[BuySetup | None, str]:
    if len(d) < 120:
        return None, "数据不足"
    return evaluate_setup_at(d, len(d) - 1, ma_tolerance, structure_tolerance, max_contraction)


def analyze_stock(
    code: str,
    name: str,
    industry: str,
    industry_score: float,
    d0: pd.DataFrame,
    market_r20: float,
    breadth: float,
    spot_row: pd.Series,
    tolerance: float,
    structure_tolerance: float,
    max_contraction: float,
) -> tuple[dict[str, Any] | None, str]:
    del market_r20, breadth
    d = add_indicators(d0)
    if len(d) < 120:
        return None, "数据不足"
    setup, stage = find_buy_setup(d, tolerance, structure_tolerance, max_contraction)
    if setup is None:
        return None, stage

    last = d.iloc[-1]
    price, ma20 = num(last["close"]), num(last["ma20"])
    impulse = setup.impulse

    # 最后一道硬失效校验，确保网页不出现违反用户规则的候选。
    if setup.buy_point == 1 and price < ma20:
        return None, "买点一失效·最新收盘跌破MA20"
    if setup.buy_point in {2, 3} and price < setup.support_price * 0.95:
        return None, f"买点{setup.buy_point}失效·收盘跌破支撑5%"
    if setup.buy_point == 4 and price < setup.support_price:
        return None, "买点四失效·收盘跌破历史波谷"

    tol = tolerance if setup.buy_point == 1 else structure_tolerance
    proximity = 1 - min(setup.support_distance / max(tol, 1e-9), 1)
    score = (
        clamp((impulse.gain - 0.08) / 0.45, 0, 1) * 24
        + clamp((impulse.breakout_volume_ratio - 1.15) / 1.85, 0, 1) * 16
        + clamp((1 - setup.contraction) / 0.70, 0, 1) * 22
        + proximity * 16
        + clamp((setup.confirmation_body_ratio - 1.0) / 2.5, 0, 1) * 9
        + clamp((setup.confirmation_volume_ratio - 1.0) / 1.5, 0, 1) * 7
        + {1: 8, 2: 6, 3: 4, 4: 2}[setup.buy_point]
        + (4 if setup.buy_point == 2 and setup.confirmation_above_ma20 else 0)
    )
    score = clamp(score, 0, 100)

    confirm_row = d.iloc[setup.confirmation_idx]
    trigger = num(confirm_row["high"]) * 1.002
    if setup.buy_point == 1:
        stop = ma20
        invalidation_rule = "收盘价跌破MA20"
    elif setup.buy_point in {2, 3}:
        stop = setup.support_price * 0.95
        invalidation_rule = f"收盘价跌破{setup.support_name}5%"
    else:
        stop = setup.support_price
        invalidation_rule = "收盘价跌破历史波谷支撑"
    if stop <= 0 or stop >= trigger:
        return None, "失效线不低于触发价"
    risk_pct = (trigger - stop) / trigger
    target = trigger + 2 * (trigger - stop)
    change_pct = num(spot_row.get("change_pct"))

    return {
        "code": code,
        "name": name,
        "industry": industry or "未分类",
        "industry_score": round(industry_score, 1),
        "state": setup.state,
        "buy_point": setup.buy_point,
        "signal": setup.signal,
        "score": round(score, 1),
        "price": round(price, 3),
        "change_pct": safe_round(change_pct, 2),
        "ma20": round(ma20, 3),
        "distance_ma20_pct": pct(price / ma20 - 1) if ma20 > 0 else None,
        "impulse_start_date": d.iloc[impulse.start_idx]["date"].strftime("%Y-%m-%d"),
        "prior_peak_date": d.iloc[impulse.prior_peak_idx]["date"].strftime("%Y-%m-%d"),
        "prior_peak_price": round(impulse.prior_high, 3),
        "prior_trough_date": d.iloc[impulse.prior_trough_idx]["date"].strftime("%Y-%m-%d"),
        "prior_trough_price": round(impulse.prior_trough, 3),
        "breakout_date": d.iloc[impulse.breakout_idx]["date"].strftime("%Y-%m-%d"),
        "impulse_peak_date": d.iloc[impulse.peak_idx]["date"].strftime("%Y-%m-%d"),
        "impulse_gain_pct": pct(impulse.gain),
        "breakout_volume_ratio": round(impulse.breakout_volume_ratio, 2),
        "above_ma20_ratio_pct": pct(impulse.above_ma20_ratio),
        "pullback_days": setup.pullback_days,
        "drawdown_pct": pct(setup.drawdown),
        "pullback_volume_ratio": round(setup.contraction, 2),
        "support_name": setup.support_name,
        "support_price": round(setup.support_price, 3),
        "support_date": d.iloc[setup.support_idx]["date"].strftime("%Y-%m-%d"),
        "support_distance_pct": pct(setup.support_distance),
        "support_validations": setup.support_validations,
        "confirmation_date": confirm_row["date"].strftime("%Y-%m-%d"),
        "confirmation_body_ratio": round(setup.confirmation_body_ratio, 2),
        "confirmation_body_pct": pct(setup.confirmation_body_pct),
        "confirmation_volume_ratio": round(setup.confirmation_volume_ratio, 2),
        "confirmation_return_pct": pct(setup.confirmation_return),
        "confirmation_above_ma20": setup.confirmation_above_ma20,
        "trigger": round(trigger, 3),
        "stop": round(stop, 3),
        "target_2r": round(target, 3),
        "risk_pct": pct(risk_pct),
        "risk_reward": 2.0,
        "invalidation_rule": invalidation_rule,
        "invalidation_level": round(stop, 3),
        "amount_yi": safe_round(num(spot_row.get("amount")) / 1e8, 2),
        "turnover": safe_round(spot_row.get("turnover"), 2),
        "volume_ratio_spot": safe_round(spot_row.get("volume_ratio"), 2),
        "chart_url": "",
        "reason": [
            f"{d.iloc[impulse.breakout_idx]['date']:%Y-%m-%d} 在MA20上方放量突破明显前高 {impulse.prior_high:.2f}，突破量比 {impulse.breakout_volume_ratio:.2f}",
            f"突破后上涨段最大涨幅 {pct(impulse.gain):.1f}%，MA20上方收盘占比 {pct(impulse.above_ma20_ratio):.1f}%",
            f"随后回调 {setup.pullback_days} 日，回撤 {pct(setup.drawdown):.1f}%，回调中位量缩至上涨段的 {setup.contraction:.2f}",
            f"回调最低收盘价距离 {setup.support_name} {setup.support_price:.2f} 为 {pct(setup.support_distance):.1f}%",
            f"确认日收阳，实体为上一根 {setup.confirmation_body_ratio:.2f} 倍，成交量为上一根 {setup.confirmation_volume_ratio:.2f} 倍",
            f"触发 {trigger:.2f} / 失效线 {stop:.2f}（{invalidation_rule}）/ 2R参考 {target:.2f}",
        ],
    }, stage


def build_industry_map(spot_codes: set[str], warnings: list[str]) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]]]:
    mapping: dict[str, str] = {}
    scores: dict[str, float] = {}
    ranking: list[dict[str, Any]] = []
    try:
        boards = retry_call(lambda: ak.stock_board_industry_name_em(), attempts=2, base_sleep=1.5)
        if boards is None or boards.empty:
            raise ValueError("行业板块为空")
        boards = boards.copy()
        vals = pd.to_numeric(boards["涨跌幅"], errors="coerce")
        ranks = vals.rank(pct=True) * 100
        for i, row in boards.iterrows():
            board_name = str(row.get("板块名称", "未分类"))
            scores[board_name] = num(ranks.loc[i], 50)
            ranking.append({
                "name": board_name,
                "change_pct": safe_round(row.get("涨跌幅"), 2),
                "score": round(scores[board_name], 1),
            })

        def one(row: pd.Series):
            board_name = str(row.get("板块名称"))
            board_code = str(row.get("板块代码"))
            try:
                c = ak.stock_board_industry_cons_em(symbol=board_code)
                return board_name, [str(x).zfill(6) for x in c.get("代码", [])]
            except Exception:
                return board_name, []

        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            for board_name, codes in ex.map(one, [r for _, r in boards.iterrows()]):
                for code in codes:
                    if code in spot_codes and code not in mapping:
                        mapping[code] = board_name
        ranking.sort(key=lambda x: x["score"], reverse=True)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"行业映射失败，行业仅显示为未分类：{type(exc).__name__}")
    return mapping, scores, ranking[:20]


def choose_universe(spot: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    d = spot.copy()
    mainboard_pattern = r"^(?:000|001|002|003|600|601|603|605)\d{3}$"
    d = d[d["code"].str.match(mainboard_pattern, na=False)]
    # 基础数据卫生：排除ST、退市整理、无成交数据，不属于策略条件。
    d = d[~d["name"].astype(str).str.contains(r"ST|退|N |C ", case=False, regex=True, na=False)]
    d = d[(d["price"] >= args.min_price) & (d["price"] <= args.max_price)]
    d = d[d["amount"].fillna(0) >= args.min_amount_yi * 1e8]
    d = d[d["volume"].fillna(0) > 0]
    d = d.sort_values("amount", ascending=False)
    if args.max_stocks > 0:
        d = d.head(args.max_stocks)
    return d.reset_index(drop=True)



def svg_escape(s: Any) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def render_candidate_chart(d0: pd.DataFrame, item: dict[str, Any], out_path: Path, bars: int = 120) -> None:
    """生成不依赖matplotlib的轻量SVG日K图：K线 + MA20 + 支撑 + 成交量。"""
    d = add_indicators(d0).copy().reset_index(drop=True)
    if d.empty:
        return
    # 从上涨段前约10日开始展示，同时限制最大K线数量。
    dates = pd.to_datetime(d["date"], errors="coerce")
    start_date = pd.to_datetime(item.get("impulse_start_date"), errors="coerce")
    start_idx = 0
    if pd.notna(start_date):
        idxs = d.index[dates >= start_date]
        if len(idxs):
            start_idx = max(0, int(idxs[0]) - 10)
    start_idx = max(start_idx, len(d) - bars)
    v = d.iloc[start_idx:].copy().reset_index(drop=True)
    if len(v) < 10:
        return

    width, height = 960, 560
    left, right, top = 64, 24, 46
    price_h, gap, vol_h = 350, 24, 90
    vol_top = top + price_h + gap
    plot_w = width - left - right
    highs = pd.to_numeric(v["high"], errors="coerce")
    lows = pd.to_numeric(v["low"], errors="coerce")
    support = num(item.get("support_price"))
    ymin = min(num(lows.min()), support if support > 0 else num(lows.min()))
    ymax = max(num(highs.max()), support if support > 0 else num(highs.max()))
    pad = max((ymax - ymin) * 0.08, ymax * 0.01)
    ymin, ymax = ymin - pad, ymax + pad
    step = plot_w / max(len(v), 1)
    candle_w = max(2.0, min(7.0, step * 0.58))

    def x(i: int) -> float:
        return left + (i + 0.5) * step

    def y(price: float) -> float:
        return top + (ymax - price) / max(ymax - ymin, 1e-9) * price_h

    max_vol = max(num(v["volume"].max()), 1.0)
    def vy(vol: float) -> float:
        return vol_top + vol_h - vol / max_vol * vol_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="18" font-weight="700" fill="#1c2738">{svg_escape(item.get("name"))} {svg_escape(item.get("code"))} · {svg_escape(item.get("signal"))}</text>',
        f'<text x="{width-right}" y="26" text-anchor="end" font-size="13" fill="#6f7b8f">确认 {svg_escape(item.get("confirmation_date"))}</text>',
    ]
    # 网格与价格标签
    for k in range(5):
        py = top + price_h * k / 4
        price = ymax - (ymax - ymin) * k / 4
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#edf0f5"/>')
        parts.append(f'<text x="{left-8}" y="{py+4:.1f}" text-anchor="end" font-size="11" fill="#8a94a6">{price:.2f}</text>')

    # 支撑线
    if support > 0 and ymin <= support <= ymax:
        sy = y(support)
        parts.append(f'<line x1="{left}" y1="{sy:.1f}" x2="{width-right}" y2="{sy:.1f}" stroke="#2f6fed" stroke-width="2" stroke-dasharray="8 5"/>')
        parts.append(f'<text x="{width-right-4}" y="{sy-6:.1f}" text-anchor="end" font-size="12" fill="#2f6fed">{svg_escape(item.get("support_name"))} {support:.2f}</text>')

    # K线
    for i, row in v.iterrows():
        opn, close, high, low, vol = [num(row[c]) for c in ("open", "close", "high", "low", "volume")]
        if not all(math.isfinite(z) for z in (opn, close, high, low, vol)):
            continue
        color = "#d84a3a" if close >= opn else "#168a50"
        xi = x(i)
        parts.append(f'<line x1="{xi:.1f}" y1="{y(high):.1f}" x2="{xi:.1f}" y2="{y(low):.1f}" stroke="{color}" stroke-width="1.2"/>')
        y1, y2 = y(max(opn, close)), y(min(opn, close))
        h = max(1.4, y2-y1)
        fill = "#ffffff" if close >= opn else color
        parts.append(f'<rect x="{xi-candle_w/2:.1f}" y="{y1:.1f}" width="{candle_w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{color}" stroke-width="1.2"/>')
        vtop = vy(vol)
        parts.append(f'<rect x="{xi-candle_w/2:.1f}" y="{vtop:.1f}" width="{candle_w:.1f}" height="{vol_top+vol_h-vtop:.1f}" fill="{color}" opacity="0.75"/>')

    # MA20
    pts = []
    for i, val in enumerate(v["ma20"]):
        m = num(val)
        if math.isfinite(m) and ymin <= m <= ymax:
            pts.append(f"{x(i):.1f},{y(m):.1f}")
    if pts:
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#7b3ff2" stroke-width="2.2"/>')
        parts.append(f'<text x="{left+8}" y="{top+18}" font-size="12" fill="#7b3ff2">MA20</text>')

    # 确认K线标记
    confirm_date = pd.to_datetime(item.get("confirmation_date"), errors="coerce")
    if pd.notna(confirm_date):
        hits = v.index[pd.to_datetime(v["date"]).dt.normalize() == confirm_date.normalize()]
        if len(hits):
            ci = int(hits[-1]); cx = x(ci)
            parts.append(f'<line x1="{cx:.1f}" y1="{top}" x2="{cx:.1f}" y2="{vol_top+vol_h}" stroke="#f0a000" stroke-width="1.6" stroke-dasharray="5 4"/>')
            parts.append(f'<text x="{cx-5:.1f}" y="{top+16}" text-anchor="end" font-size="12" fill="#b66d00">确认K</text>')

    # 日期轴
    for frac in (0.0, 0.5, 1.0):
        i = min(len(v)-1, max(0, int(round((len(v)-1)*frac))))
        dt = pd.to_datetime(v.iloc[i]["date"], errors="coerce")
        label = dt.strftime("%Y-%m-%d") if pd.notna(dt) else ""
        parts.append(f'<text x="{x(i):.1f}" y="{height-16}" text-anchor="middle" font-size="11" fill="#8a94a6">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{vol_top+vol_h}" x2="{width-right}" y2="{vol_top+vol_h}" stroke="#dfe4ec"/>')
    parts.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    (DATA_DIR / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "state", "buy_point", "signal", "score", "code", "name", "industry", "price", "change_pct",
        "prior_peak_date", "prior_peak_price", "prior_trough_date", "prior_trough_price",
        "breakout_date", "impulse_peak_date", "impulse_gain_pct", "breakout_volume_ratio", "above_ma20_ratio_pct",
        "pullback_days", "drawdown_pct", "pullback_volume_ratio", "support_name", "support_price", "support_date",
        "support_distance_pct", "support_validations", "confirmation_date", "confirmation_body_ratio",
        "confirmation_volume_ratio", "confirmation_above_ma20", "distance_ma20_pct", "trigger", "stop", "target_2r",
        "risk_pct", "risk_reward", "amount_yi", "turnover", "chart_url",
    ]
    with (DATA_DIR / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(payload.get("candidates", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intraday", "close"], default="close")
    ap.add_argument("--max-price", type=float, default=10000.0)
    ap.add_argument("--min-price", type=float, default=0.5)
    ap.add_argument("--min-amount-yi", type=float, default=0.0)
    ap.add_argument("--max-stocks", type=int, default=0, help="0=沪深主板全部")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--bars", type=int, default=360)
    ap.add_argument("--top", type=int, default=56)
    ap.add_argument("--near-tolerance", type=float, default=0.10, help="买点一最低收盘距离MA20最大偏差")
    ap.add_argument("--structure-tolerance", type=float, default=0.05, help="买点二/三/四支撑最大偏差")
    ap.add_argument("--max-contraction", type=float, default=0.85, help="回调中位量/上涨段中位量上限")
    args = ap.parse_args()

    started = time.time()
    now = datetime.now(SH_TZ)
    warnings: list[str] = []
    print(f"[{now:%F %T}] 开始四买点结构扫描 V5.0：{args.mode}")
    spot, source, src_warnings = get_spot()
    warnings.extend(src_warnings)
    universe = choose_universe(spot, args)
    print(f"实时快照 {len(spot)} 只，沪深主板基础过滤后 {len(universe)} 只；数据源 {source}")
    if len(universe) < 300:
        raise RuntimeError(f"基础过滤后仅 {len(universe)} 只，疑似数据异常，停止发布")

    industry_map, industry_scores, industry_ranking = build_industry_map(set(universe["code"]), warnings)
    histories: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    fallback_count = 0
    rows_by_code = {str(r["code"]): r for _, r in universe.iterrows()}

    with cf.ThreadPoolExecutor(max_workers=max(4, args.workers)) as ex:
        futs = {ex.submit(fetch_history, c, args.bars): c for c in rows_by_code}
        total = len(futs)
        done = 0
        for fut in cf.as_completed(futs):
            code, hist, note = fut.result()
            done += 1
            if hist is not None and len(hist) >= 120:
                histories[code] = merge_spot_bar(hist, rows_by_code[code], now)
                if note:
                    fallback_count += 1
            else:
                failures.append(code)
            if done % 250 == 0 or done == total:
                print(f"历史行情进度 {done}/{total}，成功 {len(histories)}，失败 {len(failures)}")

    ok_ratio = len(histories) / max(1, len(rows_by_code))
    if len(histories) < 300 or ok_ratio < 0.70:
        raise RuntimeError(f"历史行情成功率仅 {ok_ratio:.1%}，停止发布，保留上一版结果")
    if failures:
        warnings.append(f"历史行情失败 {len(failures)} 只；成功率 {ok_ratio:.1%}")
    if fallback_count:
        warnings.append(f"{fallback_count} 只使用AKShare个股历史回退源")

    summary_rows = []
    for code, h in histories.items():
        d = add_indicators(h)
        if len(d) >= 65:
            last = d.iloc[-1]
            summary_rows.append((code, num(last["close"]), num(last["ma20"]), num(last["r20"])))
    breadth = sum(1 for _, c, m, _ in summary_rows if c > m) / max(1, len(summary_rows))
    market_r20 = float(np.nanmedian([r for *_, r in summary_rows]))
    market_state = "强势" if breadth >= 0.62 else "正常" if breadth >= 0.45 else "偏弱" if breadth >= 0.30 else "弱势"

    candidates: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    for code, h in histories.items():
        r = rows_by_code[code]
        industry = industry_map.get(code, "未分类")
        industry_score = industry_scores.get(industry, 50.0)
        try:
            item, stage = analyze_stock(
                code, str(r["name"]), industry, industry_score, h,
                market_r20, breadth, r, args.near_tolerance,
                args.structure_tolerance, args.max_contraction,
            )
            stage_counts[stage] += 1
            if item:
                candidates.append(item)
        except Exception as exc:  # noqa: BLE001
            stage_counts["计算异常"] += 1
            if len(warnings) < 12:
                warnings.append(f"{code} 计算异常：{type(exc).__name__}")

    state_rank = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
    candidates.sort(key=lambda x: (
        state_rank.get(x["state"], 9),
        -x["score"],
        x.get("support_distance_pct", 99),
        x.get("pullback_volume_ratio", 99),
        -num(x.get("amount_yi"), 0),
    ))
    caps = {"B1": 20, "B2": 16, "B3": 12, "B4": 8}
    selected: list[dict[str, Any]] = []
    counts = {"B1": 0, "B2": 0, "B3": 0, "B4": 0}
    for item in candidates:
        s = item["state"]
        if counts.get(s, 0) < caps.get(s, 0) and len(selected) < args.top:
            selected.append(item)
            counts[s] += 1

    # 每次扫描只保留本次候选图表，避免旧图混淆。
    chart_dir = DATA_DIR / "charts"
    chart_dir.mkdir(exist_ok=True)
    for old in chart_dir.glob("*.svg"):
        try:
            old.unlink()
        except OSError:
            pass
    for item in selected:
        code = item["code"]
        chart_name = f"{code}.svg"
        try:
            render_candidate_chart(histories[code], item, chart_dir / chart_name)
            item["chart_url"] = f"data/charts/{chart_name}"
        except Exception as exc:  # noqa: BLE001
            if len(warnings) < 18:
                warnings.append(f"{code} K线图生成失败：{type(exc).__name__}")

    generated = datetime.now(SH_TZ)
    payload = {
        "schema": 10,
        "strategy_version": "V5.0",
        "meta": {
            "status": "success",
            "mode": args.mode,
            "mode_name": "盘中预警" if args.mode == "intraday" else "收盘确认",
            "market_scope": "A股沪深主板",
            "generated_at": generated.isoformat(timespec="seconds"),
            "market_date": generated.strftime("%Y-%m-%d"),
            "data_source": source + " + 腾讯日K",
            "snapshot_count": int(len(spot)),
            "universe_count": int(len(universe)),
            "history_success": int(len(histories)),
            "history_failed": int(len(failures)),
            "elapsed_seconds": round(time.time() - started, 1),
            "near_tolerance_pct": round(args.near_tolerance * 100, 1),
            "structure_tolerance_pct": round(args.structure_tolerance * 100, 1),
            "max_contraction": round(args.max_contraction, 2),
            "warnings": warnings[:20],
            "disclaimer": "仅供量化研究与复盘，不构成投资建议。盘中信号尚未收盘确认。",
        },
        "market": {"breadth": round(breadth * 100, 1), "state": market_state, "median_r20_pct": pct(market_r20)},
        "summary": {
            "total": len(selected),
            "B1": counts["B1"], "B2": counts["B2"], "B3": counts["B3"], "B4": counts["B4"],
        },
        "diagnostics": {
            "stages": dict(stage_counts.most_common()),
            "explanation": "V5.0只认最新K线：阳线、实体大于上一根、量能大于上一根。主策略只用MA20；买点二/三/四依次使用前高、前低、历史更低波谷支撑。",
        },
        "industry_ranking": industry_ranking,
        "candidates": selected,
    }
    write_outputs(payload)
    print(f"完成：买点一 {counts['B1']} / 买点二 {counts['B2']} / 买点三 {counts['B3']} / 买点四 {counts['B4']}；总候选 {len(selected)}，耗时 {payload['meta']['elapsed_seconds']} 秒")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"SCAN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
