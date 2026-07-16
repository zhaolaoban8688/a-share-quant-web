#!/usr/bin/env python3
"""A股沪深主板双买点结构扫描器 V4.0.

数据：AKShare 实时快照 + 腾讯日K（失败时回退 AKShare 个股日线）。
输出：data/latest.json 与 data/candidates.csv。

策略：买点一=MA20/MA30附近放量上涨后的缩量回踩确认；
买点二=跌破均线后在明显前波峰/前波谷附近止跌确认。

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
    price, opn, high, low = [num(row.get(c)) for c in ("price", "open", "high", "low")]
    vol, amount = num(row.get("volume"), 0), num(row.get("amount"), 0)
    if not all(math.isfinite(x) and x > 0 for x in (price, opn, high, low)):
        return h.tail(300).reset_index(drop=True)

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
        new = {
            "date": today, "open": opn, "close": price, "high": high, "low": low,
            "volume": max(vol, 0), "amount": max(amount, 0),
        }
        h = pd.concat([h, pd.DataFrame([new])], ignore_index=True)
    return h.tail(300).reset_index(drop=True)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    for n in (5, 10, 20, 30, 60):
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
    prior_high: float
    start_low: float
    peak_high: float
    gain: float
    breakout_volume_ratio: float
    segment_volume_ratio: float
    above_ma_ratio: float


@dataclass
class StructuralLevel:
    idx: int
    kind: str
    price: float
    rise_before: float
    pullback_after: float


@dataclass
class BuySetup:
    state: str
    signal: str
    impulse: Impulse
    pullback_days: int
    drawdown: float
    contraction: float
    support_name: str
    support_price: float
    support_distance: float
    support_date: str
    touch_idx: int
    confirmation_body_ratio: float
    confirmation_body_pct: float
    confirmation_return: float
    broke_both_ma: bool
    structural_amplitude: float


def local_peak(d: pd.DataFrame, idx: int, radius: int = 3) -> bool:
    left = max(0, idx - radius)
    right = min(len(d), idx + radius + 1)
    v = num(d.iloc[idx]["high"])
    return math.isfinite(v) and v >= num(d.iloc[left:right]["high"].max()) * 0.998


def local_low(d: pd.DataFrame, idx: int, radius: int = 3) -> bool:
    left = max(0, idx - radius)
    right = min(len(d), idx + radius + 1)
    v = num(d.iloc[idx]["low"])
    return math.isfinite(v) and v <= num(d.iloc[left:right]["low"].min()) * 1.002


def body_size(row: pd.Series) -> float:
    return abs(num(row["close"]) - num(row["open"]))


def is_large_bullish_confirmation(d: pd.DataFrame, idx: int) -> tuple[bool, float, float, float]:
    """当前K线为阳线，且实体严格大于前一根K线实体。

    为避免十字星之间的微小比较产生噪声，当前实体还需达到开盘价的0.30%。
    这不是额外趋势条件，只是最低可识别实体要求。
    """
    if idx < 1:
        return False, 0.0, 0.0, 0.0
    cur, prev = d.iloc[idx], d.iloc[idx - 1]
    opn, close = num(cur["open"]), num(cur["close"])
    prev_close = num(prev["close"])
    if not all(math.isfinite(x) and x > 0 for x in (opn, close, prev_close)):
        return False, 0.0, 0.0, 0.0
    cur_body = close - opn
    prev_body = body_size(prev)
    body_pct = cur_body / opn
    body_ratio = cur_body / max(prev_body, opn * 0.001)
    ret = close / prev_close - 1
    ok = cur_body > 0 and cur_body > prev_body and body_pct >= 0.003
    return ok, body_ratio, body_pct, ret


def find_impulses(d: pd.DataFrame, confirm_idx: int) -> list[Impulse]:
    """寻找近期位于MA20/MA30之上的“放量破前高上涨段”。"""
    out: list[Impulse] = []
    left_bound = max(65, confirm_idx - 75)
    right_bound = confirm_idx - 2
    if right_bound <= left_bound:
        return out

    for breakout_idx in range(left_bound, right_bound + 1):
        row = d.iloc[breakout_idx]
        close, opn = num(row["close"]), num(row["open"])
        ma20, ma30 = num(row["ma20"]), num(row["ma30"])
        vol = num(row["volume"])
        prior = d.iloc[max(0, breakout_idx - 30):breakout_idx]
        if len(prior) < 20:
            continue
        prior_high = num(prior["high"].max())
        prior_vol = num(prior["volume"].median())
        if not all(math.isfinite(x) and x > 0 for x in (close, opn, ma20, ma30, vol, prior_high, prior_vol)):
            continue
        vol_ratio = vol / prior_vol
        # 破前高、阳线、位于两条均线之上、成交量至少放大20%。
        if not (
            close > opn
            and close > ma20
            and close > ma30
            and close >= prior_high * 1.001
            and vol_ratio >= 1.20
        ):
            continue

        peak_right = min(confirm_idx - 1, breakout_idx + 32)
        if peak_right <= breakout_idx:
            continue
        peak_idx = int(d.iloc[breakout_idx:peak_right + 1]["high"].idxmax())
        peak_high = num(d.iloc[peak_idx]["high"])
        start_left = max(40, breakout_idx - 22)
        start_idx = int(d.iloc[start_left:breakout_idx + 1]["low"].idxmin())
        start_low = num(d.iloc[start_idx]["low"])
        gain = peak_high / max(start_low, 1e-9) - 1
        if gain < 0.10 or peak_high < prior_high * 1.01:
            continue

        segment = d.iloc[start_idx + 1:peak_idx + 1]
        if len(segment) < 3:
            continue
        above_ma = (
            (segment["close"] > segment["ma20"])
            & (segment["close"] > segment["ma30"])
        )
        above_ratio = float(above_ma.mean())
        segment_volume_ratio = num(segment["volume"].median()) / prior_vol
        if above_ratio < 0.55:
            continue

        out.append(
            Impulse(
                start_idx=start_idx,
                breakout_idx=breakout_idx,
                peak_idx=peak_idx,
                prior_high=prior_high,
                start_low=start_low,
                peak_high=peak_high,
                gain=gain,
                breakout_volume_ratio=vol_ratio,
                segment_volume_ratio=segment_volume_ratio,
                above_ma_ratio=above_ratio,
            )
        )

    # 同一波上涨可能出现多个破前高日，只保留结构质量更高且较近期的少量候选。
    out.sort(
        key=lambda x: (
            x.peak_idx,
            x.gain,
            x.breakout_volume_ratio,
            x.above_ma_ratio,
        ),
        reverse=True,
    )
    unique: list[Impulse] = []
    used_peaks: list[int] = []
    for item in out:
        if any(abs(item.peak_idx - p) <= 3 for p in used_peaks):
            continue
        unique.append(item)
        used_peaks.append(item.peak_idx)
        if len(unique) >= 8:
            break
    return unique


def find_structural_levels(d: pd.DataFrame, end_idx: int, lookback: int = 150) -> list[StructuralLevel]:
    """识别由明确上涨和回调形成的前波峰、前波谷。

    前波峰：此前至少上涨8%，此后至少回调6%。
    前波谷：此前至少回调6%，此后至少上涨8%。
    """
    levels: list[StructuralLevel] = []
    start = max(30, end_idx - lookback)
    stop = min(end_idx - 4, len(d) - 5)
    if stop <= start:
        return levels

    for i in range(start, stop + 1):
        left = d.iloc[max(0, i - 25):i]
        right = d.iloc[i + 1:min(len(d), i + 26)]
        if len(left) < 8 or len(right) < 5:
            continue

        if local_peak(d, i, 3):
            price = num(d.iloc[i]["high"])
            left_low = num(left["low"].min())
            right_low = num(right["low"].min())
            rise = price / max(left_low, 1e-9) - 1
            pullback = 1 - right_low / max(price, 1e-9)
            if rise >= 0.08 and pullback >= 0.06:
                levels.append(StructuralLevel(i, "前波峰", price, rise, pullback))

        if local_low(d, i, 3):
            price = num(d.iloc[i]["low"])
            left_high = num(left["high"].max())
            right_high = num(right["high"].max())
            decline = 1 - price / max(left_high, 1e-9)
            rebound = right_high / max(price, 1e-9) - 1
            if decline >= 0.06 and rebound >= 0.08:
                levels.append(StructuralLevel(i, "前波谷", price, rebound, decline))

    levels.sort(key=lambda x: x.idx, reverse=True)
    return levels


def nearest_ma_touch(
    d: pd.DataFrame,
    start_idx: int,
    confirm_idx: int,
    tolerance: float,
) -> tuple[str, float, float, int] | None:
    """最近5根K线内是否触及MA20或MA30附近。"""
    begin = max(start_idx, confirm_idx - 4)
    best: tuple[str, float, float, int] | None = None
    for i in range(begin, confirm_idx + 1):
        row = d.iloc[i]
        for name, col in (("MA20", "ma20"), ("MA30", "ma30")):
            level = num(row[col])
            if not math.isfinite(level) or level <= 0:
                continue
            # 低点、收盘、开盘任一靠近均线即可视为回踩到位。
            dist = min(
                abs(num(row["low"]) / level - 1),
                abs(num(row["close"]) / level - 1),
                abs(num(row["open"]) / level - 1),
            )
            if dist <= tolerance and (best is None or dist < best[2]):
                best = (name, level, dist, i)
    return best


def nearest_structural_touch(
    d: pd.DataFrame,
    levels: list[StructuralLevel],
    start_idx: int,
    confirm_idx: int,
    tolerance: float,
) -> tuple[StructuralLevel, float, int] | None:
    begin = max(start_idx, confirm_idx - 4)
    best: tuple[StructuralLevel, float, int] | None = None
    for i in range(begin, confirm_idx + 1):
        row = d.iloc[i]
        for level in levels:
            dist = min(
                abs(num(row["low"]) / level.price - 1),
                abs(num(row["close"]) / level.price - 1),
            )
            if dist <= tolerance:
                # 距离优先，距离相同时优先较近的结构位。
                if best is None or (dist, -level.idx) < (best[1], -best[0].idx):
                    best = (level, dist, i)
    return best


def find_buy_setup(d: pd.DataFrame, tolerance: float = 0.15) -> BuySetup | None:
    """严格按用户新定义识别买点一和买点二，确认K线必须是最新一根。"""
    n = len(d)
    if n < 110:
        return None
    confirm_idx = n - 1
    ok, body_ratio, body_pct, confirm_ret = is_large_bullish_confirmation(d, confirm_idx)
    if not ok:
        return None

    best: BuySetup | None = None
    best_quality = -1e9
    for impulse in find_impulses(d, confirm_idx):
        pullback_days = confirm_idx - impulse.peak_idx
        if not 2 <= pullback_days <= 36:
            continue
        pull = d.iloc[impulse.peak_idx + 1:confirm_idx + 1]
        pull_core = d.iloc[impulse.peak_idx + 1:confirm_idx]
        impulse_bars = d.iloc[impulse.start_idx + 1:impulse.peak_idx + 1]
        if len(pull_core) < 1 or len(impulse_bars) < 3:
            continue

        pull_low = num(pull["low"].min())
        drawdown = 1 - pull_low / max(impulse.peak_high, 1e-9)
        if not 0.02 <= drawdown <= 0.48:
            continue

        impulse_vol = num(impulse_bars["volume"].median())
        pull_vol = num(pull_core["volume"].median())
        contraction = pull_vol / max(impulse_vol, 1e-9)
        # “缩量回调”采用中位量，避免单日异常量柱扭曲判断。
        if contraction > 0.95:
            continue

        broke_both = bool(
            (
                (pull["close"] < pull["ma20"] * 0.98)
                & (pull["close"] < pull["ma30"] * 0.98)
            ).any()
        )

        setup: BuySetup | None = None

        # 买点二优先：有效跌破两条均线后，寻找明显前波峰/前波谷支撑。
        if broke_both:
            levels = find_structural_levels(d, impulse.breakout_idx - 1)
            touch = nearest_structural_touch(
                d, levels, impulse.peak_idx + 1, confirm_idx, tolerance
            )
            if touch is not None:
                level, dist, touch_idx = touch
                # “止跌企稳”：支撑触碰发生在最近5根内，确认日低点未明显失控。
                recent = d.iloc[max(impulse.peak_idx + 1, confirm_idx - 4):confirm_idx + 1]
                recent_low = num(recent["low"].min())
                current_low = num(d.iloc[confirm_idx]["low"])
                stable = current_low >= recent_low * 0.97
                if stable:
                    setup = BuySetup(
                        state="B",
                        signal="买点二·结构支撑确认",
                        impulse=impulse,
                        pullback_days=pullback_days,
                        drawdown=drawdown,
                        contraction=contraction,
                        support_name=level.kind,
                        support_price=level.price,
                        support_distance=dist,
                        support_date=d.iloc[level.idx]["date"].strftime("%Y-%m-%d"),
                        touch_idx=touch_idx,
                        confirmation_body_ratio=body_ratio,
                        confirmation_body_pct=body_pct,
                        confirmation_return=confirm_ret,
                        broke_both_ma=True,
                        structural_amplitude=min(level.rise_before, level.pullback_after),
                    )

        # 买点一：没有有效跌破两条均线，最近5根内回踩MA20/MA30附近。
        if setup is None and not broke_both:
            touch = nearest_ma_touch(
                d, impulse.peak_idx + 1, confirm_idx, tolerance
            )
            if touch is not None:
                name, level, dist, touch_idx = touch
                current_level = num(d.iloc[confirm_idx]["ma20" if name == "MA20" else "ma30"])
                current_close = num(d.iloc[confirm_idx]["close"])
                # 确认K线不能已经远离均线超过“附近”上限。
                if current_close <= current_level * (1 + tolerance):
                    setup = BuySetup(
                        state="A",
                        signal="买点一·均线回踩确认",
                        impulse=impulse,
                        pullback_days=pullback_days,
                        drawdown=drawdown,
                        contraction=contraction,
                        support_name=name,
                        support_price=level,
                        support_distance=dist,
                        support_date=d.iloc[touch_idx]["date"].strftime("%Y-%m-%d"),
                        touch_idx=touch_idx,
                        confirmation_body_ratio=body_ratio,
                        confirmation_body_pct=body_pct,
                        confirmation_return=confirm_ret,
                        broke_both_ma=False,
                        structural_amplitude=0.0,
                    )

        if setup is None:
            continue

        proximity = 1 - min(setup.support_distance / max(tolerance, 1e-9), 1)
        quality = (
            min(impulse.gain, 0.60) * 40
            + min(impulse.breakout_volume_ratio, 3.0) * 7
            + impulse.above_ma_ratio * 12
            + max(0, 1 - contraction) * 22
            + proximity * 18
            + min(body_ratio, 4.0) * 5
            + (6 if setup.state == "B" else 4)
            + min(setup.structural_amplitude, 0.25) * 20
        )
        if quality > best_quality:
            best_quality = quality
            best = setup

    return best


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
) -> dict[str, Any] | None:
    del market_r20, breadth  # 新策略不以市场或行业强弱作为入选条件。
    d = add_indicators(d0)
    if len(d) < 110:
        return None
    setup = find_buy_setup(d, tolerance=tolerance)
    if setup is None:
        return None

    last, prev = d.iloc[-1], d.iloc[-2]
    price = num(last["close"])
    ma20, ma30 = num(last["ma20"]), num(last["ma30"])
    dist20 = price / ma20 - 1 if ma20 > 0 else math.nan
    dist30 = price / ma30 - 1 if ma30 > 0 else math.nan
    impulse = setup.impulse

    proximity_score = 1 - min(setup.support_distance / max(tolerance, 1e-9), 1)
    score = (
        clamp((impulse.gain - 0.10) / 0.40, 0, 1) * 22
        + clamp((impulse.breakout_volume_ratio - 1.20) / 1.80, 0, 1) * 16
        + clamp((impulse.above_ma_ratio - 0.55) / 0.45, 0, 1) * 10
        + clamp((0.95 - setup.contraction) / 0.55, 0, 1) * 20
        + proximity_score * 17
        + clamp((setup.confirmation_body_ratio - 1.0) / 2.5, 0, 1) * 10
        + (5 if setup.state == "B" else 3)
    )
    score = clamp(score, 0, 100)

    recent_low = num(d.iloc[max(impulse.peak_idx + 1, len(d) - 5):]["low"].min())
    trigger = num(last["high"]) * 1.002
    stop = min(recent_low, setup.support_price) * 0.975
    if stop <= 0 or stop >= trigger:
        stop = trigger * 0.94
    risk_pct = (trigger - stop) / trigger
    target = trigger + 2 * (trigger - stop)

    change_pct = num(spot_row.get("change_pct"))
    pull_speed = setup.drawdown / max(setup.pullback_days, 1)
    return {
        "code": code,
        "name": name,
        "industry": industry or "未分类",
        "industry_score": round(industry_score, 1),
        "state": setup.state,
        "signal": setup.signal,
        "score": round(score, 1),
        "price": round(price, 3),
        "change_pct": safe_round(change_pct, 2),
        "ma20": round(ma20, 3),
        "ma30": round(ma30, 3),
        "distance_ma20_pct": pct(dist20),
        "distance_ma30_pct": pct(dist30),
        "impulse_start_date": d.iloc[impulse.start_idx]["date"].strftime("%Y-%m-%d"),
        "breakout_date": d.iloc[impulse.breakout_idx]["date"].strftime("%Y-%m-%d"),
        "impulse_peak_date": d.iloc[impulse.peak_idx]["date"].strftime("%Y-%m-%d"),
        "impulse_gain_pct": pct(impulse.gain),
        "breakout_volume_ratio": round(impulse.breakout_volume_ratio, 2),
        "impulse_volume_ratio": round(impulse.segment_volume_ratio, 2),
        "above_ma_ratio_pct": pct(impulse.above_ma_ratio),
        "pullback_days": setup.pullback_days,
        "drawdown_pct": pct(setup.drawdown),
        "pullback_speed_pct_day": pct(pull_speed),
        "pullback_volume_ratio": round(setup.contraction, 2),
        "support_name": setup.support_name,
        "support_price": round(setup.support_price, 3),
        "support_date": setup.support_date,
        "support_distance_pct": pct(setup.support_distance),
        "confirmation_body_ratio": round(setup.confirmation_body_ratio, 2),
        "confirmation_body_pct": pct(setup.confirmation_body_pct),
        "confirmation_return_pct": pct(setup.confirmation_return),
        "broke_both_ma": setup.broke_both_ma,
        "trigger": round(trigger, 3),
        "stop": round(stop, 3),
        "target_2r": round(target, 3),
        "risk_pct": pct(risk_pct),
        "risk_reward": 2.0,
        "amount_yi": safe_round(num(spot_row.get("amount")) / 1e8, 2),
        "turnover": safe_round(spot_row.get("turnover"), 2),
        "volume_ratio_spot": safe_round(spot_row.get("volume_ratio"), 2),
        "reason": [
            (
                f"{d.iloc[impulse.breakout_idx]['date']:%Y-%m-%d} 放量破前高，"
                f"突破量为前20日中位量的 {impulse.breakout_volume_ratio:.2f} 倍"
            ),
            (
                f"上涨段涨幅 {pct(impulse.gain):.1f}%，"
                f"{pct(impulse.above_ma_ratio):.1f}% 的交易日收在MA20和MA30之上"
            ),
            (
                f"回调 {setup.pullback_days} 日、回撤 {pct(setup.drawdown):.1f}%，"
                f"回调中位量缩至上涨段的 {setup.contraction:.2f}"
            ),
            (
                f"最近在 {setup.support_name} {setup.support_price:.2f} 附近止跌，"
                f"偏差 {pct(setup.support_distance):.1f}%"
            ),
            (
                f"最新阳线实体为前一根K线实体的 {setup.confirmation_body_ratio:.2f} 倍，"
                f"实体涨幅 {pct(setup.confirmation_body_pct):.2f}%"
            ),
            f"触发 {trigger:.2f} / 结构失效参考 {stop:.2f} / 2R参考 {target:.2f}",
        ],
    }


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


def write_outputs(payload: dict[str, Any]) -> None:
    (DATA_DIR / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "state", "signal", "score", "code", "name", "industry", "price", "change_pct",
        "impulse_start_date", "breakout_date", "impulse_peak_date", "impulse_gain_pct",
        "breakout_volume_ratio", "impulse_volume_ratio", "above_ma_ratio_pct",
        "pullback_days", "drawdown_pct", "pullback_speed_pct_day", "pullback_volume_ratio",
        "support_name", "support_price", "support_date", "support_distance_pct",
        "confirmation_body_ratio", "confirmation_body_pct", "confirmation_return_pct",
        "distance_ma20_pct", "distance_ma30_pct", "trigger", "stop", "target_2r",
        "risk_pct", "risk_reward", "amount_yi", "turnover",
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
    ap.add_argument("--min-amount-yi", type=float, default=0.0, help="当日最低成交额，亿元")
    ap.add_argument("--max-stocks", type=int, default=0, help="0=全部")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--bars", type=int, default=320)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--near-tolerance", type=float, default=0.15, help="支撑附近允许偏差，0.15=15%")
    args = ap.parse_args()

    started = time.time()
    now = datetime.now(SH_TZ)
    warnings: list[str] = []
    print(f"[{now:%F %T}] 开始双买点结构扫描：{args.mode}")
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
            if hist is not None and len(hist) >= 110:
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
    for code, h in histories.items():
        r = rows_by_code[code]
        industry = industry_map.get(code, "未分类")
        industry_score = industry_scores.get(industry, 50.0)
        try:
            item = analyze_stock(
                code, str(r["name"]), industry, industry_score, h,
                market_r20, breadth, r, args.near_tolerance,
            )
            if item:
                candidates.append(item)
        except Exception as exc:  # noqa: BLE001
            if len(warnings) < 12:
                warnings.append(f"{code} 计算异常：{type(exc).__name__}")

    state_rank = {"A": 0, "B": 1}
    candidates.sort(
        key=lambda x: (
            state_rank.get(x["state"], 9),
            -x["score"],
            x.get("support_distance_pct", 99),
            x.get("pullback_volume_ratio", 99),
            -num(x.get("amount_yi"), 0),
        )
    )
    caps = {"A": 20, "B": 20}
    selected: list[dict[str, Any]] = []
    counts = {"A": 0, "B": 0}
    for item in candidates:
        state = item["state"]
        if counts.get(state, 0) < caps.get(state, 0) and len(selected) < args.top:
            selected.append(item)
            counts[state] += 1

    generated = datetime.now(SH_TZ)
    payload = {
        "schema": 6,
        "strategy_version": "V4.0",
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
            "warnings": warnings[:20],
            "disclaimer": "仅供量化研究与复盘，不构成投资建议。盘中信号尚未收盘确认。",
        },
        "market": {
            "breadth": round(breadth * 100, 1),
            "state": market_state,
            "median_r20_pct": pct(market_r20),
        },
        "summary": {
            "total": len(selected),
            "A": counts["A"],
            "B": counts["B"],
        },
        "industry_ranking": industry_ranking,
        "candidates": selected,
    }
    write_outputs(payload)
    print(
        f"完成：买点一 {counts['A']} / 买点二 {counts['B']}，"
        f"总候选 {len(selected)}，耗时 {payload['meta']['elapsed_seconds']} 秒"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"SCAN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
