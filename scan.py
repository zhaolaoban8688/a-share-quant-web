#!/usr/bin/env python3
"""A股沪深主板缩量回踩精选扫描器 V3.1平衡版.

数据：AKShare 实时快照 + 腾讯日K（失败时回退 AKShare 个股日线）。
输出：data/latest.json 与 data/candidates.csv。

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
    """Normalize Eastmoney/Sina AKShare spot columns."""
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
    for col in ["price", "change_pct", "volume", "amount", "high", "low", "open", "prev_close", "volume_ratio", "turnover", "r60_spot"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 新浪接口成交量单位为股；东财为手。按数量级与成交额粗略统一成手。
    median_vol = df["volume"].replace(0, np.nan).median()
    if math.isfinite(num(median_vol)) and median_vol > 1e7:
        df["volume"] = df["volume"] / 100.0
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
    """Merge intraday/final snapshot as current date bar."""
    h = hist.copy()
    today = pd.Timestamp(scan_time.date())
    price, opn, high, low = [num(row.get(c)) for c in ("price", "open", "high", "low")]
    vol, amount = num(row.get("volume"), 0), num(row.get("amount"), 0)
    if not all(math.isfinite(x) and x > 0 for x in (price, opn, high, low)):
        return h
    new = {"date": today, "open": opn, "close": price, "high": high, "low": low, "volume": max(vol, 0), "amount": max(amount, 0)}
    if not h.empty and pd.Timestamp(h.iloc[-1]["date"]).normalize() == today:
        for k, v in new.items():
            h.at[h.index[-1], k] = v
    else:
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
class PullbackSetup:
    state: str
    impulse_start_idx: int
    impulse_peak_idx: int
    impulse_gain: float
    impulse_days: int
    impulse_volume_ratio: float
    impulse_max_volume_ratio: float
    impulse_efficiency: float
    strong_up_days: int
    pullback_days: int
    drawdown: float
    pullback_speed: float
    contraction: float
    down_volume_ratio: float
    support_name: str
    support: float
    support_distance: float
    confirmation_return: float
    confirmation_strength: float


def local_peak(d: pd.DataFrame, idx: int, radius: int = 2) -> bool:
    left = max(0, idx - radius)
    right = min(len(d), idx + radius + 1)
    peak = num(d.iloc[idx]["high"])
    return math.isfinite(peak) and peak >= num(d.iloc[left:right]["high"].max()) * 0.995


def local_low(d: pd.DataFrame, idx: int, radius: int = 2) -> bool:
    left = max(0, idx - radius)
    right = min(len(d), idx + radius + 1)
    low = num(d.iloc[idx]["low"])
    return math.isfinite(low) and low <= num(d.iloc[left:right]["low"].min()) * 1.005


def find_pullback_setup(d: pd.DataFrame) -> PullbackSetup | None:
    """识别“放量主升—温和缩量回踩—MA20/MA30企稳”结构。

    V3.1平衡版：
    A：最新一日已经收阳确认；
    B：结构合格但尚待收阳确认。
    红线条件仍一票否决，其余通过评分优选，避免全市场零候选。
    """
    n = len(d)
    if n < 110:
        return None

    latest = d.iloc[-1]
    prev = d.iloc[-2]
    price = num(latest["close"])
    ma20, ma30, ma60 = num(latest["ma20"]), num(latest["ma30"]), num(latest["ma60"])
    if not all(math.isfinite(x) and x > 0 for x in (price, ma20, ma30, ma60)):
        return None

    ma20_slope5 = ma20 / max(num(d.iloc[-6]["ma20"]), 1e-9) - 1
    ma30_slope5 = ma30 / max(num(d.iloc[-6]["ma30"]), 1e-9) - 1
    ma60_slope10 = ma60 / max(num(d.iloc[-11]["ma60"]), 1e-9) - 1

    # 保留趋势底线，但允许正常回踩时MA20短暂走平。
    if not (
        price >= ma30 * 0.97
        and price >= ma60 * 0.98
        and ma20 >= ma30 * 0.97
        and ma30 >= ma60 * 0.94
        and ma20_slope5 >= -0.005
        and ma30_slope5 >= -0.008
        and ma60_slope10 >= -0.020
    ):
        return None

    best: PullbackSetup | None = None
    best_quality = -1e9

    # 高点距离现在2—18日，兼容稍长一些的温和整理。
    for peak_idx in range(max(70, n - 19), n - 2):
        if not local_peak(d, peak_idx, 2):
            continue

        pullback_days = n - 1 - peak_idx
        if not 2 <= pullback_days <= 18:
            continue

        peak_high = num(d.iloc[peak_idx]["high"])
        peak_close = num(d.iloc[peak_idx]["close"])
        if not all(math.isfinite(x) and x > 0 for x in (peak_high, peak_close)):
            continue

        # 主升段允许4—28日。
        start_left = max(55, peak_idx - 30)
        start_right = peak_idx - 3
        if start_right <= start_left:
            continue

        start_candidates = [
            i for i in range(start_left, start_right + 1)
            if local_low(d, i, 2)
        ]
        if not start_candidates:
            start_candidates = [int(d.iloc[start_left:start_right + 1]["low"].idxmin())]

        for start_idx in start_candidates:
            impulse_days = peak_idx - start_idx
            if not 4 <= impulse_days <= 28:
                continue

            start_low = num(d.iloc[start_idx]["low"])
            impulse_gain = peak_high / max(start_low, 1e-9) - 1
            if not 0.10 <= impulse_gain <= 0.60:
                continue

            impulse = d.iloc[start_idx + 1: peak_idx + 1].copy()
            prior = d.iloc[max(0, start_idx - 20):start_idx].copy()
            if len(impulse) < 4 or len(prior) < 10:
                continue

            prior_vol = num(prior["volume"].median())
            impulse_vol = num(impulse["volume"].mean())
            if not math.isfinite(prior_vol) or prior_vol <= 0 or not math.isfinite(impulse_vol):
                continue

            impulse_volume_ratio = impulse_vol / prior_vol
            day_volume_ratio = impulse["volume"] / impulse["vma20"].replace(0, np.nan)
            impulse_max_volume_ratio = num(day_volume_ratio.max())
            impulse_returns = impulse["close"].pct_change().fillna(
                impulse.iloc[0]["close"] / max(num(d.iloc[start_idx]["close"]), 1e-9) - 1
            )
            strong_up_days = int(
                ((impulse_returns >= 0.022) & (day_volume_ratio >= 1.05)).sum()
            )
            path = d.iloc[start_idx:peak_idx + 1]["close"].astype(float)
            impulse_efficiency = (
                (num(path.iloc[-1]) - num(path.iloc[0]))
                / max(num(path.diff().abs().sum()), 1e-9)
            )

            # 主升必须真实存在，但不再要求四项同时达到极高阈值。
            if not (
                impulse_volume_ratio >= 1.05
                and impulse_max_volume_ratio >= 1.25
                and (strong_up_days >= 1 or impulse_gain >= 0.18)
                and impulse_efficiency >= 0.22
            ):
                continue

            pull = d.iloc[peak_idx + 1:n].copy()
            if pull.empty:
                continue

            pullback_low = num(pull["low"].min())
            drawdown = (peak_high - pullback_low) / peak_high
            pullback_speed = drawdown / max(pullback_days, 1)
            current_from_peak = price / peak_high - 1

            # 红线：深跌、快速下杀、已经重新大幅创新高。
            if not (
                0.02 <= drawdown <= 0.22
                and pullback_speed <= 0.032
                and -0.22 <= current_from_peak <= 0.03
            ):
                continue

            pull_returns = pd.concat(
                [pd.Series([peak_close]), pull["close"].reset_index(drop=True)],
                ignore_index=True,
            ).pct_change().dropna()
            min_day = num(pull_returns.min(), 0)
            min_two_day = num(
                (1 + pull_returns).rolling(2).apply(np.prod, raw=True).min() - 1,
                0,
            )
            if min_day < -0.08 or min_two_day < -0.12:
                continue

            pull_vol = num(pull["volume"].mean())
            contraction = pull_vol / max(impulse_vol, 1e-9)
            down_mask = pull["close"] < pull["close"].shift(1)
            down_vol = num(pull.loc[down_mask, "volume"].mean(), pull_vol)
            down_volume_ratio = down_vol / max(impulse_vol, 1e-9)

            # 只排除明显放量长阴派发；普通回调量能交给评分处理。
            distribution = pull[
                (pull["close"].pct_change() <= -0.055)
                & (pull["volume"] >= pull["vma20"] * 1.50)
            ]
            if contraction > 0.98 or down_volume_ratio > 1.00 or len(distribution) > 0:
                continue

            # 使用近5日最低点判断是否完成过MA20/MA30回踩。
            recent5 = d.iloc[-5:]
            recent_low = num(recent5["low"].min())
            supports = {"MA20": ma20, "MA30": ma30}
            support_name, support = min(
                supports.items(),
                key=lambda kv: abs(recent_low / max(kv[1], 1e-9) - 1),
            )
            support_distance = recent_low / support - 1

            recent_support_touch = (
                recent_low <= support * 1.06
                and num(recent5["close"].min()) >= support * 0.94
            )
            ma30_hold_ratio = float(
                (
                    pull["close"]
                    >= pull["ma30"].replace(0, np.nan) * 0.94
                ).mean()
            )

            if not (
                -0.05 <= support_distance <= 0.06
                and price >= support * 0.985
                and recent_support_touch
                and ma30_hold_ratio >= 0.75
            ):
                continue

            # 阳线确认仍要求最新日有企稳动作，但允许小阳、下影线反包。
            rng = max(num(latest["high"]) - num(latest["low"]), 1e-9)
            close_position = (price - num(latest["low"])) / rng
            confirmation_return = price / max(num(prev["close"]), 1e-9) - 1
            body_return = (price - num(latest["open"])) / max(num(prev["close"]), 1e-9)
            lower_shadow = (
                min(num(latest["open"]), price) - num(latest["low"])
            ) / rng
            stable_range = (
                num(recent5["close"].max()) / max(num(recent5["close"].min()), 1e-9) - 1
            )

            confirmation = (
                price > num(latest["open"]) * 0.998
                and confirmation_return >= 0.0
                and confirmation_return <= 0.065
                and close_position >= 0.50
                and price >= support * 0.99
                and stable_range <= 0.09
                and (
                    body_return >= 0.002
                    or lower_shadow >= 0.22
                    or price >= num(prev["high"]) * 0.995
                )
            )

            latest_volume_ratio_to_impulse = num(latest["volume"]) / max(impulse_vol, 1e-9)
            if latest_volume_ratio_to_impulse > 1.35:
                continue

            state = "A" if confirmation else "B"
            if state == "B":
                if confirmation_return < -0.03 or stable_range > 0.10:
                    continue

            confirmation_strength = (
                clamp(close_position, 0, 1) * 0.38
                + clamp((confirmation_return + 0.012) / 0.06, 0, 1) * 0.27
                + clamp(lower_shadow / 0.45, 0, 1) * 0.18
                + clamp((0.09 - stable_range) / 0.09, 0, 1) * 0.17
            )

            quality = (
                impulse_gain * 65
                + min(impulse_volume_ratio, 2.5) * 6
                + impulse_efficiency * 10
                - pullback_speed * 210
                - abs(drawdown - 0.09) * 28
                - contraction * 6
                + confirmation_strength * 12
                + (5 if state == "A" else 0)
            )

            if quality > best_quality:
                best_quality = quality
                best = PullbackSetup(
                    state=state,
                    impulse_start_idx=start_idx,
                    impulse_peak_idx=peak_idx,
                    impulse_gain=impulse_gain,
                    impulse_days=impulse_days,
                    impulse_volume_ratio=impulse_volume_ratio,
                    impulse_max_volume_ratio=impulse_max_volume_ratio,
                    impulse_efficiency=impulse_efficiency,
                    strong_up_days=strong_up_days,
                    pullback_days=pullback_days,
                    drawdown=drawdown,
                    pullback_speed=pullback_speed,
                    contraction=contraction,
                    down_volume_ratio=down_volume_ratio,
                    support_name=support_name,
                    support=support,
                    support_distance=support_distance,
                    confirmation_return=confirmation_return,
                    confirmation_strength=confirmation_strength,
                )

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
) -> dict[str, Any] | None:
    d = add_indicators(d0)
    if len(d) < 110:
        return None

    setup = find_pullback_setup(d)
    if not setup:
        return None

    last, prev = d.iloc[-1], d.iloc[-2]
    price = num(last["close"])
    ma20, ma30, ma60 = num(last["ma20"]), num(last["ma30"]), num(last["ma60"])
    atr = num(last["atr14"])
    r10, r20, r60 = num(last["r10"]), num(last["r20"]), num(last["r60"])
    dist20 = price / ma20 - 1 if ma20 > 0 else math.nan
    dist30 = price / ma30 - 1 if ma30 > 0 else math.nan

    # 避免已经再次加速、离均线太远的股票。
    if (
        dist20 > 0.14
        or dist30 > 0.16
        or r10 > 0.32
        or not math.isfinite(atr)
        or atr <= 0
    ):
        return None

    ma20_slope5 = ma20 / max(num(d.iloc[-6]["ma20"]), 1e-9) - 1
    ma30_slope5 = ma30 / max(num(d.iloc[-6]["ma30"]), 1e-9) - 1

    # 1. 趋势质量：15分
    trend_score = 4.0
    trend_score += clamp(ma20_slope5 / 0.025, 0, 1) * 5
    trend_score += clamp((ma30_slope5 + 0.003) / 0.018, 0, 1) * 3
    trend_score += 2 if ma20 >= ma30 else 0
    trend_score += 1 if ma30 >= ma60 else 0
    trend_score = clamp(trend_score, 0, 15)

    # 2. 主升段质量：25分
    gain_sweet = 1 - min(abs(setup.impulse_gain - 0.24) / 0.20, 1)
    impulse_score = gain_sweet * 8
    impulse_score += clamp((setup.impulse_volume_ratio - 1.10) / 0.90, 0, 1) * 7
    impulse_score += clamp((setup.impulse_max_volume_ratio - 1.30) / 1.20, 0, 1) * 4
    impulse_score += clamp((setup.impulse_efficiency - 0.30) / 0.50, 0, 1) * 4
    impulse_score += clamp(setup.strong_up_days / 3, 0, 1) * 2
    impulse_score = clamp(impulse_score, 0, 25)

    # 3. 回调质量：30分。缩量和慢速回调权重最高。
    contraction_score = clamp((0.82 - setup.contraction) / 0.40, 0, 1) * 11
    speed_score = clamp((0.024 - setup.pullback_speed) / 0.017, 0, 1) * 9
    drawdown_score = (1 - min(abs(setup.drawdown - 0.085) / 0.09, 1)) * 5
    duration_score = clamp(setup.pullback_days / 8, 0, 1) * 3
    down_volume_score = clamp((0.80 - setup.down_volume_ratio) / 0.38, 0, 1) * 2
    pullback_score = clamp(
        contraction_score + speed_score + drawdown_score
        + duration_score + down_volume_score,
        0,
        30,
    )

    # 4. 均线支撑与阳线确认：20分
    support_closeness = 1 - min(abs(setup.support_distance) / 0.04, 1)
    support_score = support_closeness * 7
    support_score += 4 if price >= setup.support * 1.003 else 2
    support_score += setup.confirmation_strength * 6
    support_score += 3 if setup.state == "A" else 0
    support_score = clamp(support_score, 0, 20)

    # 5. 相对强度与行业只占5分，不能覆盖形态缺陷。
    relative = clamp(
        2.5 + (r20 - market_r20) * 10 + (industry_score - 50) * 0.025,
        0,
        5,
    )

    pullback_low = num(
        d.iloc[setup.impulse_peak_idx + 1:]["low"].min()
    )
    stop = min(setup.support * 0.965, pullback_low * 0.985)
    trigger = max(
        num(last["high"]),
        num(d.iloc[-3:-1]["high"].max()),
    ) * 1.002
    risk_pct = (trigger - stop) / trigger if trigger > 0 else 1
    if risk_pct < 0.022:
        stop = trigger * 0.975
        risk_pct = 0.025
    if risk_pct > 0.13:
        return None
    target = trigger + 2.0 * (trigger - stop)
    risk_score = clamp((0.13 - risk_pct) / 0.105 * 5, 0, 5)

    score = (
        trend_score
        + impulse_score
        + pullback_score
        + support_score
        + relative
        + risk_score
    )
    score = clamp(score, 0, 100)

    # 优中选优：确认型和观察型都设最低门槛。
    threshold = 75 if setup.state == "A" else 70
    if score < threshold:
        return None

    change_pct = num(spot_row.get("change_pct"))
    signal_name = (
        "缩量回踩·阳线确认"
        if setup.state == "A"
        else "缩量回踩·等待确认"
    )
    return {
        "code": code,
        "name": name,
        "industry": industry or "未分类",
        "state": setup.state,
        "signal": signal_name,
        "score": round(score, 1),
        "price": round(price, 3),
        "change_pct": safe_round(change_pct, 2),
        # 兼容旧网页/CSV字段，同时把平台字段改为当前均线支撑。
        "platform": round(setup.support, 3),
        "platform_touches": 0,
        "platform_days": setup.pullback_days,
        "breakout_date": d.iloc[setup.impulse_start_idx]["date"].strftime("%Y-%m-%d"),
        "days_after_breakout": setup.pullback_days,
        "breakout_volume_ratio": round(setup.impulse_volume_ratio, 2),
        "pullback_volume_ratio": round(setup.contraction, 2),
        "drawdown_pct": pct(setup.drawdown),
        "ma20": round(ma20, 3),
        "ma30": round(ma30, 3),
        "ma60": round(ma60, 3),
        "distance_ma20_pct": pct(dist20),
        "distance_ma30_pct": pct(dist30),
        "r10_pct": pct(r10),
        "r20_pct": pct(r20),
        "r60_pct": pct(r60),
        "industry_score": round(industry_score, 1),
        "relative_strength": round(relative, 1),
        "macd_above_zero": bool(num(last["dif"]) > 0 and num(last["dea"]) > 0),
        "macd_turning_up": bool(num(last["macd_hist"]) > num(prev["macd_hist"])),
        "trigger": round(trigger, 3),
        "stop": round(stop, 3),
        "target_2r": round(target, 3),
        "risk_pct": pct(risk_pct),
        "risk_reward": 2.0,
        "amount_yi": safe_round(num(spot_row.get("amount")) / 1e8, 2),
        "turnover": safe_round(spot_row.get("turnover"), 2),
        "volume_ratio_spot": safe_round(spot_row.get("volume_ratio"), 2),
        "impulse_start_date": d.iloc[setup.impulse_start_idx]["date"].strftime("%Y-%m-%d"),
        "impulse_peak_date": d.iloc[setup.impulse_peak_idx]["date"].strftime("%Y-%m-%d"),
        "impulse_gain_pct": pct(setup.impulse_gain),
        "impulse_days": setup.impulse_days,
        "impulse_volume_ratio": round(setup.impulse_volume_ratio, 2),
        "impulse_max_volume_ratio": round(setup.impulse_max_volume_ratio, 2),
        "pullback_days": setup.pullback_days,
        "pullback_speed_pct_day": pct(setup.pullback_speed),
        "down_volume_ratio": round(setup.down_volume_ratio, 2),
        "support_name": setup.support_name,
        "support_price": round(setup.support, 3),
        "support_distance_pct": pct(setup.support_distance),
        "confirmation_return_pct": pct(setup.confirmation_return),
        "scores": {
            "trend": round(trend_score, 1),
            "impulse": round(impulse_score, 1),
            "pullback": round(pullback_score, 1),
            "support_confirmation": round(support_score, 1),
            "relative_industry": round(relative, 1),
            "risk": round(risk_score, 1),
        },
        "reason": [
            (
                f"主升段 {setup.impulse_days} 日上涨 "
                f"{pct(setup.impulse_gain):.1f}%，平均量能放大 "
                f"{setup.impulse_volume_ratio:.2f} 倍"
            ),
            (
                f"回调 {setup.pullback_days} 日、回撤 "
                f"{pct(setup.drawdown):.1f}%，平均每天仅 "
                f"{pct(setup.pullback_speed):.2f}%"
            ),
            (
                f"回调量缩至主升段的 {setup.contraction:.2f}，"
                f"下跌日量能比 {setup.down_volume_ratio:.2f}"
            ),
            (
                f"最新低点靠近 {setup.support_name} "
                f"{setup.support:.2f}，收阳确认涨幅 "
                f"{pct(setup.confirmation_return):.2f}%"
                if setup.state == "A"
                else (
                    f"最新低点靠近 {setup.support_name} "
                    f"{setup.support:.2f}，尚待收阳确认"
                )
            ),
            f"触发 {trigger:.2f} / 失效止损 {stop:.2f} / 2R目标 {target:.2f}",
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
        pct_col = "涨跌幅"
        vals = pd.to_numeric(boards[pct_col], errors="coerce")
        ranks = vals.rank(pct=True) * 100
        for i, row in boards.iterrows():
            name = str(row.get("板块名称", "未分类"))
            scores[name] = num(ranks.loc[i], 50)
            ranking.append({"name": name, "change_pct": safe_round(row.get(pct_col), 2), "score": round(scores[name], 1)})
        # Fetch constituents concurrently; this is optional and failures are neutral.
        def one(row):
            name = str(row.get("板块名称"))
            code = str(row.get("板块代码"))
            try:
                c = ak.stock_board_industry_cons_em(symbol=code)
                return name, [str(x).zfill(6) for x in c.get("代码", [])]
            except Exception:
                return name, []
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            for name, codes in ex.map(one, [r for _, r in boards.iterrows()]):
                for code in codes:
                    if code in spot_codes and code not in mapping:
                        mapping[code] = name
        ranking.sort(key=lambda x: x["score"], reverse=True)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"行业映射失败，板块分按中性处理：{type(exc).__name__}")
    return mapping, scores, ranking[:20]


def choose_universe(spot: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    d = spot.copy()
    # 仅扫描沪深主板：
    # 深市主板 000/001/002/003；沪市主板 600/601/603/605。
    # 明确排除创业板 300/301、科创板 688/689、北交所等。
    mainboard_pattern = r"^(?:000|001|002|003|600|601|603|605)\d{3}$"
    d = d[d["code"].str.match(mainboard_pattern, na=False)]
    d = d[~d["name"].astype(str).str.contains(r"ST|退|N |C ", case=False, regex=True, na=False)]
    d = d[(d["price"] >= args.min_price) & (d["price"] <= args.max_price)]
    d = d[d["amount"].fillna(0) >= args.min_amount_yi * 1e8]
    d = d[d["volume"].fillna(0) > 0]
    # Current amount is used only as a liquidity prefilter; sort for optional cap.
    d = d.sort_values("amount", ascending=False)
    if args.max_stocks > 0:
        d = d.head(args.max_stocks)
    return d.reset_index(drop=True)


def write_outputs(payload: dict[str, Any]) -> None:
    (DATA_DIR / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "state", "signal", "score", "code", "name", "industry", "price", "change_pct",
        "impulse_start_date", "impulse_peak_date", "impulse_gain_pct", "impulse_days",
        "impulse_volume_ratio", "impulse_max_volume_ratio", "pullback_days",
        "drawdown_pct", "pullback_speed_pct_day", "pullback_volume_ratio",
        "down_volume_ratio", "support_name", "support_price", "support_distance_pct",
        "confirmation_return_pct", "distance_ma20_pct", "distance_ma30_pct",
        "r10_pct", "r20_pct", "industry_score", "trigger", "stop", "target_2r",
        "risk_pct", "risk_reward", "amount_yi", "turnover",
    ]
    with (DATA_DIR / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(payload.get("candidates", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intraday", "close"], default="close")
    ap.add_argument("--max-price", type=float, default=150.0)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--min-amount-yi", type=float, default=0.8, help="当日最低成交额，亿元")
    ap.add_argument("--max-stocks", type=int, default=0, help="0=全部；验证时可限制数量")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--bars", type=int, default=280)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    started = time.time()
    now = datetime.now(SH_TZ)
    warnings: list[str] = []
    print(f"[{now:%F %T}] 开始 {args.mode} 扫描")
    spot, source, src_warnings = get_spot()
    warnings.extend(src_warnings)
    universe = choose_universe(spot, args)
    print(f"实时快照 {len(spot)} 只，基础过滤后 {len(universe)} 只；数据源 {source}")
    if len(universe) < 300:
        raise RuntimeError(f"基础过滤后仅 {len(universe)} 只，疑似数据异常，停止发布")

    industry_map, industry_scores, industry_ranking = build_industry_map(set(universe["code"]), warnings)

    histories: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    fallback_count = 0
    rows_by_code = {str(r["code"]): r for _, r in universe.iterrows()}

    def task(code: str):
        return fetch_history(code, args.bars)

    codes = list(rows_by_code)
    with cf.ThreadPoolExecutor(max_workers=max(4, args.workers)) as ex:
        futs = {ex.submit(task, c): c for c in codes}
        total = len(futs)
        done = 0
        for fut in cf.as_completed(futs):
            code, hist, note = fut.result()
            done += 1
            if hist is not None and len(hist) >= 90:
                histories[code] = merge_spot_bar(hist, rows_by_code[code], now)
                if note:
                    fallback_count += 1
            else:
                failures.append(code)
            if done % 250 == 0 or done == total:
                print(f"历史行情进度 {done}/{total}，成功 {len(histories)}，失败 {len(failures)}")

    ok_ratio = len(histories) / max(1, len(codes))
    if len(histories) < 300 or ok_ratio < 0.70:
        raise RuntimeError(f"历史行情成功率仅 {ok_ratio:.1%}，停止发布，保留上一版结果")
    if failures:
        warnings.append(f"历史行情失败 {len(failures)} 只；成功率 {ok_ratio:.1%}")
    if fallback_count:
        warnings.append(f"{fallback_count} 只使用AKShare个股历史回退源")

    # Market breadth and median returns from valid histories.
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
            item = analyze_stock(code, str(r["name"]), industry, industry_score, h, market_r20, breadth, r)
            if item:
                candidates.append(item)
        except Exception as exc:  # noqa: BLE001
            failures.append(code)
            if len(warnings) < 10:
                warnings.append(f"{code} 计算异常：{type(exc).__name__}")

    state_rank = {"A": 0, "B": 1}
    candidates.sort(
        key=lambda x: (
            state_rank.get(x["state"], 9),
            -x["score"],
            x.get("pullback_speed_pct_day", 99),
            -num(x["amount_yi"], 0),
        )
    )
    # 平衡版：确认型最多15只，待确认最多15只。
    caps = {"A": 15, "B": 15}
    selected: list[dict[str, Any]] = []
    counts = {"A": 0, "B": 0, "C": 0}
    for x in candidates:
        s = x["state"]
        if s in caps and counts[s] < caps[s] and len(selected) < min(args.top, 30):
            selected.append(x)
            counts[s] += 1

    generated = datetime.now(SH_TZ)
    payload = {
        "schema": 3,
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
            "C": counts["C"],
        },
        "industry_ranking": industry_ranking,
        "candidates": selected,
    }
    write_outputs(payload)
    print(f"完成：阳线确认 {counts['A']} / 待确认 {counts['B']}，总候选 {len(selected)}，耗时 {payload['meta']['elapsed_seconds']} 秒")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"SCAN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
