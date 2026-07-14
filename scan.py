#!/usr/bin/env python3
"""A股全市场手动扫描器（零成本验证版）.

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


def separated_count(indices: Iterable[int], gap: int = 5) -> int:
    count, last = 0, -10_000
    for i in sorted(indices):
        if i - last >= gap:
            count += 1
            last = i
    return count


def platform_before(d: pd.DataFrame, end_idx: int, lookback: int = 180) -> dict[str, Any] | None:
    """Find a horizontal resistance cluster before end_idx."""
    start = max(0, end_idx - lookback)
    w = d.iloc[start:end_idx].copy()
    if len(w) < 55:
        return None
    highs = w["high"].to_numpy(float)
    # local highs; edges excluded
    locs: list[int] = []
    for i in range(2, len(highs) - 2):
        if highs[i] >= np.nanmax(highs[i - 2 : i + 3]):
            locs.append(i)
    if len(locs) < 2:
        return None
    best = None
    for anchor in locs:
        level0 = highs[anchor]
        if not math.isfinite(level0) or level0 <= 0:
            continue
        members = [i for i in locs if abs(highs[i] / level0 - 1) <= 0.04]
        touches = separated_count(members, 6)
        if touches < 2:
            continue
        vals = np.array([highs[i] for i in members], dtype=float)
        level = float(np.median(vals))
        dispersion = float(np.std(vals / level))
        duration = max(members) - min(members) if len(members) > 1 else 0
        recency = max(members) / max(1, len(w) - 1)
        # prefer meaningful levels near recent price and long multi-touch zones
        recent_close = num(w.iloc[-1]["close"])
        distance = abs(recent_close / level - 1)
        score = touches * 8 + min(duration, 100) * 0.08 + recency * 5 - dispersion * 160 - distance * 8
        if best is None or score > best["cluster_score"]:
            best = {
                "level": level,
                "touches": touches,
                "duration": int(duration),
                "dispersion": dispersion,
                "cluster_score": score,
                "first_touch_idx": start + min(members),
                "last_touch_idx": start + max(members),
            }
    return best


@dataclass
class Detection:
    state: str
    breakout_idx: int
    platform: dict[str, Any]
    breakout_volume_ratio: float
    breakout_close_position: float
    contraction: float
    drawdown: float
    support: float
    restart_strength: float


def detect_state(d: pd.DataFrame) -> Detection | None:
    n = len(d)
    if n < 90:
        return None
    latest = d.iloc[-1]
    detections: list[Detection] = []
    # Search last 28 trading days for the relevant breakout event.
    for j in range(max(65, n - 28), n):
        p = platform_before(d, j, lookback=180)
        if not p:
            continue
        level = p["level"]
        row = d.iloc[j]
        prev = d.iloc[j - 1]
        atr = num(row.get("atr14"))
        vol20 = num(row.get("vma20"))
        if not math.isfinite(atr) or not math.isfinite(vol20) or vol20 <= 0:
            continue
        vr = num(row["volume"]) / vol20
        rng = max(1e-9, num(row["high"]) - num(row["low"]))
        close_pos = (num(row["close"]) - num(row["low"])) / rng
        breakout = (
            num(row["close"]) >= level * 1.006
            and num(row["close"]) >= num(prev["close"]) * 1.008
            and num(prev["close"]) <= level * 1.055
            and (num(row["close"]) - level) >= 0.15 * atr
            and vr >= 1.12
            and close_pos >= 0.56
        )
        if not breakout:
            continue
        after = d.iloc[j + 1 :]
        days = n - 1 - j
        peak = num(d.iloc[j:]["high"].max())
        drawdown = (peak - num(latest["close"])) / peak if peak > 0 else 0
        contraction = 1.0
        if len(after):
            contraction = num(after["volume"].mean()) / max(num(row["volume"]), 1e-9)
        support = max(level, num(latest.get("ma20")), num(latest.get("ma30")))
        support_hold = num(d.iloc[j:]["close"].min()) >= level * 0.94 and num(latest["close"]) >= level * 0.975
        near_support = num(latest["low"]) <= support * 1.045 and num(latest["close"]) >= support * 0.975
        # A类允许“最近几天完成回踩、今天已离开支撑重新启动”，而不要求今天仍贴着支撑。
        recent_window = d.iloc[max(j + 1, n - 8) :]
        recent_support_touch = (
            len(recent_window) > 0
            and num(recent_window["low"].min()) <= support * 1.05
            and num(recent_window["close"].min()) >= level * 0.94
        )
        # restart: close above prior 3-day high, MACD histogram turns up, volume/price confirms
        prior3 = d.iloc[max(j + 1, n - 4) : n - 1]
        prior_high = num(prior3["high"].max()) if len(prior3) else math.nan
        hist_rising = num(latest.get("macd_hist")) > num(d.iloc[-2].get("macd_hist"))
        volume_recover = num(latest["volume"]) >= num(latest.get("vma5")) * 1.02
        price_recover = num(latest["close"]) > prior_high * 1.002 if math.isfinite(prior_high) else False
        restart_strength = (1 if hist_rising else 0) + (1 if volume_recover else 0) + (1 if price_recover else 0)
        if days <= 2:
            state = "C"
        elif support_hold and 2 <= days <= 24 and contraction <= 0.92 and recent_support_touch and restart_strength >= 2 and num(latest["close"]) <= support * 1.16:
            state = "A"
        elif support_hold and 2 <= days <= 24 and contraction <= 0.92 and (near_support or drawdown >= 0.025):
            state = "B"
        else:
            continue
        detections.append(Detection(state, j, p, vr, close_pos, contraction, drawdown, support, restart_strength))
    if not detections:
        return None
    priority = {"A": 3, "B": 2, "C": 1}
    return max(detections, key=lambda x: (priority[x.state], x.breakout_idx))


def analyze_stock(code: str, name: str, industry: str, industry_score: float, d0: pd.DataFrame, market_r20: float, breadth: float, spot_row: pd.Series) -> dict[str, Any] | None:
    d = add_indicators(d0)
    if len(d) < 90:
        return None
    det = detect_state(d)
    if not det:
        return None
    last, prev = d.iloc[-1], d.iloc[-2]
    price = num(last["close"])
    ma20, ma30, ma60 = num(last["ma20"]), num(last["ma30"]), num(last["ma60"])
    atr = num(last["atr14"])
    r10, r20, r60 = num(last["r10"]), num(last["r20"]), num(last["r60"])
    dist20 = price / ma20 - 1 if ma20 > 0 else math.nan
    upper_shadow = (num(last["high"]) - max(num(last["open"]), price)) / max(price, 1e-9)
    # Hard exclusions: overheated/distribution/broken trend.
    if not (ma20 > ma30 * 0.985 and ma30 > ma60 * 0.94 and ma20 >= num(d.iloc[-6]["ma20"]) * 0.995):
        return None
    if dist20 > 0.18 or r10 > 0.38 or upper_shadow > 0.085:
        return None
    if price < det.platform["level"] * 0.965:
        return None
    neg_big = d.iloc[max(det.breakout_idx, len(d) - 12) :]
    neg_big = neg_big[(neg_big["close"] < neg_big["open"] * 0.96) & (neg_big["volume"] > neg_big["vma20"] * 1.4)]
    if len(neg_big) >= 2:
        return None

    structure = 12 + det.platform["touches"] * 4 + min(det.platform["duration"], 100) * 0.07
    structure += 4 if ma20 > ma30 else 0
    structure += 4 if ma30 > ma60 else 0
    structure = clamp(structure, 0, 30)

    breakout_q = 8 + clamp((det.breakout_volume_ratio - 1.0) * 8, 0, 8) + det.breakout_close_position * 5
    breakout_q += 6 if det.state in ("A", "B") else 2
    breakout_q += 3 if price >= det.platform["level"] else 0
    breakout_q = clamp(breakout_q, 0, 25)

    volume_q = clamp(14 - max(0, det.contraction - 0.55) * 20, 2, 14)
    if det.state == "A" and num(last["volume"]) > num(last["vma5"]):
        volume_q += 1
    volume_q = clamp(volume_q, 0, 15)

    relative = clamp(7.5 + (r20 - market_r20) * 30 + (industry_score - 50) * 0.075, 0, 15)
    market_score = clamp(3 + breadth * 11, 0, 10)

    support = max(det.platform["level"], ma20, ma30)
    stop = max(det.platform["level"] * 0.955, support - 1.0 * atr)
    trigger = max(num(last["high"]), num(d.iloc[-4:-1]["high"].max()), det.platform["level"]) * 1.002
    risk_pct = (trigger - stop) / trigger if trigger > 0 else 1
    if risk_pct < 0.018:
        stop = trigger * 0.975
        risk_pct = 0.025
    if risk_pct > 0.13:
        return None
    target = trigger + 2.2 * (trigger - stop)
    rr = (target - trigger) / max(trigger - stop, 1e-9)
    risk_score = clamp(1 + (0.10 - risk_pct) * 50, 0, 5)

    state_bonus = {"A": 6, "B": 2, "C": 0}[det.state]
    score = structure + breakout_q + volume_q + relative + market_score + risk_score + state_bonus
    score -= max(0, dist20 - 0.10) * 80
    score -= max(0, r10 - 0.25) * 30
    score = clamp(score, 0, 100)

    change_pct = num(spot_row.get("change_pct"))
    signal_name = {"A": "二次启动确认", "B": "缩量回踩待触发", "C": "平台突破跟踪"}[det.state]
    return {
        "code": code,
        "name": name,
        "industry": industry or "未分类",
        "state": det.state,
        "signal": signal_name,
        "score": round(score, 1),
        "price": round(price, 3),
        "change_pct": safe_round(change_pct, 2),
        "platform": round(det.platform["level"], 3),
        "platform_touches": det.platform["touches"],
        "platform_days": det.platform["duration"],
        "breakout_date": d.iloc[det.breakout_idx]["date"].strftime("%Y-%m-%d"),
        "days_after_breakout": len(d) - 1 - det.breakout_idx,
        "breakout_volume_ratio": round(det.breakout_volume_ratio, 2),
        "pullback_volume_ratio": round(det.contraction, 2),
        "drawdown_pct": pct(det.drawdown),
        "ma20": round(ma20, 3),
        "ma30": round(ma30, 3),
        "ma60": round(ma60, 3),
        "distance_ma20_pct": pct(dist20),
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
        "risk_reward": round(rr, 2),
        "amount_yi": safe_round(num(spot_row.get("amount")) / 1e8, 2),
        "turnover": safe_round(spot_row.get("turnover"), 2),
        "volume_ratio_spot": safe_round(spot_row.get("volume_ratio"), 2),
        "scores": {
            "structure": round(structure, 1),
            "breakout_pullback": round(breakout_q, 1),
            "volume": round(volume_q, 1),
            "relative_industry": round(relative, 1),
            "market": round(market_score, 1),
            "risk": round(risk_score, 1),
        },
        "reason": [
            f"平台约 {det.platform['level']:.2f}，有效触碰 {det.platform['touches']} 次",
            f"突破量比 {det.breakout_volume_ratio:.2f}，回踩量缩至 {det.contraction:.2f}",
            f"距离MA20 {pct(dist20):.1f}% ，20日相对市场超额 {pct(r20-market_r20):.1f}%",
            f"触发 {trigger:.2f} / 失效止损 {stop:.2f} / 2.2R目标 {target:.2f}",
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
        "state", "signal", "score", "code", "name", "industry", "price", "change_pct", "platform",
        "breakout_date", "days_after_breakout", "breakout_volume_ratio", "pullback_volume_ratio",
        "distance_ma20_pct", "r10_pct", "r20_pct", "industry_score", "trigger", "stop", "target_2r",
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
    ap.add_argument("--top", type=int, default=80)
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

    state_rank = {"A": 0, "B": 1, "C": 2}
    candidates.sort(key=lambda x: (state_rank[x["state"]], -x["score"], -num(x["amount_yi"], 0)))
    # Per-state caps avoid C class crowding out the preferred A/B signals.
    caps = {"A": 30, "B": 35, "C": 25}
    selected: list[dict[str, Any]] = []
    counts = {"A": 0, "B": 0, "C": 0}
    for x in candidates:
        s = x["state"]
        if counts[s] < caps[s] and len(selected) < args.top:
            selected.append(x)
            counts[s] += 1

    generated = datetime.now(SH_TZ)
    payload = {
        "schema": 2,
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
    print(f"完成：A {counts['A']} / B {counts['B']} / C {counts['C']}，总候选 {len(selected)}，耗时 {payload['meta']['elapsed_seconds']} 秒")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"SCAN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
