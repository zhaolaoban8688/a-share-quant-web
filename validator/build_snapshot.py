#!/usr/bin/env python3
"""A股波段启动二次验证器 V1.1 - 全市场量化快照。

目标：不改变原技术选股器，只复用其稳定的数据抓取函数，生成供手机网页
本地筛选/排名的全市场特征快照。V1.1 是纯量化层：市场环境、主线代理、
板块、资金、基础质量代理、相对强度、风险收益。新闻/政策/产业催化仍留给AI终审。

仅用于研究与复盘，不构成投资建议。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scan import (
    SH_TZ,
    add_indicators,
    build_industry_map,
    fetch_history,
    get_spot,
    merge_spot_bar,
    num,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "latest.json"

MAINBOARD_RE = r"^(?:000|001|002|003|600|601|603|605)\d{3}$"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def scale(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x) or hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def fround(x: Any, n: int = 3) -> float | None:
    v = num(x)
    return round(v, n) if math.isfinite(v) else None


def raw_spot_value(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row.index:
            value = num(row.get(name))
            if math.isfinite(value):
                return value
    return math.nan


def choose_universe(spot: pd.DataFrame, max_stocks: int) -> pd.DataFrame:
    d = spot.copy()
    d = d[d["code"].astype(str).str.match(MAINBOARD_RE, na=False)]
    d = d[~d["name"].astype(str).str.contains(r"ST|退|N |C ", case=False, regex=True, na=False)]
    d = d[(pd.to_numeric(d["price"], errors="coerce") >= 0.5)]
    d = d[pd.to_numeric(d["volume"], errors="coerce").fillna(0) > 0]
    d = d.sort_values("amount", ascending=False)
    if max_stocks > 0:
        d = d.head(max_stocks)
    return d.reset_index(drop=True)


def calc_row(code: str, name: str, industry: str, hist: pd.DataFrame, spot_row: pd.Series) -> dict[str, Any] | None:
    d = add_indicators(hist).copy().reset_index(drop=True)
    if len(d) < 80:
        return None
    last = d.iloc[-1]
    close = num(last.get("close"))
    ma20 = num(last.get("ma20"))
    atr = num(last.get("atr14"))
    if not math.isfinite(close) or close <= 0:
        return None

    def ret(n: int) -> float:
        if len(d) <= n:
            return math.nan
        old = num(d.iloc[-1 - n].get("close"))
        return close / old - 1 if old > 0 else math.nan

    r5, r10, r20, r60 = ret(5), ret(10), ret(20), ret(60)
    recent20 = d.tail(20).copy()
    recent21 = d.tail(21).copy()
    prev20 = d.iloc[max(0, len(d) - 21):len(d) - 1]

    vol_now = num(last.get("volume"), 0)
    vol_med = num(prev20["volume"].median()) if len(prev20) else math.nan
    volume_ratio = vol_now / vol_med if vol_med > 0 else math.nan

    amount_now = num(last.get("amount"), 0)
    amount_med = num(prev20["amount"].median()) if len(prev20) and "amount" in prev20 else math.nan
    amount_ratio = amount_now / amount_med if amount_med > 0 else math.nan

    prev_close = recent21["close"].shift(1)
    up_mask = recent21["close"] >= prev_close
    down_mask = recent21["close"] < prev_close
    up_vol = num(recent21.loc[up_mask, "volume"].median())
    down_vol = num(recent21.loc[down_mask, "volume"].median())
    up_down_volume_ratio = up_vol / down_vol if down_vol > 0 else 1.0

    px_ret = recent21["close"].pct_change()
    vol_ret = recent21["volume"].pct_change()
    pv_corr = num(px_ret.corr(vol_ret), 0.0)

    high20 = num(recent20["high"].max())
    drawdown20 = close / high20 - 1 if high20 > 0 else math.nan
    atr_pct = atr / close if close > 0 and math.isfinite(atr) else math.nan
    ma20_dist = close / ma20 - 1 if ma20 > 0 else math.nan

    pe = raw_spot_value(spot_row, ["市盈率-动态", "市盈率", "pe"])
    pb = raw_spot_value(spot_row, ["市净率", "pb"])
    market_cap = raw_spot_value(spot_row, ["总市值", "market_cap"])
    float_cap = raw_spot_value(spot_row, ["流通市值", "float_cap"])
    turnover = num(spot_row.get("turnover"))
    change_pct = num(spot_row.get("change_pct"))

    return {
        "code": code,
        "name": name,
        "industry": industry or "未分类",
        "price": round(close, 3),
        "change_pct": fround(change_pct, 2),
        "r5": r5,
        "r10": r10,
        "r20": r20,
        "r60": r60,
        "ma20_dist": ma20_dist,
        "atr_pct": atr_pct,
        "drawdown20": drawdown20,
        "volume_ratio": volume_ratio,
        "amount_ratio": amount_ratio,
        "up_down_volume_ratio": up_down_volume_ratio,
        "pv_corr": pv_corr,
        "turnover": turnover,
        "amount_yi": amount_now / 1e8 if amount_now > 0 else math.nan,
        "pe": pe,
        "pb": pb,
        "market_cap_yi": market_cap / 1e8 if market_cap > 0 else math.nan,
        "float_cap_yi": float_cap / 1e8 if float_cap > 0 else math.nan,
    }


def market_score(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    breadth_ma20 = float((frame["ma20_dist"] > 0).mean()) if len(frame) else 0.0
    breadth_r20 = float((frame["r20"] > 0).mean()) if len(frame) else 0.0
    med_r20 = float(frame["r20"].median()) if len(frame) else 0.0
    score = (
        scale(breadth_ma20, 0.32, 0.68) * 5
        + scale(breadth_r20, 0.32, 0.68) * 5
        + scale(med_r20, -0.06, 0.10) * 5
    )
    return round(score, 1), {
        "breadth_above_ma20": round(breadth_ma20 * 100, 1),
        "breadth_r20_positive": round(breadth_r20 * 100, 1),
        "median_r20_pct": round(med_r20 * 100, 2),
    }


def quality_proxy(row: pd.Series) -> float:
    """短中期基础质量代理，不冒充真实基本面/催化评分。"""
    score = 2.0  # 通过主板/ST/成交数据卫生过滤
    amount_yi = num(row.get("amount_yi"))
    pe = num(row.get("pe"))
    pb = num(row.get("pb"))
    cap = num(row.get("market_cap_yi"))

    score += scale(amount_yi, 0.5, 12.0) * 4
    if math.isfinite(pe):
        if 0 < pe <= 80:
            score += 4
        elif 80 < pe <= 150:
            score += 2.5
        elif pe > 150:
            score += 1.0
        else:
            score += 0.5
    else:
        score += 1.5
    if math.isfinite(pb):
        score += 3.0 if 0 < pb <= 8 else 1.5 if 0 < pb <= 15 else 0.5
    else:
        score += 1.0
    score += scale(cap, 20, 800) * 2
    return round(clamp(score, 0, 15), 1)


def add_scores(frame: pd.DataFrame, industry_current: dict[str, float], market_sc: float) -> pd.DataFrame:
    d = frame.copy()
    if d.empty:
        return d

    groups = d.groupby("industry").agg(
        industry_r5=("r5", "median"),
        industry_r20=("r20", "median"),
        industry_breadth20=("r20", lambda x: float((x > 0).mean())),
        industry_count=("code", "count"),
    )
    groups["r5_rank"] = groups["industry_r5"].rank(pct=True, method="average") * 100
    groups["r20_rank"] = groups["industry_r20"].rank(pct=True, method="average") * 100

    d = d.join(groups, on="industry")
    d["industry_current"] = d["industry"].map(industry_current).fillna(50.0)

    market_r5 = num(d["r5"].median(), 0)
    market_r20 = num(d["r20"].median(), 0)
    market_r60 = num(d["r60"].median(), 0)

    rows = []
    for _, row in d.iterrows():
        current = num(row.get("industry_current"), 50)
        r5_rank = num(row.get("r5_rank"), 50)
        r20_rank = num(row.get("r20_rank"), 50)
        ind_r5 = num(row.get("industry_r5"), 0)
        ind_r20 = num(row.get("industry_r20"), 0)
        ind_breadth = num(row.get("industry_breadth20"), 0.5)

        mainline = current / 100 * 8 + r5_rank / 100 * 6 + r20_rank / 100 * 6
        sector = (
            scale(ind_r5, -0.04, 0.10) * 5
            + scale(ind_r20, -0.08, 0.18) * 5
            + scale(ind_breadth, 0.30, 0.78) * 5
        )

        funds = (
            scale(num(row.get("volume_ratio")), 0.65, 2.0) * 6
            + scale(num(row.get("amount_ratio")), 0.65, 2.0) * 5
            + scale(num(row.get("up_down_volume_ratio")), 0.65, 1.8) * 4
            + scale(num(row.get("pv_corr")), -0.25, 0.55) * 3
            + scale(num(row.get("turnover")), 0.5, 8.0) * 2
        )

        r5 = num(row.get("r5"), 0)
        r10 = num(row.get("r10"), 0)
        r20 = num(row.get("r20"), 0)
        r60 = num(row.get("r60"), 0)
        rs = (
            scale(r5 - ind_r5, -0.04, 0.08) * 2.5
            + scale(r20 - ind_r20, -0.08, 0.16) * 2.5
            + scale(r20 - market_r20, -0.08, 0.16) * 2.5
            + scale(r60 - market_r60, -0.15, 0.30) * 2.5
        )

        atr_pct = num(row.get("atr_pct"))
        ma20_dist = abs(num(row.get("ma20_dist")))
        dd = abs(min(0.0, num(row.get("drawdown20"), 0.0)))
        risk = (
            (1 - scale(atr_pct, 0.025, 0.09)) * 2
            + (1 - scale(ma20_dist, 0.03, 0.16)) * 2
            + (1 - scale(dd, 0.08, 0.28)) * 1
        )
        risk = clamp(risk, 0, 5)
        fundamental = quality_proxy(row)

        total = clamp(market_sc + mainline + sector + funds + fundamental + rs + risk, 0, 100)
        veto = None
        if atr_pct > 0.13:
            veto = "波动率过高（ATR14/价格>13%）"
        elif dd > 0.35:
            veto = "近20日回撤过深（>35%）"

        grade = "A+" if total >= 85 else "A" if total >= 78 else "B" if total >= 70 else "C" if total >= 60 else "D"
        reasons = [
            f"行业当前强度分位 {current:.0f}，行业5日/20日强度分位 {r5_rank:.0f}/{r20_rank:.0f}",
            f"个股5日/20日收益 {r5*100:.1f}%/{r20*100:.1f}%，行业中位 {ind_r5*100:.1f}%/{ind_r20*100:.1f}%",
            f"量比(对20日中位量) {num(row.get('volume_ratio')):.2f}，成交额比 {num(row.get('amount_ratio')):.2f}，上涨/下跌日量能比 {num(row.get('up_down_volume_ratio')):.2f}",
            f"ATR {atr_pct*100:.1f}%，距MA20 {num(row.get('ma20_dist'))*100:.1f}%，20日高点回撤 {num(row.get('drawdown20'))*100:.1f}%",
        ]

        item = row.to_dict()
        item.update({
            "market_score": round(market_sc, 1),
            "mainline_score": round(mainline, 1),
            "sector_score": round(sector, 1),
            "funds_score": round(funds, 1),
            "fundamental_score": round(fundamental, 1),
            "rs_score": round(rs, 1),
            "risk_score": round(risk, 1),
            "quant_score": round(total, 1),
            "grade": grade,
            "veto": veto,
            "reasons": reasons,
        })
        rows.append(item)

    out = pd.DataFrame(rows)
    return out.sort_values(["quant_score", "funds_score", "rs_score"], ascending=False)


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=0)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--bars", type=int, default=180)
    args = ap.parse_args()

    started = time.time()
    now = datetime.now(SH_TZ)
    warnings: list[str] = []
    print(f"[{now:%F %T}] 开始二次验证器 V1.1 全市场快照")

    spot, source, src_warnings = get_spot()
    warnings.extend(src_warnings)
    universe = choose_universe(spot, args.max_stocks)
    if len(universe) < 250 and args.max_stocks == 0:
        raise RuntimeError(f"主板有效股票仅 {len(universe)}，疑似快照异常")
    print(f"主板基础股票 {len(universe)}；快照源 {source}")

    industry_map, industry_current, industry_ranking = build_industry_map(set(universe["code"]), warnings)
    rows_by_code = {str(r["code"]): r for _, r in universe.iterrows()}

    histories: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    fallback_count = 0
    with cf.ThreadPoolExecutor(max_workers=max(4, args.workers)) as ex:
        futs = {ex.submit(fetch_history, code, args.bars): code for code in rows_by_code}
        for done, fut in enumerate(cf.as_completed(futs), 1):
            code, hist, note = fut.result()
            if hist is not None and len(hist) >= 80:
                histories[code] = merge_spot_bar(hist, rows_by_code[code], now)
                if note:
                    fallback_count += 1
            else:
                failures.append(code)
            if done % 250 == 0 or done == len(futs):
                print(f"历史行情 {done}/{len(futs)}，成功 {len(histories)}，失败 {len(failures)}")

    ok_ratio = len(histories) / max(1, len(rows_by_code))
    if args.max_stocks == 0 and (len(histories) < 300 or ok_ratio < 0.70):
        raise RuntimeError(f"历史行情成功率仅 {ok_ratio:.1%}，停止覆盖上一份验证快照")
    if failures:
        warnings.append(f"历史行情失败 {len(failures)} 只；成功率 {ok_ratio:.1%}")
    if fallback_count:
        warnings.append(f"{fallback_count} 只使用AKShare历史回退源")

    records = []
    stages: Counter[str] = Counter()
    for code, hist in histories.items():
        srow = rows_by_code[code]
        try:
            item = calc_row(code, str(srow["name"]), industry_map.get(code, "未分类"), hist, srow)
            if item:
                records.append(item)
                stages["特征成功"] += 1
            else:
                stages["特征不足"] += 1
        except Exception as exc:  # noqa: BLE001
            stages["特征异常"] += 1
            if len(warnings) < 12:
                warnings.append(f"{code} 特征异常：{type(exc).__name__}")

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("没有生成有效特征")

    mscore, mstats = market_score(frame)
    scored = add_scores(frame, industry_current, mscore)

    candidates: list[dict[str, Any]] = []
    keep_fields = [
        "code", "name", "industry", "price", "change_pct", "quant_score", "grade", "veto",
        "market_score", "mainline_score", "sector_score", "funds_score", "fundamental_score", "rs_score", "risk_score",
        "r5", "r10", "r20", "r60", "industry_r5", "industry_r20", "industry_breadth20", "industry_current",
        "volume_ratio", "amount_ratio", "up_down_volume_ratio", "pv_corr", "turnover", "amount_yi",
        "atr_pct", "ma20_dist", "drawdown20", "pe", "pb", "market_cap_yi", "reasons",
    ]
    for _, row in scored.iterrows():
        obj = {k: clean_json_value(row.get(k)) for k in keep_fields}
        candidates.append(obj)

    state = "强势" if mscore >= 11.5 else "正常" if mscore >= 8 else "偏弱" if mscore >= 5 else "弱势"
    payload = {
        "schema": 1,
        "validator_version": "V1.1",
        "meta": {
            "status": "success",
            "generated_at": now.isoformat(timespec="seconds"),
            "market_scope": "A股沪深主板",
            "data_source": f"{source} + 腾讯日K/AKShare回退",
            "universe_count": int(len(universe)),
            "feature_count": int(len(candidates)),
            "history_success_ratio": round(ok_ratio, 4),
            "elapsed_seconds": round(time.time() - started, 1),
            "warnings": warnings[:20],
            "note": "V1.1为纯量化二次验证层；基本面/催化分仅是基础质量代理，不包含实时新闻、政策与产业催化，AI终审将在后续版本接入。",
            "disclaimer": "仅供量化研究与复盘，不构成投资建议。",
        },
        "market": {"score": mscore, "state": state, **mstats},
        "industry_ranking": industry_ranking[:20],
        "diagnostics": dict(stages),
        "stocks": candidates,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"完成：{len(candidates)}只；市场 {state} {mscore}/15；耗时 {payload['meta']['elapsed_seconds']} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
