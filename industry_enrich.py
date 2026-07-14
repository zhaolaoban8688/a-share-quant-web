#!/usr/bin/env python3
"""为扫描结果补充行业、行业强度和行业排名。

主分类源：BaoStock 免费行业分类（无需注册）。
候选缺失时：AKShare 东方财富个股资料兜底。
行业强度：按全市场当日行业中位涨幅与上涨家数比例计算。
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LATEST = DATA_DIR / "latest.json"
CACHE = DATA_DIR / "industry_cache.json"
CSV_PATH = DATA_DIR / "candidates.csv"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_code(value: Any) -> str:
    text = str(value).lower().replace("sh.", "").replace("sz.", "").replace("bj.", "")
    text = text.replace("sh", "").replace("sz", "").replace("bj", "")
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def load_cache() -> dict[str, str]:
    if not CACHE.exists():
        return {}
    try:
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
        return {normalize_code(k): str(v).strip() for k, v in raw.items() if str(v).strip()}
    except Exception:
        return {}


def fetch_baostock_industries() -> tuple[dict[str, str], str]:
    import baostock as bs

    mapping: dict[str, str] = {}
    login = bs.login()
    try:
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock登录失败: {login.error_msg}")
        result = bs.query_stock_industry()
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock行业查询失败: {result.error_msg}")
        fields = list(result.fields)
        while result.next():
            row = dict(zip(fields, result.get_row_data()))
            code = normalize_code(row.get("code", ""))
            industry = str(row.get("industry", "")).strip()
            if code and industry and industry not in {"None", "nan", "未知"}:
                mapping[code] = industry
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    if not mapping:
        raise RuntimeError("BaoStock行业分类为空")
    return mapping, "BaoStock行业分类"


def fetch_candidate_industry_em(code: str) -> tuple[str, str]:
    try:
        frame = ak.stock_individual_info_em(symbol=code)
        if frame is None or frame.empty:
            return code, ""
        item_col = "item" if "item" in frame.columns else "项目"
        value_col = "value" if "value" in frame.columns else "值"
        if item_col not in frame.columns or value_col not in frame.columns:
            return code, ""
        for _, row in frame.iterrows():
            item = str(row.get(item_col, ""))
            if "行业" in item:
                value = str(row.get(value_col, "")).strip()
                if value and value not in {"None", "nan", "未知"}:
                    return code, value
    except Exception:
        pass
    return code, ""


def normalize_spot(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {"代码": "code", "名称": "name", "涨跌幅": "change_pct"}
    result = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns}).copy()
    for col in ("code", "name", "change_pct"):
        if col not in result.columns:
            result[col] = None
    result["code"] = result["code"].map(normalize_code)
    result["change_pct"] = pd.to_numeric(result["change_pct"], errors="coerce")
    return result.dropna(subset=["change_pct"])


def get_market_spot() -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for label, getter in (
        ("东方财富", ak.stock_zh_a_spot_em),
        ("新浪", ak.stock_zh_a_spot),
    ):
        try:
            frame = getter()
            if frame is not None and not frame.empty:
                return normalize_spot(frame), label
        except Exception as exc:
            errors.append(f"{label}:{type(exc).__name__}")
            time.sleep(1)
    raise RuntimeError("行业强度快照失败 " + "/".join(errors))


def build_industry_scores(
    spot: pd.DataFrame, mapping: dict[str, str]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    data = spot.copy()
    data["industry"] = data["code"].map(mapping)
    data = data.dropna(subset=["industry", "change_pct"])
    data = data[data["industry"].astype(str).str.len() > 0]
    if data.empty:
        return {}, []

    grouped = data.groupby("industry").agg(
        stock_count=("code", "count"),
        median_change=("change_pct", "median"),
        mean_change=("change_pct", "mean"),
        breadth=("change_pct", lambda x: float((x > 0).mean() * 100)),
    )
    grouped = grouped[grouped["stock_count"] >= 3].copy()
    if grouped.empty:
        return {}, []

    grouped["raw"] = (
        grouped["median_change"] * 0.72
        + (grouped["breadth"] - 50.0) * 0.035
    )
    grouped["score"] = (
        grouped["raw"].rank(pct=True, method="average") * 100.0
    )
    scores = {
        str(idx): round(float(row["score"]), 1)
        for idx, row in grouped.iterrows()
    }
    ranking = []
    ordered = grouped.sort_values(
        ["score", "median_change"], ascending=False
    ).head(20)
    for name, row in ordered.iterrows():
        ranking.append(
            {
                "name": str(name),
                "change_pct": round(float(row["median_change"]), 2),
                "breadth": round(float(row["breadth"]), 1),
                "stock_count": int(row["stock_count"]),
                "score": round(float(row["score"]), 1),
            }
        )
    return scores, ranking


def write_candidates_csv(candidates: list[dict[str, Any]]) -> None:
    fields = [
        "state", "signal", "score", "code", "name", "industry",
        "price", "change_pct", "platform", "breakout_date",
        "days_after_breakout", "breakout_volume_ratio",
        "pullback_volume_ratio", "distance_ma20_pct", "r10_pct",
        "r20_pct", "industry_score", "trigger", "stop",
        "target_2r", "risk_pct", "risk_reward", "amount_yi",
        "turnover",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(candidates)


def main() -> int:
    if not LATEST.exists():
        print("没有 data/latest.json，跳过行业补充")
        return 0

    payload = json.loads(LATEST.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        print("本次没有候选，跳过行业补充")
        return 0

    warnings = list(payload.get("meta", {}).get("warnings", []))
    cache = load_cache()
    source_notes: list[str] = []

    try:
        fresh, source = fetch_baostock_industries()
        cache.update(fresh)
        source_notes.append(source)
        print(f"BaoStock行业分类 {len(fresh)} 只")
    except Exception as exc:
        warnings.append(f"BaoStock行业分类失败：{type(exc).__name__}")
        print(warnings[-1])

    missing = [
        normalize_code(x.get("code"))
        for x in candidates
        if not cache.get(normalize_code(x.get("code")))
    ]
    if missing:
        with cf.ThreadPoolExecutor(max_workers=4) as executor:
            for code, industry in executor.map(
                fetch_candidate_industry_em, missing
            ):
                if industry:
                    cache[code] = industry
        source_notes.append("东方财富个股资料兜底")

    CACHE.write_text(
        json.dumps(
            dict(sorted(cache.items())),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    industry_scores: dict[str, float] = {}
    ranking: list[dict[str, Any]] = []
    try:
        spot, spot_source = get_market_spot()
        industry_scores, ranking = build_industry_scores(spot, cache)
        source_notes.append(f"{spot_source}板块强度")
        print(f"行业强度覆盖 {len(industry_scores)} 个行业")
    except Exception as exc:
        warnings.append(
            f"行业强度计算失败，使用中性分：{type(exc).__name__}"
        )
        print(warnings[-1])

    filled = 0
    for item in candidates:
        code = normalize_code(item.get("code"))
        industry = cache.get(code, "未分类")
        if industry != "未分类":
            filled += 1

        old_industry_score = to_float(
            item.get("industry_score"), 50.0
        )
        new_industry_score = to_float(
            industry_scores.get(industry), 50.0
        )
        old_relative = to_float(
            item.get("relative_strength"), 7.5
        )
        new_relative = clamp(
            old_relative
            + (new_industry_score - old_industry_score) * 0.075,
            0.0,
            15.0,
        )
        old_score = to_float(item.get("score"), 0.0)
        new_score = clamp(
            old_score + new_relative - old_relative,
            0.0,
            100.0,
        )

        item["industry"] = industry
        item["industry_score"] = round(new_industry_score, 1)
        item["relative_strength"] = round(new_relative, 1)
        item["score"] = round(new_score, 1)

        scores = item.get("scores")
        if isinstance(scores, dict):
            scores["relative_industry"] = round(new_relative, 1)

        reasons = item.get("reason")
        if isinstance(reasons, list) and industry != "未分类":
            reasons.insert(
                0,
                f"所属行业 {industry}，当日板块强度 "
                f"{new_industry_score:.1f}",
            )

    state_order = {"A": 0, "B": 1, "C": 2}
    candidates.sort(
        key=lambda x: (
            state_order.get(str(x.get("state")), 9),
            -to_float(x.get("score"), 0),
        )
    )
    payload["candidates"] = candidates
    payload["industry_ranking"] = ranking

    meta = payload.setdefault("meta", {})
    meta["warnings"] = [
        x for x in warnings
        if "行业映射失败" not in str(x)
    ]
    meta["industry_filled"] = filled
    meta["industry_total"] = len(candidates)
    if source_notes:
        meta["industry_source"] = " + ".join(
            dict.fromkeys(source_notes)
        )
        current_source = str(meta.get("data_source", ""))
        meta["data_source"] = (
            current_source
            + (" + " if current_source else "")
            + meta["industry_source"]
        )

    LATEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_candidates_csv(candidates)
    print(
        f"行业补充完成：{filled}/{len(candidates)} 只候选已分类"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
