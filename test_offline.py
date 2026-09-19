#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线冒烟测试：用合成数据验证指标计算、买点判定、评分与 HTML 生成。
不联网，专门用来在无法访问外网的环境下验证代码逻辑正确性。
运行: python test_offline.py
"""
import datetime
import numpy as np
import pandas as pd

import screener as S


def make_series(kind):
    """合成三种形态的日线：breakout / pullback / diprecover
    先生成基础走势，再按真实均线水平微调最后一根 K 线，确保形态精确达标。
    """
    n = 90
    rng = np.random.default_rng(7)
    if kind == "breakout":
        # 长期横盘 -> 末日放量突破
        closes = np.concatenate([10 + rng.normal(0, 0.12, 72),
                                 np.linspace(10.1, 10.9, 17), [10.95]])
        vol = np.concatenate([rng.uniform(1.0e5, 1.4e5, 89), [1.3e5]])
    elif kind == "pullback":
        # 稳步上涨 -> 末段缩量回踩
        closes = np.concatenate([np.linspace(9.0, 12.0, 70),
                                 np.linspace(12.0, 15.0, 15),
                                 np.linspace(14.9, 14.2, 5)])
        vol = np.concatenate([rng.uniform(4e5, 6e5, 85), rng.uniform(2e5, 3e5, 5)])
    else:
        # 上涨后在 MA20 附近横盘 -> 末日盘中砸破再拉回
        closes = np.concatenate([np.linspace(9.0, 12.0, 70),
                                 13.4 + rng.normal(0, 0.08, 20)])
        vol = rng.uniform(4e5, 6e5, 90)

    highs = closes + np.abs(rng.normal(0, 0.10, n))
    lows = closes - np.abs(rng.normal(0, 0.10, n))
    opens = closes + rng.normal(0, 0.05, n)
    dates = [(datetime.date(2026, 5, 1) + datetime.timedelta(days=int(i * 7 / 5))
              ).strftime("%Y-%m-%d") for i in range(n)]
    df = pd.DataFrame({"date": dates, "open": opens, "close": closes,
                       "high": highs, "low": lows, "volume": vol})

    # 依据已算出的均线，精确调整最后一根 K 线
    d = S.calc_indicators(df)
    cur = d.iloc[-1]
    prev5_vol = d["volume"].iloc[-6:-1].mean()
    prev_close = df["close"].iloc[-2]
    if kind == "breakout":
        target = max(cur["boll_up"], df["high"].iloc[-21:-1].max())
        df.loc[n - 1, "close"] = target * 1.03
        df.loc[n - 1, "high"] = target * 1.05
        df.loc[n - 1, "low"] = target * 0.99
        df.loc[n - 1, "volume"] = prev5_vol * 3.0
    elif kind == "pullback":
        ma20 = cur["ma20"]
        df.loc[n - 1, "close"] = ma20 * 1.035
        df.loc[n - 1, "low"] = ma20 * 1.010
        df.loc[n - 1, "high"] = ma20 * 1.055
        df.loc[n - 1, "volume"] = prev5_vol * 0.5
    else:
        ma20 = cur["ma20"]
        df.loc[n - 1, "low"] = min(ma20 * 0.99, prev_close * 0.975)
        df.loc[n - 1, "close"] = prev_close * 1.02
        df.loc[n - 1, "high"] = prev_close * 1.035
        df.loc[n - 1, "volume"] = prev5_vol * 1.2
    return df


def main():
    print("=" * 60)
    print("离线冒烟测试（合成数据，不联网）")
    print("=" * 60)

    cases = [("breakout", 1, "买点1 突破转势"),
             ("pullback", 2, "买点2 回踩不破"),
             ("diprecover", 3, "买点3 破线拉回")]
    payload = {"date": "2026-09-18", "total": 0, "results": {"1": [], "2": [], "3": []}}
    ok = True

    for kind, expect, label in cases:
        df = make_series(kind)
        d = S.calc_indicators(df)
        cur = d.iloc[-1]
        amp = (cur["high"] - cur["low"]) / d.iloc[-2]["close"] * 100
        prev_close = d.iloc[-2]["close"]
        spot = {"pct": (cur["close"] / prev_close - 1) * 100,
                "turn": 6.5, "amp": amp, "vr": 1.9,
                "pe": 28.0, "pb": 2.4, "float_mv": 120e8}
        bp, why = S.judge_buypoint(d, spot)
        mark = "OK " if bp == expect else "FAIL"
        if bp != expect:
            ok = False
        print(f"[{mark}] {label:<16} 判定=买点{bp}  期望=买点{expect}")
        print(f"        收盘{cur['close']:.2f} MA20={cur['ma20']:.2f} "
              f"MA5={cur['ma5']:.2f} DIF={cur['dif']:.3f} DEA={cur['dea']:.3f}")
        print(f"        理由: {why}")
        if bp:
            rec = {"code": "000001", "name": f"测试{kind}", "bp": bp, "why": why,
                   "close": round(float(cur["close"]), 2),
                   "pct": round(spot["pct"], 2), "turn": 6.5, "pe": 28.0, "pb": 2.4,
                   "float_mv": 120.0,
                   "ma5": round(float(cur["ma5"]), 2), "ma10": round(float(cur["ma10"]), 2),
                   "ma20": round(float(cur["ma20"]), 2),
                   "ma60": round(float(cur["ma60"]), 2) if not pd.isna(cur["ma60"]) else None,
                   "dif": round(float(cur["dif"]), 3), "dea": round(float(cur["dea"]), 3),
                   "ret20": round(float(cur["ret20"]), 1) if not pd.isna(cur["ret20"]) else None,
                   "vr": 1.9,
                   "bias20": round((float(cur["close"]) / float(cur["ma20"]) - 1) * 100, 1),
                   "chart": {"dates": d["date"].tolist()[-60:],
                             "ohlc": [[float(o), float(c), float(l), float(h)] for o, c, l, h
                                      in zip(d["open"].tolist()[-60:], d["close"].tolist()[-60:],
                                             d["low"].tolist()[-60:], d["high"].tolist()[-60:])],
                             "vol": [float(x) for x in d["volume"].tolist()[-60:]],
                             "ma5": [None if pd.isna(x) else round(float(x), 2) for x in d["ma5"].tolist()[-60:]],
                             "ma10": [None if pd.isna(x) else round(float(x), 2) for x in d["ma10"].tolist()[-60:]],
                             "ma20": [None if pd.isna(x) else round(float(x), 2) for x in d["ma20"].tolist()[-60:]]}}
            rec["score"] = S.score(rec, bp)
            print(f"        综合评分: {rec['score']}")
            payload["results"][str(bp)].append(rec)

    payload["total"] = sum(len(v) for v in payload["results"].values())

    print("\n" + "=" * 60)
    print("HTML 报告生成测试")
    print("=" * 60)
    html = S.build_html(payload)
    checks = {
        "包含 DOCTYPE 与中文标题": "<!DOCTYPE html>" in html and "波段买点" in html,
        "ECharts CDN 已引入": "echarts.min.js" in html,
        "K线数据已注入": '"ohlc"' in html or "candlestick" in html,
        "红涨绿跌配色": "#d32f2f" in html and "#2e9e6b" in html,
        "无未替换占位符": "__DATE__" not in html and "__BODY__" not in html and "__JS__" not in html,
        "含免责声明": "不构成任何投资建议" in html,
    }
    for k, v in checks.items():
        print(f"[{'OK ' if v else 'FAIL'}] {k}")
        if not v:
            ok = False
    print(f"\nHTML 长度: {len(html):,} 字符")

    with open("output/test_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("已写出 output/test_report.html")

    print("\n" + "=" * 60)
    print("测试结果:", "全部通过" if ok else "存在失败项")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
