#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股波段买点云端筛选器
=====================
数据源 : AkShare（东方财富）
运行环境: GitHub Actions（无需本机开机）
输出   : HTML 报告（ECharts 交互K线）+ 微信推送摘要

三种买点定义（收盘后口径）：
  买点1 突破转势 : 收盘破布林上轨或创20日新高 + 均线转多 + MACD转强 + 放量
  买点2 回踩不破 : 上升趋势中缩量回调 + 最低价未破 MA20 + 收盘站上 MA20
  买点3 破线拉回 : 盘中跌破 MA20 + 收盘重新站上 MA20 + 振幅放大
"""

import os
import sys
import json
import time
import random
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import akshare as ak

# ---------------------------------------------------------------- 配置
MIN_FLOAT_MV = 30_0000_0000      # 流通市值下限 30亿
MAX_FLOAT_MV = 800_0000_0000     # 流通市值上限 800亿
HIST_DAYS = 90                   # 拉取历史天数
CHART_DAYS = 60                  # 绘图展示天数
MAX_WORKERS = 8                  # 并发拉历史数据的线程数
# 每类买点在报告中展示的股票数：买点1/2 各 15 只，买点3 只展示最强的 3 只
# （买点3 是破线拉回，属于趋势后段、假信号最多，故收紧展示名额）
TOP_N_BY_BP = {1: 15, 2: 15, 3: 15}
TOP_SAVE = 40                    # 每类买点写入 JSON 的股票数（便于事后核对完整命中）
MAX_CODES = int(os.environ.get("MAX_CODES", "0"))   # >0 时只处理前 N 只，用于本地调试
# 亏损股处理（方案B·分层）：又亏又贵的一刀剔除，跌透的只扣分不剥夺资格
LOSS_PB_LIMIT = 5.0             # PE<=0 且 PB>此值 -> 硬剔除
LOSS_PENALTY = 5                # PE<=0 但 PB<=此值 -> 保留，仅在总分里扣分
# 观察池（L2）：进过榜的票会被记住，回调后即使排名掉出名额也单独回访
WATCH_DAYS = 20                 # 观察期（交易日）
WATCH_EXPIRE_NATURAL = 32       # 约等于 20 个交易日的自然日上限（不依赖交易日历的兜底）
WATCH_DROP_PCT = -10.0          # 相对入选价跌破此幅度 -> 移出（逻辑已坏）
MAX_REVISIT = 6                 # 回访区最多展示只数
ALT_N = 12                      # 备选区：各买点第 16 ~ 15+ALT_N 名，简表展示
STATE_DIR = "state"             # 观察池状态目录（需随仓库提交，才能在云端跨天保留）
# 粗筛后送入日线计算的股票上限。必须按「成交额」降序取头部，
# 绝不能按代码顺序截断——否则只会扫到沪市 600 开头的一小段，结果严重失真。
POOL_SIZE = int(os.environ.get("POOL_SIZE", "1200"))
OUT_DIR = "output"


def today_str():
    return datetime.date.today().strftime("%Y%m%d")


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 交易日判断
def resolve_date():
    """返回 (yyyymmdd, yyyy-mm-dd)。

    今天是交易日就用今天；否则回退到最近一个已过去的交易日。
    FORCE_RUN 补跑时若继续用当天日期，报告会把周一的票标成周六的数据。
    """
    t = datetime.date.today()
    fallback = (t.strftime("%Y%m%d"), t.strftime("%Y-%m-%d"))
    try:
        df = ak.tool_trade_date_hist_sina()
        days = sorted(str(d).replace("-", "") for d in df["trade_date"])
        if today_str() in days:
            return fallback
        past = [d for d in days if d <= today_str()]
        if past:
            d = past[-1]
            return d, f"{d[:4]}-{d[4:6]}-{d[6:]}"
    except Exception as e:
        log(f"交易日历不可用，按当日处理: {e}")
    return fallback


def is_trading_day():
    """返回 True/False；无法判断时返回 None（按交易日处理并警告）"""
    try:
        df = ak.tool_trade_date_hist_sina()
        days = {str(d).replace("-", "") for d in df["trade_date"]}
        return today_str() in days
    except Exception as e:
        log(f"交易日历获取失败: {e}")
        return None


# ---------------------------------------------------------------- 全市场快照
SPOT_COL_MAP = {
    "代码": "code", "名称": "name", "最新价": "close", "涨跌幅": "pct",
    "成交量": "volume", "成交额": "amount", "振幅": "amp", "换手率": "turn",
    "量比": "vr", "市盈率-动态": "pe", "市净率": "pb",
    "流通市值": "float_mv", "总市值": "total_mv",
}
# 新浪快照字段较少（无市值/换手/PE），作为东财不可用时的降级源
SINA_COL_MAP = {
    "代码": "code", "名称": "name", "最新价": "close", "涨跌幅": "pct",
    "成交量": "volume", "成交额": "amount", "最高": "high", "最低": "low", "昨收": "preclose",
}
NEED_COLS = ["code", "name", "close", "pct", "volume", "amount", "amp",
             "turn", "vr", "pe", "pb", "float_mv"]
# code / name 是字符串，绝不能做数值转换，否则会被置为 NaN
NUM_COLS = ["close", "pct", "volume", "amount", "amp", "turn",
            "vr", "pe", "pb", "float_mv"]


def _normalize(df, colmap):
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.extract(r"(\d{6})")[0]
    if "name" in df.columns:
        df["name"] = df["name"].astype(str)
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


TX_BATCH = 100


def _tx_symbol(code):
    """6 位代码 -> 腾讯带市场前缀的代码"""
    if code.startswith(("60", "68", "9")):
        return "sh" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sz" + code


def gen_all_codes():
    """穷举 A 股可能的代码段，供腾讯批量行情兜底使用"""
    segs = [(600000, 601999), (603000, 603999), (605000, 605999),
            (688000, 688999), (689000, 689099), (1, 1999),
            (3000, 3999), (300000, 301999), (302000, 302999)]
    codes = []
    for lo, hi in segs:
        codes += [str(c).zfill(6) for c in range(lo, hi + 1)]
    return codes


def tx_batch_quotes(codes):
    """腾讯批量实时行情：一次 100 只，字段含 PE/PB/换手率/市值/量比/振幅"""
    import urllib.request
    chunks = [codes[i:i + TX_BATCH] for i in range(0, len(codes), TX_BATCH)]
    rows = []

    def _get(chunk):
        url = "https://qt.gtimg.cn/q=" + ",".join(_tx_symbol(c) for c in chunk)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("gbk", errors="ignore")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for raw in ex.map(_get, chunks):
            for line in raw.split(";"):
                line = line.strip()
                if not line.startswith("v_"):
                    continue
                try:
                    f = line.split('="', 1)[1].rstrip('"').split("~")
                    if len(f) < 50 or not f[2]:
                        continue
                    rows.append({"code": f[2], "name": f[1], "close": f[3],
                                 "preclose": f[4], "open": f[5], "volume": f[6],
                                 "pct": f[32], "high": f[33], "low": f[34],
                                 "amount": f[37], "turn": f[38], "pe": f[39],
                                 "amp": f[43], "float_mv": f[44], "total_mv": f[45],
                                 "pb": f[46], "vr": f[49]})
                except Exception:
                    continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["name"] = df["name"].astype(str)
    for c in NUM_COLS + ["total_mv"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["float_mv"] = df["float_mv"] * 1e8     # 亿元 -> 元，与东财口径统一
    df["total_mv"] = df["total_mv"] * 1e8
    df["amount"] = df["amount"] * 1e4         # 万元 -> 元
    df["volume"] = df["volume"] * 100         # 手 -> 股
    return df


def load_spot():
    """全市场快照：东财优先，失败时自动改用腾讯批量行情"""
    try:
        df = _normalize(ak.stock_zh_a_spot_em(), SPOT_COL_MAP)
        df["_src"] = "em"
        log("快照源：东方财富")
        return df
    except Exception as e:
        log(f"东财快照不可用（{type(e).__name__}），改用腾讯批量行情")
    codes = gen_all_codes()
    df = tx_batch_quotes(codes)
    if df is None or df.empty:
        raise RuntimeError("所有快照源均不可用，任务终止")
    df["_src"] = "tx"
    log(f"快照源：腾讯（穷举 {len(codes)} 个代码，命中 {len(df)} 只）")
    return df


def prefilter(df, pool_size=None):
    """粗筛：把 5000+ 缩到千只以内，减少后续历史数据请求。

    关键：截断必须按成交额降序（覆盖全市场最活跃的股票）。
    若按代码顺序截断，只会扫到 600xxx 开头的一小段沪市主板，结果会严重失真。
    """
    if pool_size is None:
        pool_size = POOL_SIZE
    src = df["_src"].iloc[0] if "_src" in df.columns else "em"
    d = df.dropna(subset=["code", "name", "close"]).copy()
    # 只保留沪深主板/中小/创业/科创，排除北交所(4xx/8xx)
    d = d[~d["code"].astype(str).str.startswith(("4", "8", "9"))]
    # 剔除 ST / 退市
    d = d[~d["name"].astype(str).str.upper().str.contains("ST|退", na=False)]
    d = d[d["close"] > 2.0]
    if src in ("em", "tx"):
        d = d[(d["float_mv"] >= MIN_FLOAT_MV) & (d["float_mv"] <= MAX_FLOAT_MV)]
        d = d[(d["turn"] >= 1.0) & (d["turn"] <= 25.0)]
        d = d[d["vr"] >= 0.7]
        d = d[d["pct"] >= -6.0]
    # 统一按成交额降序保留头部：保证覆盖各板块的活跃股，而不是某一代码段
    if pool_size > 0 and len(d) > pool_size:
        before = len(d)
        d = d.sort_values("amount", ascending=False).head(pool_size)
        log(f"按成交额降序取头部 {pool_size} 只（粗筛后 {before} 只）")
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- 技术指标
def calc_indicators(df):
    """输入含 open/high/low/close/volume 的日线，追加全部指标"""
    d = df.copy()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    for n in (5, 10, 20, 60):
        d[f"ma{n}"] = c.rolling(n).mean()
    # MACD
    d["dif"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["dea"] = d["dif"].ewm(span=9, adjust=False).mean()
    d["macd_bar"] = 2 * (d["dif"] - d["dea"])
    # KDJ
    llv, hhv = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - llv) / (hhv - llv).replace(0, np.nan) * 100
    d["k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["d"] = d["k"].ewm(alpha=1 / 3, adjust=False).mean()
    d["j"] = 3 * d["k"] - 2 * d["d"]
    # BOLL
    std20 = c.rolling(20).std()
    d["boll_mid"] = d["ma20"]
    d["boll_up"] = d["ma20"] + 2 * std20
    d["boll_low"] = d["ma20"] - 2 * std20
    # 量能
    d["vma5"] = v.rolling(5).mean()
    # 动量
    d["ret20"] = (c / c.shift(20) - 1) * 100
    d["ret60"] = (c / c.shift(60) - 1) * 100
    return d


# ---------------------------------------------------------------- 历史数据
ACTIVE_SRC = None   # 探测后固定的日线源，避免每只股票都试错


def _fetch_one(code, src, start, end):
    if src == "tx":
        df = ak.stock_zh_a_hist_tx(symbol=_tx_symbol(code), start_date=start,
                                   end_date=end, adjust="qfq")
        ren = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
               "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turn"}
    else:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start,
                                end_date=end, adjust="qfq")
        ren = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
               "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turn"}
    df = df.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def detect_source(start, end):
    """用一只票探测哪个日线源可用，全局只做一次"""
    for src in ("tx", "em"):
        try:
            df = _fetch_one("600519", src, start, end)
            if df is not None and len(df) >= 25:
                log(f"日线源：{'腾讯' if src == 'tx' else '东方财富'} 可用")
                return src
        except Exception as e:
            log(f"日线源 {src} 不可用（{type(e).__name__}）")
    return "tx"


def fetch_hist(code, start, end):
    """带重试的单只日线拉取"""
    srcs = [ACTIVE_SRC] if ACTIVE_SRC else ["tx", "em"]
    for src in srcs:
        for attempt in range(2):
            try:
                df = _fetch_one(code, src, start, end)
                if df is not None and len(df) >= 25:
                    return code, df
                return code, None
            except Exception:
                time.sleep(0.3 * (attempt + 1) + random.random() * 0.3)
    return code, None


def fetch_all(codes, start, end):
    out, done = {}, 0
    total = len(codes)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_hist, c, start, end): c for c in codes}
        for f in as_completed(futs):
            code, df = f.result()
            done += 1
            if df is not None:
                out[code] = df
            if done % 200 == 0:
                log(f"  历史数据进度 {done}/{total}，成功 {len(out)}")
    return out


# ---------------------------------------------------------------- 三种买点
def judge_buypoint(d, spot):
    """d 为带指标的完整日线，spot 为当日快照。返回 (买点标记, 说明)"""
    if len(d) < 30:
        return None, "历史数据不足"
    cur, prev = d.iloc[-1], d.iloc[-2]
    if pd.isna(cur["ma20"]) or pd.isna(cur["ma60"]):
        return None, "均线未成形"

    close, low = cur["close"], cur["low"]
    ma20, vma5 = cur["ma20"], cur["vma5"]
    ret20 = cur.get("ret20", np.nan)
    pct = spot.get("pct", np.nan)
    turn = spot.get("turn", np.nan)
    amp = spot.get("amp", np.nan)

    def ok(x):
        return x is not None and not pd.isna(x)

    # --- 买点2：回踩不破（先判断，优先级高于买点1的模糊边界）
    if (ok(ret20) and ret20 > 10
            and low >= ma20 * 0.995
            and close > ma20
            and ok(vma5) and cur["volume"] < vma5 * 0.95
            and ok(pct) and -5.0 <= pct <= 2.0
            and ok(turn) and 2.0 <= turn <= 15.0):
        return 2, f"20日涨{ret20:.1f}%，回踩最低{low:.2f}未破MA20({ma20:.2f})，缩量收{pct:+.2f}%"

    # --- 买点3：破线拉回
    if (low < ma20 and close > ma20
            and ok(amp) and amp > 3.0
            and ok(turn) and 2.0 <= turn <= 15.0
            and ok(pct) and pct > 0):
        return 3, f"盘中最低{low:.2f}破MA20({ma20:.2f})，收盘{close:.2f}拉回站上，振幅{amp:.2f}%"

    # --- 买点1：突破转势
    prev20_high = d["high"].iloc[-21:-1].max() if len(d) >= 21 else np.nan
    breakout = (ok(cur["boll_up"]) and close > cur["boll_up"]) or \
               (ok(prev20_high) and close >= prev20_high)
    if (breakout
            and cur["ma5"] > cur["ma10"]
            and ma20 >= prev["ma20"]
            and cur["dif"] > cur["dea"]
            and ok(vma5) and cur["volume"] > vma5 * 1.5
            and ok(pct) and pct > 0):
        return 1, f"突破上轨/20日高，收盘{close:.2f}，放量{cur['volume']/vma5:.1f}倍，MACD转强"

    return None, "未触发"


# ---------------------------------------------------------------- 评分
PE_POOL = None   # 全市场有效 PE 的排序数组，run() 中构建，供分位打分使用


def pe_percentile(pe):
    """个股 PE 在全市场中的百分位（0-100，越小越便宜）。无法计算时返回 None。"""
    if PE_POOL is None or len(PE_POOL) < 200:
        return None
    if pe is None or pd.isna(pe) or pe <= 0:
        return None
    return round(float(np.searchsorted(PE_POOL, pe)) / len(PE_POOL) * 100, 1)


def pe_percentile_score(pe):
    """按全市场 PE 分位打分（0-6）。
    绝对值门槛会系统性歧视高 PE 成长股：全市场 PE 中位数约 40，
    而半导体/软件的行业中枢在 80-150，用 35 的线去卡等于整建制判死刑。
    改为相对分位后，科技股只需在自己的估值分布里不太极端即可。"""
    pct = pe_percentile(pe)
    if pct is None:
        # 分位池不可用（如离线测试）：降级用绝对门槛，但缩放到 6 分制
        if pe is None or pd.isna(pe) or pe <= 0:
            return 1
        return 6 if pe <= 35 else (4 if pe <= 60 else 2)
    if pct <= 20:
        return 6
    if pct <= 40:
        return 5
    if pct <= 60:
        return 3
    if pct <= 80:
        return 2
    return 1


def score(row, bp):
    """综合评分（0-100）。
    估值已从 20 分降权到 8 分并改用全市场 PE 分位；省下的 12 分补给
    连续型指标（乖离 6 + 量能配合 6），避免离散档位过多导致并列扎堆。"""
    s = 0.0
    # 技术面 40
    if row.get("ma5", np.nan) > row.get("ma10", np.nan):
        s += 10
    if row.get("ma10", np.nan) > row.get("ma20", np.nan):
        s += 10
    if row.get("ma20", np.nan) > row.get("ma60", np.nan):
        s += 10
    if row.get("dif", 0) > row.get("dea", 0):
        s += 10
    # 资金面 20
    vr = row.get("vr", np.nan)
    if not pd.isna(vr):
        s += 10 if vr >= 1.5 else (6 if vr >= 1.0 else 2)
    turn = row.get("turn", np.nan)
    if not pd.isna(turn):
        s += 10 if 3 <= turn <= 12 else (5 if turn < 3 else 3)
    # 动量 15
    r20 = row.get("ret20", np.nan)
    if not pd.isna(r20):
        s += 15 if 5 <= r20 <= 40 else (8 if r20 < 5 else 4)
    # 估值 8（PE 分位 6 + PB 2）——降权后不再决定成败，只做微调
    s += pe_percentile_score(row.get("pe", np.nan))
    pb = row.get("pb", np.nan)
    if not pd.isna(pb) and pb > 0:
        s += 2 if pb <= 3 else (1 if pb <= 6 else 0)
    # 乖离合理性 6（L1·按买点分档）
    # 旧逻辑一律「|bias| 越小越好」，会系统性误伤回踩到位的票：回踩必然压低
    # ret20/bias20，于是真正的好回踩排名反而输给正在冲高的票——与买点2 的本意相反。
    bias = row.get("bias20", np.nan)
    if not pd.isna(bias):
        bz = float(bias)
        if bp == 2:
            # 买点2 要的就是「贴着 MA20 缩量回踩」，0~3% 才是教科书位置
            if 0 <= bz <= 3:
                s += 6
            elif -5 <= bz <= 6:
                s += 4
            elif bz <= 10:
                s += 2
            else:
                s += 0      # 离 MA20 太远，那不是回踩是追高
        elif bp == 3:
            ab = abs(bz)
            s += 6 if ab <= 5 else (4 if ab <= 10 else (2 if ab <= 18 else 0))
        else:
            ab = abs(bz)    # 买点1 刚突破，正乖离本就正常，标准放宽
            s += 6 if ab <= 8 else (4 if ab <= 15 else (2 if ab <= 25 else 0))
    # 量能配合 6：买点1/3 要放量确认，买点2 要缩量回调
    vratio = row.get("vratio", np.nan)
    if vratio is not None and not pd.isna(vratio):
        r = float(vratio)
        if bp == 2:
            s += 6 if r <= 0.8 else (4 if r <= 1.0 else (2 if r <= 1.3 else 0))
        else:
            s += 6 if r >= 1.5 else (4 if r >= 1.0 else (2 if r >= 0.8 else 0))
    # 买点溢价 10（买点2 风险收益比最好）
    s += {1: 4, 2: 10, 3: 6}.get(bp, 0)
    # 亏损软惩罚（方案B）：PE<=0 但 PB 不高（跌透的周期底部股）保留资格、只扣分；
    # PE<=0 且 PB>LOSS_PB_LIMIT 属「又亏又贵」，已在 run() 阶段硬剔除，走不到这里。
    pe = row.get("pe", np.nan)
    if pe is not None and not pd.isna(pe) and pe <= 0:
        s -= LOSS_PENALTY
    return max(0, min(100, round(s)))


# ---------------------------------------------------------------- 观察池（L2）
# 系统原本是无状态每日快照：某票 D1 进买点1 榜单 -> 回调几天 -> 又满足买点2，
# 但排名掉出名额，它就在报告里彻底消失了。而「回踩到位」恰恰会压低动量与乖离分，
# 与「排进前 N 名」负相关——系统在结构上优先留下正在冲高的票，淘汰回调到位的票。
# 观察池让进过榜的票被记住：之后每天单独复算，即使不进名额也进「回访区」。
def watch_path():
    return os.path.join(STATE_DIR, "watchlist.json")


def load_watchlist():
    """{code: {name, first_date, first_bp, first_close, last_date}}"""
    p = watch_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception as e:
        log(f"观察池读取失败（{e}），按空池处理")
        return {}


def save_watchlist(wl):
    os.makedirs(STATE_DIR, exist_ok=True)
    p = watch_path()
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, p)      # 原子写，避免跑到一半被中断留下坏文件


def prune_watchlist(wl, day):
    """按观察期清理过期条目。破位/跌破止损的清理由 run() 依据当日行情处理。"""
    try:
        d1 = datetime.datetime.strptime(day, "%Y-%m-%d").date()
    except Exception:
        return []
    gone = []
    for code, m in list(wl.items()):
        try:
            d0 = datetime.datetime.strptime(m["first_date"], "%Y-%m-%d").date()
        except Exception:
            del wl[code]
            gone.append(code)
            continue
        if (d1 - d0).days > WATCH_EXPIRE_NATURAL:
            del wl[code]
            gone.append(code)
    if gone:
        log(f"观察池到期移出 {len(gone)} 只")
    return gone


# ---------------------------------------------------------------- 主流程
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=== 交易日检查 ===")
    td = is_trading_day()
    force = os.environ.get("FORCE_RUN", "0") == "1"
    if td is False and not force:
        log("今日非交易日，退出")
        push("【波段买点】今日休市", "今日为非交易日，无选股结果。")
        return 0
    if td is False and force:
        log("今日非交易日，但 FORCE_RUN=1，按上一交易日收盘数据继续")
    if td is None:
        log("无法确认交易日，继续按交易日处理")

    ymd, day = resolve_date()
    log(f"数据日期 {day}")

    log("=== 观察池 ===")
    wl = load_watchlist()
    if wl:
        prune_watchlist(wl, day)
    log(f"观察池 {len(wl)} 只（观察期约 {WATCH_DAYS} 个交易日 / {WATCH_EXPIRE_NATURAL} 自然日）")

    log("=== 拉取全市场快照 ===")
    spot = load_spot()
    log(f"全市场 {len(spot)} 只")
    cand = prefilter(spot)
    log(f"粗筛后 {len(cand)} 只")
    if cand.empty:
        log("警告：粗筛后无候选，疑似数据源字段异常，任务终止")
        push("【波段买点】数据异常", "粗筛后候选为空，请检查数据源与筛选条件。")
        return 1

    # 构建全市场 PE 分位池（估值改用分位制打分，避免用绝对门槛歧视高 PE 成长股）
    global PE_POOL
    _pe = pd.to_numeric(spot["pe"], errors="coerce")
    _pe = _pe[(_pe > 0) & (_pe < 3000)]
    PE_POOL = np.sort(_pe.values)
    if len(PE_POOL) >= 200:
        log(f"PE 分位池 {len(PE_POOL)} 只，全市场中位数 {np.median(PE_POOL):.1f}"
            f"（P25={np.percentile(PE_POOL,25):.1f} P75={np.percentile(PE_POOL,75):.1f}）")
    else:
        log(f"警告：PE 分位池仅 {len(PE_POOL)} 只，估值打分降级为绝对门槛")

    end = today_str()
    start = (datetime.date.today() - datetime.timedelta(days=HIST_DAYS * 2)).strftime("%Y%m%d")
    global ACTIVE_SRC
    ACTIVE_SRC = detect_source(start, end)
    codes = cand["code"].astype(str).str.zfill(6).tolist()
    # 观察池里已跌出当日粗筛池的票也要补拉，否则回访区会漏掉它们
    _in_pool = set(codes)
    wl_extra = [c for c in wl.keys() if c not in _in_pool]
    if wl_extra:
        log(f"观察池补拉 {len(wl_extra)} 只（已不在当日粗筛池内）")
        codes = codes + wl_extra
    if MAX_CODES > 0:
        codes = codes[:MAX_CODES]
        log(f"[调试模式] 仅处理前 {len(codes)} 只")
    log(f"=== 拉取 {len(codes)} 只历史日线 ===")
    hist = fetch_all(codes, start, end)
    log(f"成功 {len(hist)} 只")

    log("=== 计算指标并判定买点 ===")
    results = {1: [], 2: [], 3: []}
    excluded = []            # 方案B 硬剔除（又亏又贵）
    wl_today = {}            # 观察池个股当日快照，用于破位判定
    spot_idx = spot.set_index(spot["code"].astype(str).str.zfill(6))
    name_map = dict(zip(cand["code"].astype(str).str.zfill(6), cand["name"].astype(str)))
    ind_map = load_industry()

    def name_of(code):
        return name_map.get(code) or wl.get(code, {}).get("name") or code

    for code, df in hist.items():
        try:
            d = calc_indicators(df)
        except Exception:
            continue
        srow = spot_idx.loc[code] if code in spot_idx.index else {}
        srow = dict(srow)
        # 快照字段缺失时，用日线自己算当日涨跌幅/振幅/换手，保证口径一致
        cur0, prev0 = d.iloc[-1], d.iloc[-2]
        if pd.isna(srow.get("pct", np.nan)) and prev0["close"]:
            srow["pct"] = (cur0["close"] / prev0["close"] - 1) * 100
        if pd.isna(srow.get("amp", np.nan)) and prev0["close"]:
            srow["amp"] = (cur0["high"] - cur0["low"]) / prev0["close"] * 100
        if pd.isna(srow.get("turn", np.nan)) and not pd.isna(cur0.get("turn", np.nan)):
            srow["turn"] = float(cur0["turn"])
        if code in wl:
            # 无论今天有没有买点信号，都要留一份当日快照供破位判定
            _c = d.iloc[-1]
            wl_today[code] = {"close": float(_c["close"]),
                              "ma60": None if pd.isna(_c["ma60"]) else float(_c["ma60"])}
        bp, why = judge_buypoint(d, srow)
        if bp is None:
            continue
        # 方案B 硬剔除：又亏又贵（PE<=0 且 PB>5）——故事一旦证伪杀跌最狠
        _pe_v = srow.get("pe", np.nan)
        _pb_v = srow.get("pb", np.nan)
        if (not pd.isna(_pe_v)) and _pe_v <= 0 \
                and (not pd.isna(_pb_v)) and _pb_v > LOSS_PB_LIMIT:
            excluded.append({"code": code, "name": name_of(code), "bp": bp,
                             "pe": round(float(_pe_v), 1), "pb": round(float(_pb_v), 2)})
            continue
        cur = d.iloc[-1]
        rec = {
            "code": code,
            "name": name_of(code),
            "industry": ind_map.get(code, "—"),
            "bp": bp,
            "why": why,
            "close": round(float(cur["close"]), 2),
            "pct": None if pd.isna(srow.get("pct", np.nan)) else round(float(srow["pct"]), 2),
            "turn": None if pd.isna(srow.get("turn", np.nan)) else round(float(srow["turn"]), 2),
            "pe": None if pd.isna(srow.get("pe", np.nan)) else round(float(srow["pe"]), 1),
            "pb": None if pd.isna(srow.get("pb", np.nan)) else round(float(srow["pb"]), 2),
            "float_mv": None if pd.isna(srow.get("float_mv", np.nan)) else round(float(srow["float_mv"]) / 1e8, 1),
            "ma5": round(float(cur["ma5"]), 2),
            "ma10": round(float(cur["ma10"]), 2),
            "ma20": round(float(cur["ma20"]), 2),
            "ma60": round(float(cur["ma60"]), 2) if not pd.isna(cur["ma60"]) else None,
            "dif": round(float(cur["dif"]), 3) if not pd.isna(cur["dif"]) else None,
            "dea": round(float(cur["dea"]), 3) if not pd.isna(cur["dea"]) else None,
            "ret20": round(float(cur["ret20"]), 1) if not pd.isna(cur.get("ret20", np.nan)) else None,
            "vr": None if pd.isna(srow.get("vr", np.nan)) else round(float(srow["vr"]), 2),
            "bias20": round((float(cur["close"]) / float(cur["ma20"]) - 1) * 100, 1),
            "vratio": (round(float(cur["volume"] / cur["vma5"]), 2)
                       if (not pd.isna(cur["vma5"])) and cur["vma5"] else None),
            "pe_pct": pe_percentile(srow.get("pe", np.nan)),
            "chart": {
                "dates": d["date"].tolist()[-CHART_DAYS:],
                "ohlc": [[float(o), float(cl), float(lo), float(hi)]
                         for o, cl, lo, hi in zip(d["open"].tolist()[-CHART_DAYS:],
                                                  d["close"].tolist()[-CHART_DAYS:],
                                                  d["low"].tolist()[-CHART_DAYS:],
                                                  d["high"].tolist()[-CHART_DAYS:])],
                "vol": [float(x) for x in d["volume"].tolist()[-CHART_DAYS:]],
                "ma5": [None if pd.isna(x) else round(float(x), 2) for x in d["ma5"].tolist()[-CHART_DAYS:]],
                "ma10": [None if pd.isna(x) else round(float(x), 2) for x in d["ma10"].tolist()[-CHART_DAYS:]],
                "ma20": [None if pd.isna(x) else round(float(x), 2) for x in d["ma20"].tolist()[-CHART_DAYS:]],
            },
        }
        if code in wl:
            m = wl[code]
            fc = m.get("first_close")
            rec["revisit"] = {
                "first_date": m.get("first_date"),
                "first_bp": m.get("first_bp"),
                "first_close": fc,
                "chg": (round((float(cur["close"]) / fc - 1) * 100, 1) if fc else None),
            }
        else:
            rec["revisit"] = None
        rec["score"] = score(rec, bp)
        results[bp].append(rec)

    # 先记录真实命中总数，再截断展示——否则日志里的数字是截断后的，会误导
    # 排序必须完全确定：同分时若沿用并发返回顺序，每次跑出来的名单可能不同。
    # 同分裁决：① 分数高者优先 ② 乖离率绝对值小者优先（离 MA20 更近，风险更小）
    #           ③ 代码升序（保底，保证任何情况下结果可复现）
    def _rank_key(x):
        try:
            bias = abs(float(x.get("bias20") or 99))
        except (TypeError, ValueError):
            bias = 99.0
        return (-float(x.get("score") or 0), bias, str(x.get("code", "")))

    for k in results:
        results[k] = sorted(results[k], key=_rank_key)
    hits_all = {k: len(v) for k, v in results.items()}
    payload_top = {k: v[:TOP_N_BY_BP[k]] for k, v in results.items()}
    payload_save = {k: v[:TOP_SAVE] for k, v in results.items()}

    total = sum(len(v) for v in payload_top.values())
    shown = "/".join(str(TOP_N_BY_BP[k]) for k in (1, 2, 3))
    log(f"真实命中 买点1={hits_all[1]} 买点2={hits_all[2]} 买点3={hits_all[3]}"
        f"（展示名额 {shown}，各取评分前 {TOP_SAVE} 只存入 JSON）"
        f"；方案B 硬剔除 {len(excluded)} 只（亏损且 PB>{LOSS_PB_LIMIT}）")

    # ---- 回访区：今日有买点信号、在观察池里、但没挤进展示名额的票（L2）
    shown_codes = {r["code"] for v in payload_top.values() for r in v}
    revisit, _seen = [], set()
    for bp in (1, 2, 3):
        for r in results[bp]:
            c = r["code"]
            if c in wl and c not in shown_codes and c not in _seen:
                _seen.add(c)
                revisit.append(r)
    revisit.sort(key=_rank_key)
    revisit = revisit[:MAX_REVISIT]
    log(f"回访区 {len(revisit)} 只（观察池 {len(wl)} 只中今日再次出现买点信号者）")

    # ---- 破位移出：跌破 MA60，或跌破入选价 -10%（趋势已坏，不再跟踪）
    dropped = []
    for code, snap in wl_today.items():
        if code not in wl:
            continue
        fc = wl[code].get("first_close")
        if snap["ma60"] and snap["close"] < snap["ma60"]:
            dropped.append(code)
            del wl[code]
        elif fc and snap["close"] < fc * (1 + WATCH_DROP_PCT / 100.0):
            dropped.append(code)
            del wl[code]
    if dropped:
        log(f"观察池破位移出 {len(dropped)} 只")

    # ---- 今日进榜的写入观察池（已在池中的保留首次入选信息，只刷新最后出现日）
    added = 0
    for bp, v in payload_top.items():
        for r in v:
            c = r["code"]
            if c not in wl:
                wl[c] = {"name": r["name"], "first_date": day, "first_bp": int(bp),
                         "first_close": r["close"], "last_date": day}
                added += 1
            else:
                wl[c]["last_date"] = day
                wl[c]["name"] = r["name"]
    save_watchlist(wl)
    log(f"观察池新增 {added} 只，当前共 {len(wl)} 只 -> {STATE_DIR}/watchlist.json")

    payload = {"date": day, "total": total,
               "hits": {str(k): v for k, v in hits_all.items()},
               "results": {str(k): v for k, v in payload_save.items()},
               "revisit": revisit,
               "excluded": excluded,
               "watch_size": len(wl),
               "pool_size": len(cand)}
    with open(os.path.join(OUT_DIR, f"data_{ymd}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    html = build_html(payload)
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(OUT_DIR, f"report_{ymd}.html"), "w", encoding="utf-8") as f:
        f.write(html)
    log(f"报告已生成（数据日期 {day}）：{OUT_DIR}/index.html")

    push_summary(payload)
    return 0


# ---------------------------------------------------------------- HTML 报告
HTML_TPL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>波段买点 __DATE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
background:#f5f5f5;color:#1a1a1a;padding:16px;line-height:1.6}
.wrap{max-width:820px;margin:0 auto}
h1{font-size:20px;font-weight:600;margin-bottom:4px}
.meta{font-size:13px;color:#888;margin-bottom:20px}
.sec{font-size:16px;font-weight:600;margin:24px 0 12px;padding-left:10px;border-left:4px solid #185FA5}
.card{background:#fff;border-radius:12px;padding:16px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.hd{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.nm{font-size:16px;font-weight:600}
.cd{font-size:12px;color:#999}
.up{color:#d32f2f}.dn{color:#388e3c}
.sc{margin-left:auto;font-size:12px;background:#E6F1FB;color:#0C447C;padding:2px 10px;border-radius:10px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(88px,1fr));gap:8px;margin:10px 0;font-size:12px}
.kv .ind{grid-column:1/-1;background:#f5f8ff;border-radius:6px;padding:6px 8px;font-weight:600;color:#1a3a6b}
.kv div{background:#fafafa;border-radius:6px;padding:6px 8px}
.kv span{color:#999;display:block;font-size:11px}
.why{font-size:13px;background:#FFF9E6;border-left:3px solid #EF9F27;padding:8px 10px;border-radius:4px;margin:8px 0}
.rv{font-size:12px;background:#EAF3DE;border-left:3px solid #639922;padding:6px 10px;border-radius:4px;margin:8px 0}
.chart{width:100%;height:280px;margin-top:8px}
.empty{background:#fff;border-radius:12px;padding:20px;text-align:center;color:#999;font-size:14px}
.note{font-size:12px;color:#666;background:#fff;border-radius:10px;padding:10px 12px;margin-bottom:14px}
table.alt{width:100%;border-collapse:collapse;font-size:12px;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
table.alt th{background:#f0f0f0;padding:7px 8px;text-align:left;font-weight:600;white-space:nowrap}
table.alt td{padding:7px 8px;border-top:1px solid #f0f0f0;white-space:nowrap}
table.alt td.w{white-space:normal;color:#777;font-size:11px}
.foot{margin-top:28px;padding:14px;background:#FFF4F4;border-radius:10px;font-size:12px;color:#A32D2D}
</style></head><body><div class="wrap">
<h1>A股波段买点筛选</h1>
<div class="meta">__DATE__ 收盘后筛选 · 数据源 AkShare/东方财富 · 共命中 __TOTAL__ 只</div>
__BODY__
<div class="foot">本报告由程序自动抓取公开行情数据并按既定技术规则生成，仅供技术研究参考，不构成任何投资建议或个股推荐。股市有风险，投资需谨慎。</div>
</div><script>__JS__</script></body></html>"""

CARD_TPL = """<div class="card">
<div class="hd"><span class="nm">__NAME__</span><span class="cd">__CODE__</span>
<span class="__CLS__">__PRICE__ (__PCT__)</span><span class="sc">评分 __SCORE__</span></div>
__RV__
<div class="kv">
<div class="ind"><span>行业</span>__IND__</div>
<div><span>换手率</span>__TURN__%</div><div><span>PE(动)</span>__PE__</div>
<div><span>PE分位</span>__PEPCT__</div><div><span>PB</span>__PB__</div>
<div><span>流通市值</span>__MV__亿</div><div><span>MA5</span>__MA5__</div>
<div><span>MA20</span>__MA20__</div><div><span>MA60</span>__MA60__</div>
<div><span>20日涨幅</span>__R20__%</div><div><span>MA20乖离</span>__BIAS__%</div>
<div><span>量比</span>__VR__</div><div><span>量/5日均量</span>__VRATIO__</div>
</div><div class="why">__WHY__</div>
<div class="chart" id="c__IDX__"></div></div>"""


def fmt(v, suffix=""):
    return "—" if v is None else f"{v}{suffix}"


def build_html(payload):
    body_parts = []
    js_parts = []
    titles = {1: "买点1 · 突破转势（刚空转多）",
              2: "买点2 · 回踩不破（趋势中继）",
              3: "买点3 · 破线拉回（诱空修复）"}
    idx = 0

    def render_card(r):
        nonlocal idx
        cls = "dn" if (r["pct"] or 0) < 0 else "up"
        pct = "—" if r["pct"] is None else f"{r['pct']:+.2f}%"
        rv = r.get("revisit")
        rv_html = ""
        if rv:
            chg = rv.get("chg")
            rv_html = (f'<div class="rv">观察池回访 · {rv.get("first_date", "—")} 曾以'
                       f' 买点{rv.get("first_bp", "?")} 入选（{fmt(rv.get("first_close"))}），'
                       f'至今 {"—" if chg is None else f"{chg:+.1f}"}%，'
                       f'今日再现 买点{r.get("bp")}</div>')
        card = (CARD_TPL
                .replace("__NAME__", r["name"]).replace("__CODE__", r["code"])
                .replace("__CLS__", cls)
                .replace("__PRICE__", fmt(r["close"])).replace("__PCT__", pct)
                .replace("__SCORE__", str(r["score"]))
                .replace("__RV__", rv_html)
                .replace("__IND__", (r.get("industry") or "—"))
                .replace("__TURN__", fmt(r["turn"])).replace("__PE__", fmt(r["pe"]))
                .replace("__PB__", fmt(r["pb"])).replace("__MV__", fmt(r["float_mv"]))
                .replace("__MA5__", fmt(r["ma5"])).replace("__MA20__", fmt(r["ma20"]))
                .replace("__MA60__", fmt(r["ma60"])).replace("__R20__", fmt(r["ret20"]))
                .replace("__BIAS__", fmt(r["bias20"])).replace("__VR__", fmt(r["vr"]))
                .replace("__PEPCT__", ("—" if r.get("pe_pct") is None
                                       else f"P{r['pe_pct']:.0f}"))
                .replace("__VRATIO__", fmt(r.get("vratio")))
                .replace("__WHY__", r["why"]).replace("__IDX__", str(idx)))
        ch = r["chart"]
        js = ("echarts.init(document.getElementById('c%d')).setOption({"
              "animation:false,grid:[{left:44,right:14,top:16,height:150},"
              "{left:44,right:14,top:190,height:58}],"
              "xAxis:[{type:'category',data:%s,axisLabel:{fontSize:9}},"
              "{type:'category',gridIndex:1,data:%s,axisLabel:{show:false}}],"
              "yAxis:[{scale:true,axisLabel:{fontSize:9}},"
              "{gridIndex:1,scale:true,axisLabel:{show:false},splitLine:{show:false}}],"
              "dataZoom:[{type:'inside',xAxisIndex:[0,1],start:40,end:100}],"
              "series:[{type:'candlestick',data:%s,"
              "itemStyle:{color:'#d32f2f',color0:'#2e9e6b',borderColor:'#d32f2f',borderColor0:'#2e9e6b'}},"
              "{type:'line',data:%s,smooth:true,symbol:'none',lineStyle:{width:1},name:'MA5'},"
              "{type:'line',data:%s,smooth:true,symbol:'none',lineStyle:{width:1},name:'MA10'},"
              "{type:'line',data:%s,smooth:true,symbol:'none',lineStyle:{width:1.4},name:'MA20'},"
              "{type:'bar',xAxisIndex:1,yAxisIndex:1,data:%s,"
              "itemStyle:{color:'#c9c9c9'}}]});" % (
                  idx, json.dumps(ch["dates"]), json.dumps(ch["dates"]),
                  json.dumps(ch["ohlc"]), json.dumps(ch["ma5"]),
                  json.dumps(ch["ma10"]), json.dumps(ch["ma20"]),
                  json.dumps(ch["vol"])))
        idx += 1
        return card, js

    # 顶部说明：剔除统计 + 观察池状态
    exc = payload.get("excluded", [])
    notes = []
    if exc:
        names = "、".join(f'{e["name"]}(PE{e["pe"]}/PB{e["pb"]})' for e in exc[:8])
        more = f" 等 {len(exc)} 只" if len(exc) > 8 else ""
        notes.append(f"方案B 已剔除 {len(exc)} 只「亏损且 PB&gt;{LOSS_PB_LIMIT}」的标的："
                     f"{names}{more}；亏损但 PB 较低者保留，仅在评分中扣 {LOSS_PENALTY} 分。")
    notes.append(f"观察池现有 {payload.get('watch_size', 0)} 只：进过榜的票会持续跟踪约 "
                 f"{WATCH_DAYS} 个交易日，回调后即使掉出名额也会出现在「回访区」，"
                 f"跌破 MA60 或入选价 {WATCH_DROP_PCT}% 则移出。")
    body_parts.append(f'<div class="note">{"<br>".join(notes)}</div>')

    for bp in (1, 2, 3):
        full = payload["results"].get(str(bp), [])
        if bp == 3:
            # 买点3 全部用简表展示，不再单列带 K 线的卡片
            hits = payload.get("hits", {}).get(str(bp), len(full))
            body_parts.append(
                f'<div class="sec">{titles[bp]} · 共命中 {hits} 只，'
                f'以下 {min(len(full), TOP_N_BY_BP[3])} 只简表</div>')
            alt = full[:TOP_N_BY_BP[3]]
            if alt:
                rows = []
                for r in alt:
                    cls = "dn" if (r["pct"] or 0) < 0 else "up"
                    pct = "—" if r["pct"] is None else f"{r['pct']:+.2f}%"
                    rows.append(
                        f'<tr><td>{r["code"]}</td><td>{r["name"]}</td>'
                        f'<td class="{cls}">{fmt(r["close"])}（{pct}）</td>'
                        f'<td>{r["score"]}</td><td>{fmt(r["pe"])}</td>'
                        f'<td>{fmt(r["bias20"])}%</td>'
                        f'<td>{r.get("industry", "—")}</td>'
                        f'<td class="w">{r["why"][:38]}</td></tr>')
                body_parts.append(
                    '<table class="alt"><tr><th>代码</th><th>名称</th>'
                    '<th>现价(涨跌)</th><th>评分</th><th>PE</th><th>乖离</th>'
                    '<th>行业</th><th>要点</th></tr>' + "".join(rows) + "</table>")
            else:
                body_parts.append('<div class="empty">今日无符合条件的标的</div>')
            continue
        lst = full[:TOP_N_BY_BP[bp]]
        hits = payload.get("hits", {}).get(str(bp), len(lst))
        tail = f" · 展示 {len(lst)} 只" + (f"（共命中 {hits} 只）" if hits > len(lst) else "")
        body_parts.append(f'<div class="sec">{titles[bp]}{tail}</div>')
        if not lst:
            body_parts.append('<div class="empty">今日无符合条件的标的</div>')
        else:
            for r in lst:
                card, js = render_card(r)
                body_parts.append(card)
                js_parts.append(js)
        # 备选区（L0）：名额之外的紧邻名次，简表列出，不占名额、不配 K 线
        alt = full[TOP_N_BY_BP[bp]: TOP_N_BY_BP[bp] + ALT_N]
        if alt:
            rows = []
            for r in alt:
                cls = "dn" if (r["pct"] or 0) < 0 else "up"
                pct = "—" if r["pct"] is None else f"{r['pct']:+.2f}%"
                rows.append(
                    f'<tr><td>{r["code"]}</td><td>{r["name"]}</td>'
                    f'<td class="{cls}">{fmt(r["close"])}（{pct}）</td>'
                    f'<td>{r["score"]}</td><td>{fmt(r["pe"])}</td>'
                    f'<td>{fmt(r["bias20"])}%</td>'
                    f'<td>{r.get("industry", "—")}</td>'
                    f'<td class="w">{r["why"][:38]}</td></tr>')
            body_parts.append(
                f'<div class="sec" style="font-size:13px;border-left-color:#bbb;'
                f'margin:14px 0 8px">备选 · 第 {TOP_N_BY_BP[bp]+1}—'
                f'{TOP_N_BY_BP[bp]+len(alt)} 名</div>'
                '<table class="alt"><tr><th>代码</th><th>名称</th><th>现价(涨跌)</th>'
                '<th>评分</th><th>PE</th><th>乖离</th><th>行业</th><th>要点</th></tr>'
                + "".join(rows) + "</table>")

    # 回访区（L2）
    rev = payload.get("revisit", [])
    body_parts.append(f'<div class="sec" style="border-left-color:#639922">'
                      f'回访区 · 观察池回来的票（不占名额，最多 {MAX_REVISIT} 只）</div>')
    if not rev:
        body_parts.append('<div class="empty">观察池中今日无再次出现买点信号的标的</div>')
    else:
        for r in rev:
            card, js = render_card(r)
            body_parts.append(card)
            js_parts.append(js)

    html = (HTML_TPL.replace("__DATE__", payload["date"])
            .replace("__TOTAL__", str(payload["total"]))
            .replace("__BODY__", "".join(body_parts))
            .replace("__JS__", "\n".join(js_parts)))
    return html


# ---------------------------------------------------------------- 微信推送
def push(title, content_md):
    key = os.environ.get("PUSH_KEY", "").strip()
    chan = os.environ.get("PUSH_CHANNEL", "serverchan").strip().lower()
    if not key:
        log("未配置 PUSH_KEY，跳过推送")
        return
    try:
        import urllib.request
        import urllib.parse
        if chan == "pushplus":
            url = "https://www.pushplus.plus/send"
            data = json.dumps({"token": key, "title": title,
                               "content": content_md, "template": "markdown"}).encode()
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
        else:
            url = f"https://sctapi.ftqq.com/{key}.send"
            data = urllib.parse.urlencode({"title": title, "desp": content_md}).encode()
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=25) as r:
            log(f"推送完成: {r.status}")
    except Exception as e:
        log(f"推送失败: {e}")


def push_summary(payload):
    base = os.environ.get("REPORT_URL", "").strip()
    lines = [f"**{payload['date']} 收盘筛选**", ""]
    names = {1: "买点1 突破转势", 2: "买点2 回踩不破", 3: "买点3 破线拉回"}
    any_hit = False
    for bp in (1, 2, 3):
        full = payload["results"].get(str(bp), [])
        hits = payload.get("hits", {}).get(str(bp), len(full))
        if bp == 3:
            if not full:
                continue
            any_hit = True
            n = min(len(full), TOP_N_BY_BP[3])
            lines.append(f"### {names[bp]}（共命中 {hits} 只，以下 {n} 只简表）")
            lines.append("| 代码 | 名称 | 现价 | 涨跌 | 评分 | 行业 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for r in full[:TOP_N_BY_BP[3]]:
                pct = "—" if r["pct"] is None else f"{r['pct']:+.2f}%"
                lines.append(f"| {r['code']} | {r['name']} | {r['close']} | {pct} | {r['score']} | {r.get('industry', '—')} |")
            lines.append("")
            continue
        lst = payload["results"].get(str(bp), [])[:TOP_N_BY_BP[bp]]
        hits = payload.get("hits", {}).get(str(bp), len(lst))
        if not lst:
            continue
        any_hit = True
        lines.append(f"### {names[bp]}（共命中 {hits} 只，展示前 {len(lst)}）")
        lines.append("| 代码 | 名称 | 现价 | 涨跌 | 评分 | 行业 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in lst:
            pct = "—" if r["pct"] is None else f"{r['pct']:+.2f}%"
            lines.append(f"| {r['code']} | {r['name']} | {r['close']} | {pct} | {r['score']} | {r.get('industry', '—')} |")
        lines.append("")
    if not any_hit:
        lines.append("今日三类买点均无符合条件的标的。")
        lines.append("")
    rev = payload.get("revisit", [])
    if rev:
        lines.append("### 回访区（观察池回来，不占名额）")
        lines.append("| 代码 | 名称 | 现价 | 涨跌 | 评分 | 首次入选 | 至今 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rev:
            pct = "—" if r["pct"] is None else f"{r['pct']:+.2f}%"
            rv = r.get("revisit") or {}
            chg = rv.get("chg")
            lines.append(
                f"| {r['code']} | {r['name']} | {r['close']} | {pct} | {r['score']} "
                f"| {rv.get('first_date', '—')} 买点{rv.get('first_bp', '?')} "
                f"| {'—' if chg is None else f'{chg:+.1f}%'} |")
        lines.append("")
    if base:
        lines.append(f"[点击查看完整报告与K线图]({base})")
    title = f"波段买点 {payload['date'][5:]} · 共{payload['total']}只"
    push(title, "\n".join(lines))


INDUSTRY_FILE = os.path.join(STATE_DIR, "industry_map.json")
# 实时抓取用的东财接口（push2delay 在墙外/云端通常可达；失败也不影响静态底表）
_INDUSTRY_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,m:1+t:33"
_INDUSTRY_HOST = "push2delay.eastmoney.com"
_INDUSTRY_UT = "fa5fd1943c7b386f172d6893dbfba10b"


def _fetch_industry_live():
    """实时拉全市场「代码 -> 所属行业」(东财 push2delay, f100=所属行业)。失败抛异常。"""
    import ssl
    import urllib.parse
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    m, pn, pz = {}, 1, 100   # push2delay 单页上限 100，逐页翻到底
    while True:
        params = {
            "pn": str(pn), "pz": str(pz), "po": "1", "np": "1", "ut": _INDUSTRY_UT,
            "fltt": "2", "invt": "2", "fid": "f3", "fs": _INDUSTRY_FS,
            "fields": "f12,f100",
        }
        url = "https://%s/api/qt/clist/get?%s" % (_INDUSTRY_HOST, urllib.parse.urlencode(params))
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        )
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            data = json.loads(r.read().decode("utf-8"))
        items = (data.get("data") or {}).get("diff") or []
        if not items:
            break
        for it in items:
            code = str(it.get("f12") or "").strip()
            ind = it.get("f100")
            if code:
                code = code.zfill(6)
                if isinstance(ind, str) and ind.strip() and ind.strip() != "-":
                    m[code] = ind.strip()
        if len(items) < pz:
            break
        pn += 1
    if not m:
        raise ValueError("实时行业数据为空")
    return m


def load_industry():
    """返回 {code(6位): 行业名}。

    设计：**静态底表优先**。
    随仓库提交的 state/industry_map.json 已覆盖全部沪深 A 股（由 build_industry_map.py
    用东财 push2delay 一次性生成），云端直接读取、无需联网，彻底避免行业显示 '-'。
    实时抓取仅作为「补充新上市股票」的可选增强：只向底表里追加缺失项，
    绝不覆盖/删除已有映射；联网失败时静默回退到静态底表。
    """
    base = {}
    if os.path.exists(INDUSTRY_FILE):
        try:
            with open(INDUSTRY_FILE, encoding="utf-8") as f:
                base = json.load(f) or {}
            log(f"行业底表已加载：{len(base)} 只")
        except Exception as e:
            log(f"行业底表读取失败：{e}")
    if base:
        # 可选：实时补充底表缺失项（仅追加，不覆盖）
        try:
            live = _fetch_industry_live()
            merged = dict(base)
            for k, v in live.items():
                if k not in merged:
                    merged[k] = v
            if len(merged) > len(base):
                try:
                    os.makedirs(STATE_DIR, exist_ok=True)
                    tmp = INDUSTRY_FILE + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(merged, f, ensure_ascii=False, sort_keys=True)
                    os.replace(tmp, INDUSTRY_FILE)
                    log(f"行业底表已用实时数据扩充至 {len(merged)} 只")
                except Exception as e:
                    log(f"行业底表写回失败（不影响使用）：{e}")
            return merged
        except Exception as e:
            log(f"行业实时补充失败，沿用静态底表：{e}")
            return base
    # 底表缺失：尝试实时拉全量兜底
    try:
        return _fetch_industry_live()
    except Exception as e:
        log(f"行业映射加载失败：{e}")
        return {}


if __name__ == "__main__":
    sys.exit(run())
