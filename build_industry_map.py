#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性生成全市场「代码 -> 所属行业」静态底表 state/industry_map.json
数据源：东方财富 push2delay 接口（f100=所属行业），本地/云端均可直连，无需 AKShare。
用法：python build_industry_map.py
"""
import json
import os
import ssl
import time
import urllib.parse
import urllib.request

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
OUT = os.path.join(STATE_DIR, "industry_map.json")

# 沪深主板/科创板/创业板 + 北交所
FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,m:1+t:33"
HOST = "push2delay.eastmoney.com"
UT = "fa5fd1943c7b386f172d6893dbfba10b"
FIELDS = "f12,f13,f14,f100"  # 代码,市场,名称,所属行业


def _fetch_page(pn, pz):
    params = {
        "pn": str(pn), "pz": str(pz), "po": "1", "np": "1", "ut": UT,
        "fltt": "2", "invt": "2", "fid": "f3", "fs": FS, "fields": FIELDS,
    }
    url = "https://%s/api/qt/clist/get?%s" % (HOST, urllib.parse.urlencode(params))
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    )
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
        data = json.loads(r.read().decode("utf-8"))
    return (data.get("data") or {}).get("diff") or []


def build():
    os.makedirs(STATE_DIR, exist_ok=True)
    m, pn, pz = {}, 1, 100  # push2delay 单页上限 100，逐页翻到底
    while True:
        items = _fetch_page(pn, pz)
        if not items:
            break
        for it in items:
            code = str(it.get("f12") or "").strip()
            ind = it.get("f100")
            if not code:
                continue
            code = code.zfill(6)
            if isinstance(ind, str) and ind.strip():
                m[code] = ind.strip()
        if len(items) < pz:  # 最后一页
            break
        pn += 1
        time.sleep(0.3)
    return m


if __name__ == "__main__":
    m = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, sort_keys=True)
    print("已写入 %s：共 %d 只股票含行业" % (OUT, len(m)))
