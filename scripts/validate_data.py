#!/usr/bin/env python3
"""驗證 data/shops.json 與 data/docs/ 的資料品質。

用法: python3 scripts/validate_data.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOPS = ROOT / "data" / "shops.json"
DOCS = ROOT / "data" / "docs"

REQUIRED = ["name", "city", "district", "dish", "episode",
            "price_range", "hours", "lat", "lng", "note"]
HOURS_RE = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")
PRICE_RE = re.compile(r"^\d+-\d+$")

# 台灣本島大致範圍
LAT_RANGE = (21.5, 25.5)
LNG_RANGE = (119.5, 122.5)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    shops = json.loads(SHOPS.read_text(encoding="utf-8"))
    names = [s.get("name", "<無名>") for s in shops]

    for dup in {n for n in names if names.count(n) > 1}:
        errors.append(f"店名重複: {dup}")

    for shop in shops:
        name = shop.get("name", "<無名>")
        for key in REQUIRED:
            if key not in shop or shop[key] in ("", None):
                errors.append(f"{name}: 缺少欄位 {key}")
        if "hours" in shop and not HOURS_RE.match(str(shop["hours"])):
            errors.append(f"{name}: hours 格式錯誤 ({shop['hours']})，須為 HH:MM-HH:MM")
        if "price_range" in shop and not PRICE_RE.match(str(shop["price_range"])):
            errors.append(f"{name}: price_range 格式錯誤 ({shop['price_range']})，須為 低-高")
        lat, lng = shop.get("lat"), shop.get("lng")
        if isinstance(lat, (int, float)) and not LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
            errors.append(f"{name}: lat 超出台灣範圍 ({lat})")
        if isinstance(lng, (int, float)) and not LNG_RANGE[0] <= lng <= LNG_RANGE[1]:
            errors.append(f"{name}: lng 超出台灣範圍 ({lng})")
        if not shop.get("verified"):
            warnings.append(f"{name}: 出處未查證 (verified=false)，找到食尚玩家來源後補上 source 連結")

    doc_names = {p.stem for p in DOCS.glob("*.md")}
    for name in names:
        if name not in doc_names:
            warnings.append(f"{name}: 尚無 docs/{name}.md（RAG 階段需要）")
    for stem in doc_names - set(names):
        warnings.append(f"docs/{stem}.md 沒有對應的 shops.json 條目")

    for doc in DOCS.glob("*.md"):
        length = len(doc.read_text(encoding="utf-8"))
        if length < 100:
            warnings.append(f"docs/{doc.name}: 內容太短 ({length} 字)，建議 100-300 字的節目介紹")

    print(f"共 {len(shops)} 家店、{len(doc_names)} 篇文件")
    by_city: dict[str, int] = {}
    for s in shops:
        by_city[s.get("city", "?")] = by_city.get(s.get("city", "?"), 0) + 1
    print("城市分布:", ", ".join(f"{c} {n}家" for c, n in sorted(by_city.items())))

    if errors:
        print(f"\n❌ 錯誤 {len(errors)} 項:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print(f"\n⚠️  提醒 {len(warnings)} 項:")
        for w in warnings:
            print(f"  - {w}")
    if not errors and not warnings:
        print("\n✅ 全部通過")

    if len(shops) < 20:
        print(f"\n目標 20 家以上，還差 {20 - len(shops)} 家。")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
