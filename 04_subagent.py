"""階段 4：Subagent 練習——一個 orchestrator 帶多個專才。

概念：主 agent 把子任務委派給各自有獨立 context、獨立工具的子 agent，再彙整結果。
重點是「context 隔離」——子 agent 看不到彼此的對話，連工具集都分開。

架構：
  行程總監（orchestrator）—— 不掛工具，只負責委派、發現衝突、仲裁、彙整
  ├── 美食評論員 —— 只有 RAG 工具（階段 3），選店與推薦理由
  ├── 交通規劃師 —— 只有 distance / 動線工具（階段 2），排大眾運輸動線
  └── 預算控管   —— 只有 shops.json 價格工具，檢查總花費

任務（PLAN.md 原題）：
  「兩天一夜台中美食之旅，預算 3000，不騎車用大眾運輸」

刻意設計的衝突點：
  美食評論員看不到預算數字，很可能把河南夜宵生魚等招牌宵夜全排進去；
  預算控管會把住宿+大眾運輸先扣掉 2000，餐飲只剩 1000——
  若推薦名單含多家高價位、又涵蓋兩天六餐，就容易判定超標，
  orchestrator 必須仲裁（請評論員降級替換，或刪減二訪）。

執行前：
  pip install anthropic python-dotenv
  export ANTHROPIC_API_KEY=...   （或放進 .env）
執行：
  python3 04_subagent.py

驗收標準：
  [ ] 每個子 agent 的 system prompt 和工具集都不同，且互相看不到對方的 context
  [ ] 觀察一次「子 agent 結論衝突」（評論員推的店超出餐飲預算），orchestrator 怎麼仲裁
  [ ] 能回答：什麼時候該用 subagent、什麼時候一個 agent 掛多工具就好？
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

import anthropic

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
print(f"Using model: {MODEL}")

client = anthropic.Anthropic()

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "shops.json"
DOCS_DIR = ROOT / "data" / "docs"

ALL_SHOPS: list[dict] = json.loads(DATA.read_text(encoding="utf-8"))
TAICHUNG_SHOPS = [s for s in ALL_SHOPS if s["city"] == "台中"]#只選擇台中市區的店家
TAICHUNG_SHOP_NAMES = {s["name"] for s in TAICHUNG_SHOPS}

LANDMARKS: dict[str, tuple[float, float]] = {
    "台中車站": (24.1369, 120.6839),
    "逢甲夜市": (24.1798, 120.6469),
}

TASK = "兩天一夜台中美食之旅，預算 3000，不騎車用大眾運輸"
TOTAL_BUDGET = 3000
LODGING_TRANSIT_RESERVE = 2000  # 預算控管：兩天一夜食宿交通預留（不含機車）

# ---------------------------------------------------------------------------
# 階段 3 RAG（美食評論員專用——只索引台中店家文件）
# ---------------------------------------------------------------------------
# 標點符號
_PUNCTUATION = set("，。、（）「」〈〉！？：；\n\t ")

# 載入台中店家文件
def _load_taichung_docs() -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in sorted(DOCS_DIR.glob("*.md")):# 按順序讀取所有md文件
        if path.stem in TAICHUNG_SHOP_NAMES:
            docs[path.stem] = path.read_text(encoding="utf-8")
    return docs

# 分詞
def _tokenize(text: str) -> list[str]:
    chars = [c for c in text if c not in _PUNCTUATION]#去除標點符號
    if len(chars) < 2:
        return chars#如果字符串長度小於2，則直接返回字符串
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]#如果字符串長度大於2，則將字符串分成兩個字符的組合

# 建立TF-IDF索引
def _build_tfidf_index(docs: dict[str, str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    doc_tokens = {name: _tokenize(text) for name, text in docs.items()}
    doc_count = len(docs)
    df: dict[str, int] = {}
    for tokens in doc_tokens.values():
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    idf = {term: math.log(doc_count / count) + 1 for term, count in df.items()}
    vectors: dict[str, dict[str, float]] = {}
    for name, tokens in doc_tokens.items():
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        total = len(tokens)
        vectors[name] = {term: (count / total) * idf[term] for term, count in tf.items()}
    return idf, vectors

# 嵌入查詢
def _embed_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = _tokenize(query)#分詞
    tf: dict[str, int] = {}
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1#計算詞頻
    total = len(tokens) or 1#計算總詞數
    return {term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()}

# 餘弦相似度 用來計算兩個向量之間的相似度 越相似越接近1 越不相似越接近0
def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


TAICHUNG_DOCS = _load_taichung_docs()
TAICHUNG_IDF, TAICHUNG_DOC_VECTORS = _build_tfidf_index(TAICHUNG_DOCS)

# RAG 檢索：只在台中店家文件裡搜尋。
def retrieve_food_docs(query: str, k: int = 3) -> list[dict[str, Any]]:
    """RAG 檢索：只在台中店家文件裡搜尋。"""
    query_vec = _embed_query(query, TAICHUNG_IDF)
    scores = [
        (name, _cosine_similarity(query_vec, vec))
        for name, vec in TAICHUNG_DOC_VECTORS.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for name, score in scores[:k]:
        results.append({
            "shop": name,
            "score": round(score, 4),
            "excerpt": TAICHUNG_DOCS[name][:400],
        })
    return results


# ---------------------------------------------------------------------------
# 階段 2 交通工具（交通規劃師專用）
# ---------------------------------------------------------------------------

def _find_shop(name: str) -> dict:
    for s in ALL_SHOPS:
        if s["name"] == name:
            return s
    candidates = difflib.get_close_matches(name, [s["name"] for s in ALL_SHOPS], n=1, cutoff=0.4)
    hint = f"，你是不是要找「{candidates[0]}」？" if candidates else ""
    raise LookupError(f"找不到店名「{name}」{hint}")

# 找到店家的經緯度
def _find_point(name: str) -> tuple[float, float]:
    if name in LANDMARKS:
        return LANDMARKS[name]
    shop = _find_shop(name)
    return shop["lat"], shop["lng"]

# 計算兩地直線距離（公里）
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)#計算兩地經度差
    dlambda = math.radians(lng2 - lng1)#計算兩地緯度差
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2#計算兩地距離
    return 2 * r * math.asin(math.sqrt(a))


def distance(point_a: str, point_b: str) -> dict:
    lat1, lng1 = _find_point(point_a)
    lat2, lng2 = _find_point(point_b)
    km = _haversine_km(lat1, lng1, lat2, lng2)
    return {"point_a": point_a, "point_b": point_b, "distance_km": round(km, 2)}

# 粗估大眾運輸時間（練習用啟發式，非真實時刻表）。
def estimate_transit(point_a: str, point_b: str) -> dict:
    """粗估大眾運輸時間（練習用啟發式，非真實時刻表）。"""
    km = distance(point_a, point_b)["distance_km"]
    if km <= 1.2:#如果距離小於1.2公里，則為步行
        mode = "步行"
        minutes = max(10, int(km * 12))
    else:#如果距離大於1.2公里，則為公車或捷運轉乘
        mode = "公車或捷運轉乘"
        minutes = max(15, int(km * 5 + 10))#計算大眾運輸時間
    return {
        "from": point_a,
        "to": point_b,
        "distance_km": km,
        "mode": mode,
        "estimated_minutes": minutes,
        "note": "不騎車；短距離步行，較長搭大眾運輸",
    }

# 貪婪最近鄰：從起點出發，依序走訪店家（大眾運輸動線規劃用）。
def plan_route(shop_names: list[str], start: str = "台中車站") -> dict:
    """貪婪最近鄰：從起點出發，依序走訪店家（大眾運輸動線規劃用）。"""
    remaining = [s for s in shop_names if s in TAICHUNG_SHOP_NAMES]#只選擇台中市區的店家
    if not remaining:
        return {"error": "沒有有效的台中店名", "valid_shops": sorted(TAICHUNG_SHOP_NAMES)}

    route = [start]
    segments: list[dict] = []
    total_km = 0.0
    current = start

    while remaining:
        #計算當前位置到剩餘店家的距離 並選擇距離最短的店家
        nearest = min(
            remaining,
            key=lambda s: _haversine_km(*_find_point(current), *_find_point(s)),
        )
        seg = distance(current, nearest)#計算當前位置到選擇的店家的距離
        transit = estimate_transit(current, nearest)#計算當前位置到選擇的店家的大眾運輸時間
        segments.append(transit)#將大眾運輸時間加入segments
        total_km += seg["distance_km"]
        route.append(nearest)#將選擇的店家加入route
        current = nearest#將當前位置更新為選擇的店家
        remaining.remove(nearest)#將選擇的店家從remaining中移除

    return {
        "start": start,
        "route": route,
        "segments": segments,
        "total_distance_km": round(total_km, 2),
        "constraint": "不騎車，大眾運輸或步行",
    }


# ---------------------------------------------------------------------------
# 預算工具（預算控管專用）
# ---------------------------------------------------------------------------

def _price_bounds(price_range: str) -> tuple[int, int]:
    low, high = price_range.split("-")
    return int(low), int(high)


def get_taichung_price_table() -> list[dict]:
    """回傳台中店家的價位表（預算控管唯一資料來源）。"""
    return [
        {
            "name": s["name"],
            "dish": s["dish"],
            "price_range": s["price_range"],
            "price_low": _price_bounds(s["price_range"])[0],
            "price_high": _price_bounds(s["price_range"])[1],
        }
        for s in TAICHUNG_SHOPS
    ]


def estimate_food_cost(shop_names: list[str], days: int = 2, meals_per_day: int = 3) -> dict:
    """估算餐飲花費：兩天三餐共 6 個用餐時段，以各店價格上限計（含二訪）。"""
    valid = []
    for name in shop_names:
        try:
            valid.append(_find_shop(name))
        except LookupError:
            continue
    if not valid:
        return {"error": "沒有有效店名", "valid_shops": sorted(TAICHUNG_SHOP_NAMES)}

    meal_slots = days * meals_per_day
    assignments: list[dict] = []
    total = 0
    for i in range(meal_slots):
        shop = valid[i % len(valid)]
        _, high = _price_bounds(shop["price_range"])
        assignments.append({"meal": i + 1, "shop": shop["name"], "cost": high})
        total += high

    return {
        "days": days,
        "meals_per_day": meals_per_day,
        "meal_slots": meal_slots,
        "assignments": assignments,
        "estimated_food_cost": total,
    }


def check_budget(shop_names: list[str], total_budget: int = TOTAL_BUDGET) -> dict:
    """檢查總預算：先扣住宿+大眾運輸預留，再看餐飲是否超標。"""
    food = estimate_food_cost(shop_names)
    if "error" in food:
        return food

    food_cost = food["estimated_food_cost"]
    food_budget = total_budget - LODGING_TRANSIT_RESERVE
    over_food = food_cost > food_budget
    total_estimated = LODGING_TRANSIT_RESERVE + food_cost

    return {
        "total_budget": total_budget,
        "lodging_transit_reserve": LODGING_TRANSIT_RESERVE,
        "food_budget_remaining": food_budget,
        "estimated_food_cost": food_cost,
        "estimated_total_cost": total_estimated,
        "within_budget": not over_food and total_estimated <= total_budget,
        "over_by": max(0, total_estimated - total_budget),
        "food_over_by": max(0, food_cost - food_budget),
        "shop_names": shop_names,
        "detail": food["assignments"],
    }


# ---------------------------------------------------------------------------
# 通用 ReAct 子 agent 迴圈（每個子 agent 有獨立的 messages，互不相通）
# ---------------------------------------------------------------------------

ToolHandler = Callable[[dict], Any]


def _call_handler(handlers: dict[str, ToolHandler], name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        if name not in handlers:
            return f"未知工具：{name}", True
        result = handlers[name](**tool_input)
        return json.dumps(result, ensure_ascii=False), False
    except (LookupError, ValueError, TypeError) as e:
        return str(e), True

def run_subagent(
    agent_name: str,
    system_prompt: str,
    tools: list[dict],
    handlers: dict[str, ToolHandler],
    user_message: str,
    max_turns: int = 8,
) -> str:
    """執行一個子 agent 的 ReAct 迴圈（Thought → Action → Observation → … → 最終報告）。

    每個子 agent 在這裡有**獨立的 messages**，函式結束後對話不會留給下一個 agent——
    這是階段 4 context 隔離的關鍵。Orchestrator 只拿本函式的**回傳字串**（最終報告），
    不應把內部的 tool 軌跡傳給其他子 agent。

    參數：
        agent_name:    顯示用名稱，會印在 [美食評論員 Thought] 等 log 前綴。
        system_prompt: 子 agent 的角色與限制（例如「不能用價格工具」）。
        tools:         傳給 Claude API 的工具 schema 清單（Anthropic tools 格式）。
        handlers:      工具名稱 → Python 函式的對照表；Action 時由 _call_handler 派發。
        user_message:  這個子 agent 收到的任務（通常由 orchestrator 組好再傳入）。
        max_turns:     ReAct 迴圈上限；每輪最多一次 API 呼叫，超過則回傳警告字串。

    回傳：
        模型不再呼叫工具時的最終文字報告；若超過 max_turns 則回傳警告訊息。

    用法範例（見 run_food_critic / run_budget_check）：

        report = run_subagent(
            "美食評論員",
            FOOD_CRITIC_SYSTEM,
            FOOD_CRITIC_TOOLS,
            FOOD_CRITIC_HANDLERS,
            "兩天一夜台中美食之旅，請推薦 5 家店",
        )
        # report 是 str，給 orchestrator 抽店名或傳給下一個子 agent 的摘要

    與階段 2 run_react_loop 的差別：
        - 每次呼叫都新建 messages，專屬一個子 agent。
        - system_prompt、tools、handlers 由呼叫端注入，可組出不同專才。
    """
    print(f"\n{'=' * 60}")
    print(f"[{agent_name}] 開始（獨立 context，看不到其他子 agent）")
    print(f"{'=' * 60}")

    messages: list[dict] = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        # 印出子 agent 的思考過程
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[{agent_name} Thought] {block.text.strip()}")
        # 如果子 agent 不需要使用工具，則直接返回最終報告
        if response.stop_reason != "tool_use":
            final = "".join(b.text for b in response.content if b.type == "text")
            print(f"\n[{agent_name} 最終報告]\n{final}\n")
            return final
        # 如果子 agent 需要使用工具，則將工具的結果加入messages
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            # 如果子 agent 不需要使用工具，則跳過
            if block.type != "tool_use":
                continue
            # 印出子 agent 使用的工具及其參數
            print(f"[{agent_name} Action] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
            result_str, is_error = _call_handler(handlers, block.name, block.input)
            tag = "⚠️ 錯誤：" if is_error else ""#如果工具使用錯誤，則標記為錯誤
            print(f"[{agent_name} Observation] {tag}{result_str}")#印出工具使用的結果
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,#工具使用的結果
                "is_error": is_error,#工具使用是否錯誤
            })
        messages.append({"role": "user", "content": tool_results})#將工具使用的結果加入messages

    msg = f"⚠️ {agent_name} 超過 max_turns，可能卡住"
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# 三個子 agent 的設定（不同 system prompt、不同工具集）
# ---------------------------------------------------------------------------

FOOD_CRITIC_SYSTEM = """你是「美食評論員」子 agent，只負責從資料集裡的台中美食中選店並說明理由。
你只能使用 retrieve_food_docs 檢索介紹文，不能使用價格、距離、營業時間等工具——那些是別人的工作。
你不知道總預算數字，請專注推薦最具代表性的必吃名店（含宵夜傳奇店），並說明為何值得推薦。
最後請列出 5 家店名（JSON 陣列格式 shop_names），並附每家一句推薦理由。"""

FOOD_CRITIC_TOOLS = [
    {
        "name": "retrieve_food_docs",
        "description": "依關鍵字從台中店家介紹文（RAG）檢索最相關的文件片段。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "檢索關鍵字或問題"},
                "k": {"type": "integer", "description": "回傳幾篇，預設 3"},
            },
            "required": ["query"],
        },
    },
]

FOOD_CRITIC_HANDLERS: dict[str, ToolHandler] = {
    "retrieve_food_docs": retrieve_food_docs,
}

TRANSPORT_SYSTEM = """你是「交通規劃師」子 agent，只負責動線與大眾運輸。
你只能使用 distance、estimate_transit、plan_route，不能查介紹文或價格。
使用者不騎車，請以台中車站為起點，規劃造訪店家的順序，並估算每段步行或公車時間。
最後輸出依序造訪的店名清單與交通摘要。"""

TRANSPORT_TOOLS = [
    {
        "name": "distance",
        "description": "計算兩地直線距離（公里），地點可以是店名或地標（台中車站、逢甲夜市）。",
        "input_schema": {
            "type": "object",
            "properties": {"point_a": {"type": "string"}, "point_b": {"type": "string"}},
            "required": ["point_a", "point_b"],
        },
    },
    {
        "name": "estimate_transit",
        "description": "粗估兩點之間以大眾運輸或步行所需的時間（不騎車）。",
        "input_schema": {
            "type": "object",
            "properties": {"point_a": {"type": "string"}, "point_b": {"type": "string"}},
            "required": ["point_a", "point_b"],
        },
    },
    {
        "name": "plan_route",
        "description": "依貪婪最近鄰排訪店順序，從起點出發走訪多家店。",
        "input_schema": {
            "type": "object",
            "properties": {
                "shop_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要造訪的店名清單",
                },
                "start": {"type": "string", "description": "起點，預設台中車站"},
            },
            "required": ["shop_names"],
        },
    },
]
# 交通規劃師的工具處理函式 對照表，Action 時由 _call_handler 派發。
TRANSPORT_HANDLERS: dict[str, ToolHandler] = {
    "distance": distance,
    "estimate_transit": estimate_transit,
    "plan_route": plan_route,
}
# 預算控管的系統提示詞
BUDGET_SYSTEM = f"""你是「預算控管」子 agent，只負責檢查花費。
你只能使用 get_taichung_price_table、estimate_food_cost、check_budget。
總預算 {TOTAL_BUDGET} 元中，住宿+大眾運輸預留 {LODGING_TRANSIT_RESERVE} 元，餐飲只剩
{TOTAL_BUDGET - LODGING_TRANSIT_RESERVE} 元。兩天三餐共 6 個用餐時段，以各店價格上限估算。
若超出預算，明確說明超多少、建議替換哪家最貴的店。"""

BUDGET_TOOLS = [
    {
        "name": "get_taichung_price_table",
        "description": "取得台中所有店家的價格區間（唯一權威價格來源）。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "estimate_food_cost",
        "description": "依店名清單估算兩天三餐的餐飲花費（以價格上限計，含二訪）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "shop_names": {"type": "array", "items": {"type": "string"}},
                "days": {"type": "integer"},
                "meals_per_day": {"type": "integer"},
            },
            "required": ["shop_names"],
        },
    },
    {
        "name": "check_budget",
        "description": "檢查給定店名清單是否在總預算內（含住宿交通預留）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "shop_names": {"type": "array", "items": {"type": "string"}},
                "total_budget": {"type": "integer"},
            },
            "required": ["shop_names"],
        },
    },
]

BUDGET_HANDLERS: dict[str, ToolHandler] = {
    "get_taichung_price_table": lambda **_: get_taichung_price_table(),
    "estimate_food_cost": estimate_food_cost,
    "check_budget": check_budget,
}


# ---------------------------------------------------------------------------
# Orchestrator：委派、衝突仲裁、彙整（不掛子 agent 的工具）
# ---------------------------------------------------------------------------
# 從子 agent 報告中抽出店名（依在文字中出現的順序）。
def extract_shop_names(text: str) -> list[str]:
    """從子 agent 報告中抽出店名（依在文字中出現的順序）。"""
    # regular expression 正規表示式 的意思是 用來搜尋字串的規則 這裡是搜尋 shop_names 的規則
    json_match = re.search(r"shop_names\s*[:：]\s*(\[[\s\S]*?\])", text)
    if json_match:
        try:
            arr = json.loads(json_match.group(1))
            if isinstance(arr, list):
                names = [x for x in arr if x in TAICHUNG_SHOP_NAMES]
                if names:
                    return names
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\[[\s\S]*?\]", text)
    if json_match:
        try:
            arr = json.loads(json_match.group())
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                names = [x for x in arr if x in TAICHUNG_SHOP_NAMES]
                if names:
                    return names
        except json.JSONDecodeError:
            pass

    positions = [(text.find(name), name) for name in TAICHUNG_SHOP_NAMES if name in text]
    positions.sort(key=lambda x: x[0])
    return [name for _, name in positions]


def run_food_critic(task: str, extra: str = "") -> str:
    prompt = task if not extra else f"{task}\n\n補充指示：{extra}"
    return run_subagent(
        "美食評論員",
        FOOD_CRITIC_SYSTEM,
        FOOD_CRITIC_TOOLS,
        FOOD_CRITIC_HANDLERS,
        prompt,
    )


def run_budget_check(shop_names: list[str]) -> str:
    summary = (
        f"請檢查以下推薦店名的兩天一夜預算（總預算 {TOTAL_BUDGET}）：\n"
        f"店名：{json.dumps(shop_names, ensure_ascii=False)}\n"
        "請呼叫 check_budget 並說明是否超標。"
    )
    return run_subagent(
        "預算控管",
        BUDGET_SYSTEM,
        BUDGET_TOOLS,
        BUDGET_HANDLERS,
        summary,
    )


def run_transport_planner(shop_names: list[str]) -> str:
    summary = (
        f"請為以下店家規劃大眾運輸動線（不騎車），從台中車站出發：\n"
        f"{json.dumps(shop_names, ensure_ascii=False)}"
    )
    return run_subagent(
        "交通規劃師",
        TRANSPORT_SYSTEM,
        TRANSPORT_TOOLS,
        TRANSPORT_HANDLERS,
        summary,
    )

# 行程總監最後彙整——只有各子 agent 的報告摘要，沒有他們的 tool 軌跡。
def synthesize_itinerary(
    task: str,
    food_report: str,
    budget_report: str,
    transport_report: str,
    arbitrated: bool,
) -> str:
    """行程總監最後彙整——只有各子 agent 的報告摘要，沒有他們的 tool 軌跡。"""
    prompt = f"""你是行程總監。以下是三位專家的報告（他們互相看不到對方的過程）：

【任務】{task}

【美食評論員】
{food_report}

【預算控管】
{budget_report}

【交通規劃師】
{transport_report}

【是否經過仲裁】{"是——評論員曾因超預算調整推薦" if arbitrated else "否"}

請輸出兩天一夜台中美食行程摘要：每日時段、店家、交通方式、預算結論。
若曾超預算，說明你怎麼仲裁（換了哪家、為什麼）。"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text")

# 行程總監：委派、衝突仲裁、彙整（不掛子 agent 的工具）
def run_orchestrator(task: str) -> None:
    print("\n" + "#" * 60)
    print("[行程總監] 開始委派（orchestrator 自己不掛工具）")
    print("#" * 60)
    print(f"任務：{task}\n")

    # ① 美食評論員（不知道預算）
    food_report = run_food_critic(task)
    shop_names = extract_shop_names(food_report)
    if not shop_names:
        shop_names = [s["name"] for s in TAICHUNG_SHOPS]
    print(f"[行程總監] 從評論員報告抽出店名：{shop_names}")

    # ② 預算控管（看不到評論員的 RAG 軌跡，只收到店名清單）
    budget_report = run_budget_check(shop_names)
    budget_result = check_budget(shop_names)
    arbitrated = False

    # ③ 衝突仲裁
    if not budget_result.get("within_budget", True):
        arbitrated = True #衝突仲裁
        over = budget_result.get("food_over_by") or budget_result.get("over_by", 0)
        print(f"\n{'!' * 60}")
        print(f"[行程總監] ⚠️ 偵測到衝突：餐飲預估超出 {over} 元")
        print(f"[行程總監] 啟動仲裁——請美食評論員在預算內調整（不換交通規劃師）")
        print(f"{'!' * 60}\n")

        arbitration = (
            f"預算控管回報：{budget_report}\n"
            f"餐飲預算只剩 {TOTAL_BUDGET - LODGING_TRANSIT_RESERVE} 元。"
            f"請保留在地特色，但替換或刪除最貴的選項（提示：河南夜宵生魚單價高），"
            f"讓 5 家店仍具代表性且總餐飲費不超標。"
        )
        food_report = run_food_critic(task, extra=arbitration)
        shop_names = extract_shop_names(food_report)
        budget_report = run_budget_check(shop_names)
        budget_result = check_budget(shop_names)
        print(f"[行程總監] 仲裁後店名：{shop_names}，within_budget={budget_result.get('within_budget')}")

    # ④ 交通規劃師（只收到最終店名，看不到評論員與預算 agent 的對話）
    transport_report = run_transport_planner(shop_names)

    # ⑤ 彙整
    print(f"\n{'=' * 60}")
    print("[行程總監] 彙整最終行程")
    print(f"{'=' * 60}")
    final = synthesize_itinerary(
        task, food_report, budget_report, transport_report, arbitrated
    )
    print(final)

    print(f"\n{'=' * 60}")
    print("驗收檢查（人工核對）")
    print(f"{'=' * 60}")
    print("[ ] 三位子 agent 的 system prompt 和工具集是否都不同？")
    print("[ ] 子 agent 的 Action/Observation 軌跡是否各自獨立、沒有混在同一個 messages？")
    print(f"[ ] 是否觀察到預算衝突並仲裁？（本次 arbitrated={arbitrated}）")
    print("[ ] 行程總監是否只收到各 agent 的「最終報告」，而非完整 tool 軌跡？")
    print("[ ] 能回答：何時用 subagent vs 單一 agent 掛多工具？")
    print("      提示：工具太多選錯、context 汙染、需要平行專家時 → subagent")


def main() -> None:
    run_orchestrator(TASK)


if __name__ == "__main__":
    main()
