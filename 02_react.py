"""階段 2：ReAct (Reason + Act) 練習。

手刻 agent loop，不用框架。三個工具：
  search_shops(city, dish_keyword) — 依城市查店家
  is_open(shop_name, time)         — 查某店在某時間是否營業
  distance(point_a, point_b)       — 算兩點直線距離（公里），支援店名和地標名稱

測試題（PLAN.md 原題）：
  「現在晚上 11 點在逢甲，食尚玩家介紹過的宵夜哪家還開著而且最近？」

這題裡藏了一個跟階段 1 類似的陷阱，但這次是「事實陷阱」而不是「推理陷阱」：
  丸南生魚片以「半夜 12 點才開門」聞名（食尚玩家介紹的宵夜代表），
  但晚上 11 點還沒到午夜，它其實還沒開——如果 agent 只做關鍵字比對、
  沒有真的呼叫 is_open() 驗證，很容易直接答錯成丸南生魚片。
  正確答案要從真正開著的幾家店（官芝霖、旺伯臭豆腐、逢甲紅燒當歸鴨、米丹）
  裡，用 distance() 比較誰離逢甲夜市最近。

這是跟 CoT 本質不同的失敗模式：CoT 治不好這題，因為問題不是「推理路徑」，
是模型根本沒有「現在幾點、這家店在哪」這種即時資訊——這正是需要工具的原因。

執行前：
  pip install anthropic python-dotenv
  export ANTHROPIC_API_KEY=...   （或放進 .env）
執行：
  python3 02_react.py

驗收標準：
  [ ] 印出完整的 Thought/Action/Observation 軌跡，能指著軌跡解釋每一步
  [ ] 故意讓一個工具回傳錯誤（把某個工具呼叫裡的店名打錯），
      觀察 agent 會不會自我修正重查——可以另外寫一小段程式碼，
      直接呼叫 call_tool("is_open", {"shop_name": "旺伯臭都腐", "time": "23:00"})
      看回傳的錯誤訊息，再想辦法把這個錯誤情境接進主迴圈測試
  [ ] 能回答：ReAct 跟階段 1 的 CoT 差在哪？
      （提示：CoT 只能重新排列模型已經知道的資訊；ReAct 能讓模型取得新資訊）
  延伸：加一個 max_turns 上限，觀察 agent 卡住鬼打牆的樣子
"""

import difflib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

import anthropic

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
print(f"Using model: {MODEL}")

client = anthropic.Anthropic()

DATA = Path(__file__).parent / "data" / "shops.json"
ALL_SHOPS = json.loads(DATA.read_text(encoding="utf-8"))

# 「逢甲夜市」不是資料庫裡的一家店，是使用者站的地點——
# 給它一個座標，distance() 才能拿它當起點。
LANDMARKS = {
    "逢甲夜市": (24.1798, 120.6469),
}


# ---------------------------------------------------------------------------
# 工具實作（已經寫好，這階段的重點不是資料查詢，是後面的迴圈）
# ---------------------------------------------------------------------------

def _find_shop(name: str) -> dict:
    for s in ALL_SHOPS:
        if s["name"] == name:
            return s
    candidates = difflib.get_close_matches(name, [s["name"] for s in ALL_SHOPS], n=1, cutoff=0.4)
    hint = f"，你是不是要找「{candidates[0]}」？" if candidates else ""
    raise LookupError(f"找不到店名「{name}」{hint}")


def _find_point(name: str) -> tuple[float, float]:
    if name in LANDMARKS:
        return LANDMARKS[name]
    for s in ALL_SHOPS:
        if s["name"] == name:
            return s["lat"], s["lng"]
    all_names = list(LANDMARKS.keys()) + [s["name"] for s in ALL_SHOPS]
    candidates = difflib.get_close_matches(name, all_names, n=1, cutoff=0.4)
    hint = f"，你是不是要找「{candidates[0]}」？" if candidates else ""
    raise LookupError(f"找不到地點「{name}」{hint}")


def _parse_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def search_shops(city: str, dish_keyword: str = "") -> list[dict]:
    """依城市（和可選關鍵字）查店家清單。查不到就回傳空 list，不算錯誤。"""
    keyword = dish_keyword.strip()
    results = []
    for s in ALL_SHOPS:
        if s["city"] != city:
            continue
        if keyword and keyword not in s["name"] and keyword not in s["dish"] and keyword not in s["note"]:
            continue
        results.append({
            "name": s["name"],
            "district": s["district"],
            "dish": s["dish"],
            "hours": s["hours"],
            "price_range": s["price_range"],
            "note": s["note"],
        })
    return results


def is_open(shop_name: str, time: str) -> dict:
    """查某店在指定時間（HH:MM）是否營業。店名打錯會丟 LookupError。"""
    shop = _find_shop(shop_name)
    open_str, close_str = shop["hours"].split("-")
    open_m, close_m, t_m = _parse_minutes(open_str), _parse_minutes(close_str), _parse_minutes(time)
    if open_m <= close_m:
        is_open_now = open_m <= t_m <= close_m
    else:  # 跨午夜營業，例如 17:00-01:00
        is_open_now = t_m >= open_m or t_m <= close_m
    return {"shop": shop_name, "time": time, "hours": shop["hours"], "open": is_open_now}


def distance(point_a: str, point_b: str) -> dict:
    """算兩個地點（店名或地標名稱）之間的直線距離（公里）。地點打錯會丟 LookupError。"""
    lat1, lng1 = _find_point(point_a)
    lat2, lng2 = _find_point(point_b)
    km = _haversine_km(lat1, lng1, lat2, lng2)
    return {"point_a": point_a, "point_b": point_b, "distance_km": round(km, 2)}


TOOLS = [
    {
        "name": "search_shops",
        "description": "依城市（和可選的餐點/店名關鍵字）查詢食尚玩家介紹過的店家清單，回傳店名、餐點、營業時間、價位、地區、備註。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名稱，例如：台南、台中、高雄、嘉義"},
                "dish_keyword": {"type": "string", "description": "選填，篩選店名/餐點/備註中包含這個關鍵字的店家"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "is_open",
        "description": "查詢某家店在指定時間是否營業中。shop_name 必須是資料庫裡完全相符的店名（可先用 search_shops 取得正確店名）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "shop_name": {"type": "string", "description": "店名，須與資料庫中完全相符"},
                "time": {"type": "string", "description": "24 小時制時間，格式 HH:MM，例如 23:00"},
            },
            "required": ["shop_name", "time"],
        },
    },
    {
        "name": "distance",
        "description": "計算兩個地點之間的直線距離（公里）。地點可以是店名，也可以是地標名稱（例如「逢甲夜市」）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "point_a": {"type": "string"},
                "point_b": {"type": "string"},
            },
            "required": ["point_a", "point_b"],
        },
    },
]


def call_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """執行工具，回傳 (結果文字, 是否為錯誤)。錯誤時 is_error 要設 True 傳回給模型。"""
    try:
        if name == "search_shops":
            result = search_shops(**tool_input)
        elif name == "is_open":
            result = is_open(**tool_input)
        elif name == "distance":
            result = distance(**tool_input)
        else:
            return f"未知工具：{name}", True
        return json.dumps(result, ensure_ascii=False), False
    except LookupError as e:
        return str(e), True


# ---------------------------------------------------------------------------
# ReAct 迴圈（由你手刻——這是這階段真正的練習）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是台灣美食行程助手，只能透過工具取得店家資訊，不能憑空猜測或用記憶回答。
在每次呼叫工具之前，先用一句話說明你為什麼要呼叫它（這句話會被視為你的 Thought）。
拿到工具結果後，再判斷下一步：需要更多資訊就再呼叫工具，資訊夠了就直接回答。
如果多次查詢都得不到答案，誠實告訴使用者查不到，不要瞎猜。"""

TEST_QUESTION = "現在晚上 11 點在逢甲，食尚玩家介紹過的宵夜哪家還開著而且最近？"


def run_react_loop(question: str, max_turns: int = 8) -> None:
    messages = [{"role": "user", "content": question}]
    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages,
        )

        for block in response.content:
            if block.type == "text":
                print(f"[Thought] {block.text}")

        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text":
                    print(f"[Final Answer] {block.text}")
            return

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[Action] {block.name}({block.input})")
                result_str, is_error = call_tool(block.name, block.input)
                print(f"[Observation] {'⚠️ 錯誤：' if is_error else ''}{result_str}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                    "is_error": is_error,
                })
        messages.append({"role": "user", "content": tool_results})

    print("⚠️ 超過 max_turns，agent 可能卡住了")


def main() -> None:
    run_react_loop(TEST_QUESTION)


if __name__ == "__main__":
    main()
