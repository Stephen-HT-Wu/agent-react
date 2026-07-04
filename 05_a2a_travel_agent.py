"""階段 5：A2A（Agent-to-Agent）練習 —— 旅遊資訊 agent（服務端，已寫好）。

概念：Subagent（階段 4）是同一支程式裡的委派——orchestrator 用 Python 直接呼叫
run_subagent()，子 agent 沒有自己的網路位址。A2A 是兩個「各自能獨立部署」的
agent 服務，彼此不共用程式碼、不共用記憶體，只能透過 HTTP 溝通——就像兩家不同
公司的系統互相合作，你不會把對方的原始碼裝進自己的 repo，只能靠一份說明書
（agent card）加上一個固定的呼叫規格（這裡是自訂的 JSON-over-HTTP）。

這支檔案是「旅遊資訊 agent」：提供天氣、公車路線兩個能力，包成一個獨立的
HTTP 服務。這支檔案已經全部寫好——階段 5 真正的練習在 05_a2a_food_agent.py
那邊（食尚玩家美食 agent 要「動態」發現這支服務的能力，不能用寫死的 if/else）。

原本這裡是直接借用 ../open-play/taiwan-travel-ai 現成的 cwa.py/tdx.py 呼叫
真實的 CWA、TDX API——但這樣一來，agent-react 這個練習專案就跨專案依賴了
別人的 codebase 和 .env 憑證，不是真正「兩個獨立部署的服務」該有的樣子
（也違背這系列一路以來「不引入非必要依賴」的原則，見階段 3 用手刻 TF-IDF
取代 embedding API 的理由）。所以這裡改用 data/weather.json、
data/bus_routes.json 兩份手刻的模擬資料——這個練習的重點是 A2A 的協定機制
（agent card、任務派送），不是天氣/公車 API 整合本身，用假資料完全不影響
驗收標準要驗證的東西。

自訂協定（比照 PLAN.md：先手刻一版簡化的，再對照 Google A2A 規格看差在哪）：
  GET  /agent-card
    回傳 {"name", "description", "capabilities": [{"name", "description", "input_schema"}, ...]}
    input_schema 直接是 Claude tool 的 input_schema 格式，方便對方原封不動拿去當工具定義。
  POST /tasks
    請求 {"capability": "<capability 名稱>", "input": {...}}
    回應 {"status": "completed", "output": {...}} 或 {"status": "failed", "error": "..."}

執行：
  python3 05_a2a_travel_agent.py
  （另開一個 terminal）curl http://localhost:8500/agent-card
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8500

DATA_DIR = Path(__file__).parent / "data"
WEATHER_PATH = DATA_DIR / "weather.json"
BUS_ROUTES_PATH = DATA_DIR / "bus_routes.json"


def _load_mock_data() -> tuple[dict, dict]:
    """啟動時載入模擬資料，缺檔就給明確錯誤。"""
    missing = [p for p in (WEATHER_PATH, BUS_ROUTES_PATH) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"找不到模擬資料：{names}。請確認 data/weather.json 與 data/bus_routes.json 存在。"
        )
    weather = json.loads(WEATHER_PATH.read_text(encoding="utf-8"))
    bus_routes = json.loads(BUS_ROUTES_PATH.read_text(encoding="utf-8"))
    return weather, bus_routes


WEATHER, BUS_ROUTES = _load_mock_data()


# ---------------------------------------------------------------------------
# 能力實作：模擬資料查詢，統一回傳格式、統一錯誤處理
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    """查某縣市的天氣預報（模擬資料，不是即時 API）。查不到城市就丟例外。"""
    if city not in WEATHER:
        raise LookupError(f"沒有「{city}」的天氣資料，目前只支援：{list(WEATHER.keys())}")
    return WEATHER[city]


def search_bus_routes(city: str, keyword: str = "") -> dict:
    """查某縣市的市區公車路線（模擬資料，不是即時 API）。查不到城市就丟例外。"""
    if city not in BUS_ROUTES:
        raise LookupError(f"沒有「{city}」的公車路線資料，目前只支援：{list(BUS_ROUTES.keys())}")
    routes = BUS_ROUTES[city]
    if keyword:
        routes = [r for r in routes if keyword in r["route_name"] or keyword in r["note"]]
    return {"routes": routes}


CAPABILITIES = {
    "get_weather": get_weather,
    "search_bus_routes": search_bus_routes,
}

AGENT_CARD = {
    "name": "travel-info-agent",
    "description": "提供台灣天氣預報與市區公車路線查詢的旅遊資訊 agent。",
    "capabilities": [
        {
            "name": "get_weather",
            "description": "查詢指定縣市的天氣預報（溫度、降雨機率、天氣現象）。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "縣市名稱，例如：台南、台中、高雄、嘉義",
                    },
                },
                "required": ["city"],
            },
        },
        {
            "name": "search_bus_routes",
            "description": "查詢指定縣市的市區公車路線（可用關鍵字篩選路線名稱）。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "縣市名稱，例如：台南、台中、高雄、嘉義",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "選填，路線名稱關鍵字",
                    },
                },
                "required": ["city"],
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# HTTP 服務
# ---------------------------------------------------------------------------

class TravelAgentHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/agent-card":
            self._send_json(200, AGENT_CARD)
        else:
            self._send_json(404, {"error": f"未知路徑：{self.path}"})

    def do_POST(self) -> None:
        if self.path != "/tasks":
            self._send_json(404, {"error": f"未知路徑：{self.path}"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"status": "failed", "error": "請求 body 不是合法 JSON"})
            return

        capability = body.get("capability")
        tool_input = body.get("input", {})
        handler = CAPABILITIES.get(capability)
        if handler is None:
            self._send_json(400, {"status": "failed", "error": f"不支援的能力：{capability}"})
            return

        try:
            output = handler(**tool_input)
            self._send_json(200, {"status": "completed", "output": output})
        except Exception as e:
            self._send_json(200, {"status": "failed", "error": str(e)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[travel-agent] {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer(("localhost", PORT), TravelAgentHandler)
    print(f"旅遊資訊 agent 啟動：http://localhost:{PORT}")
    print(f"  GET  http://localhost:{PORT}/agent-card")
    print(f"  POST http://localhost:{PORT}/tasks")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n關閉旅遊資訊 agent")
        server.shutdown()


if __name__ == "__main__":
    main()
