"""階段 5：A2A（Agent-to-Agent）練習 —— 美食 agent（用戶端，這階段真正的練習）。

先啟動 05_a2a_travel_agent.py（旅遊資訊 agent，port 8500），再跑這支。

測試題（PLAN.md 原題）：
  「這週末去台南吃食尚玩家名店，會下雨嗎？怎麼搭車？」

流程對照 PLAN.md：
  使用者問美食 agent → 美食 agent 發現自己沒有天氣/交通能力
  → 讀對方的 agent card（GET /agent-card）→ 把對方宣告的能力動態轉成 Claude 工具
  → Claude 決定要呼叫哪個工具 → 是本地能力就直接執行，是遠端能力就發任務給旅遊 agent
    （POST /tasks）→ 彙整回覆。

這支檔案跟階段 2 的 ReAct 迴圈長得很像（一樣是 Thought/Action/Observation），
差別在工具清單不是寫死的，一部分是「執行前才向另一個服務要來的」——這就是
A2A 跟 Subagent 的本質差異：Subagent 委派時，orchestrator 老早就知道子 agent
有什麼工具（因為工具是同一支程式裡定義的）；A2A 的兩邊是不同 codebase、
可能不同團隊、不同信任邊界，一邊只能靠對方主動公開的 agent card 才知道
對方能做什麼。

執行前：
  pip install anthropic python-dotenv
  export ANTHROPIC_API_KEY=...（或放進 .env）
  另一個 terminal 先跑：python3 05_a2a_travel_agent.py
執行：
  python3 05_a2a_food_agent.py test     # 直接跑測試題，不用另外開 server
  python3 05_a2a_food_agent.py          # 啟動美食 agent 自己的 HTTP 服務（port 8600）

驗收標準（對照 PLAN.md）：
  [ ] 兩個服務分開啟動、分開的 codebase，只靠 HTTP 溝通
  [ ] 美食 agent 是從 agent card 動態得知對方能力，不是寫死的 if/else
      ——檢查你自己寫的 capabilities_to_tools() 有沒有針對特定能力名稱判斷；
      如果旅遊 agent 的 agent card 明天多宣告一個新能力，這支檔案的程式碼
      要能「一行都不改」就自動多一個可用工具
  [ ] 能回答：A2A 和 Subagent 的本質差異？（信任邊界、跨組織、能力發現 vs 同程式內委派）

延伸（PLAN.md）：
  對照 Google A2A 協定規格（https://google.github.io/A2A/），看它比這裡的
  簡化版多定義了什麼（任務狀態機、串流、認證），想想為什麼真實世界需要這些：
  - 任務狀態（pending/working/completed/failed）：這裡的任務是同步等回應，
    但真實世界很多任務（例如訂票）沒辦法馬上完成，需要輪詢或推播狀態變化
  - 串流：長時間任務要能邊執行邊回報進度，不是等到完成才吐一個結果
  - 認證：這裡兩個服務互相完全信任、沒有身分驗證；跨組織的 A2A 一定要有
    某種形式的憑證（API key、OAuth token），否則任何人都能冒充呼叫你的 agent
"""

import importlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
print(f"Using model: {MODEL}")

client = anthropic.Anthropic()

PORT = 8600
TRAVEL_AGENT_URL = "http://localhost:8500"
TEST_QUESTION = "這週末去台南吃食尚玩家名店，會下雨嗎？怎麼搭車？"

# 借用階段 3 現成的 TF-IDF 檢索（03_rag.py 檔名開頭是數字，不能用
# `import 03_rag`，要用 importlib）。
sys.path.insert(0, str(Path(__file__).parent))
rag = importlib.import_module("03_rag")


# ---------------------------------------------------------------------------
# 本地能力：美食推薦（已經寫好，這階段的重點不是 RAG 本身）
# ---------------------------------------------------------------------------

def recommend_food_itinerary(question: str) -> dict:
    """用階段 3 的 RAG 檢索，回答跟食尚玩家店家有關的問題。"""
    retrieved = rag.retrieve_top_k(question, k=3)
    prompt = rag.build_prompt_retrieved(question, retrieved, rag.ALL_DOCS)
    answer = rag.ask(prompt)
    return {"answer": answer, "sources": [name for name, _ in retrieved]}


LOCAL_TOOLS = [
    {
        "name": "recommend_food_itinerary",
        "description": "根據問題，從食尚玩家介紹過的店家資料庫裡檢索並推薦店家、回答美食相關問題。",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "跟美食/店家有關的問題"},
            },
            "required": ["question"],
        },
    },
]

LOCAL_HANDLERS = {
    "recommend_food_itinerary": recommend_food_itinerary,
}
# 遠端 agent card
AGENT_CARD = {
    "name": "food-agent",
    "description": "提供食尚玩家店家美食推薦與行程規劃的 agent。",
    "capabilities": [LOCAL_TOOLS[0]],
}

SYSTEM_PROMPT = """你是台灣美食行程助手。你手上的工具分成兩種：
一種是本地的美食推薦工具，另一種是動態發現的遠端旅遊資訊工具（天氣、公車路線）
——這些遠端工具的存在與規格，是執行當下才向另一個 agent 服務要來的，不是你原本就知道的。
遇到天氣、交通這類你自己沒有能力回答的問題，找找看有沒有合適的工具可以用。
回答前要有足夠資訊支撐，不要用記憶或猜測回答天氣、交通這類即時資訊。"""


# ---------------------------------------------------------------------------
# 這階段真正的練習：從 agent card 動態發現能力 + 統一的遠端派送
# ---------------------------------------------------------------------------

def fetch_agent_card(base_url: str) -> dict:
    """跟另一個 agent 服務要它的 agent card（GET /agent-card）。"""
    try:
        with urllib.request.urlopen(f"{base_url}/agent-card", timeout=5) as resp:
            card = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"無法連線旅遊 agent（{base_url}）。"
            "請先在另一個 terminal 執行：python3 05_a2a_travel_agent.py"
        ) from e

    if not isinstance(card.get("capabilities"), list):
        raise ValueError("agent card 格式錯誤：缺少 capabilities 清單")
    return card


def capabilities_to_tools(agent_card: dict) -> list[dict]:
    """把遠端 agent card 的 capabilities 動態轉成 Claude tool 定義清單。

    agent card 裡每個 capability 已經長得跟 Claude tool 一模一樣
    （{"name", "description", "input_schema"}），所以這裡不用重新設計 schema，
    只要把 list 原封不動抽出來就好。重點是這個函式完全不知道、也不需要知道
    capability 到底叫什麼名字——換句話說，旅遊 agent 明天多宣告一個新能力，
    這裡不用改任何一行就會自動多出一個可用工具。
    """
    tools = []
    for i, cap in enumerate(agent_card["capabilities"]):
        if not isinstance(cap, dict):
            raise ValueError(f"capabilities[{i}] 必須是物件")
        for key in ("name", "description", "input_schema"):
            if key not in cap:
                raise ValueError(f"capabilities[{i}] 缺少必要欄位：{key}")
        tools.append({
            "name": cap["name"],
            "description": cap["description"],
            "input_schema": cap["input_schema"],
        })
    return tools


def call_remote_capability(base_url: str, capability: str, tool_input: dict) -> dict:
    """呼叫遠端 agent 的某個能力（POST /tasks），回傳 output，失敗就丟例外。

    這個函式只有一條路：不管 capability 是 get_weather 還是 search_bus_routes
    還是任何未來新加的能力，都是同一段程式碼在發 HTTP 請求——「是哪個能力」
    只是 body 裡的一個字串欄位，不是程式碼路徑的分岔點。
    """
    body = json.dumps({"capability": capability, "input": tool_input}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/tasks",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"無法連線旅遊 agent（{base_url}）。"
            "請確認 05_a2a_travel_agent.py 是否仍在執行。"
        ) from e

    if result.get("status") != "completed":
        raise RuntimeError(result.get("error", "遠端任務失敗，但沒有回傳錯誤訊息"))

    output = result["output"]
    if isinstance(output, dict) and output.get("error"):
        raise RuntimeError(output["error"])
    return output


def run_food_agent_task(question: str, max_turns: int = 8) -> str:
    """核心迴圈：跟階段 2 的 run_react_loop 架構一樣（Thought/Action/Observation），
    差別只在工具清單混合了本地工具（LOCAL_TOOLS/LOCAL_HANDLERS）跟執行當下才
    向旅遊 agent 要來的遠端工具。
    """
    # 執行前才去問對方「你現在有什麼能力」——這就是 A2A 的能力發現，
    # 不是在寫程式的當下就寫死對方一定有 get_weather 這個工具。
    remote_card = fetch_agent_card(TRAVEL_AGENT_URL)
    remote_tools = capabilities_to_tools(remote_card)
    remote_names = {tool["name"] for tool in remote_tools}
    tools = LOCAL_TOOLS + remote_tools

    messages = [{"role": "user", "content": question}]
    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
            tools=tools, messages=messages,
        )

        for block in response.content:
            if block.type == "text":
                print(f"[Thought] {block.text}")

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[Action] {block.name}({block.input})")
            is_error = False
            try:
                if block.name in LOCAL_HANDLERS:
                    # 本地能力：直接呼叫 Python 函式，跟階段 2 一樣
                    output = LOCAL_HANDLERS[block.name](**block.input)
                elif block.name in remote_names:
                    # 遠端能力：發 HTTP 任務給旅遊 agent，不是直接呼叫函式
                    output = call_remote_capability(TRAVEL_AGENT_URL, block.name, block.input)
                else:
                    raise LookupError(f"未知工具：{block.name}")
                result_str = json.dumps(output, ensure_ascii=False)
            except Exception as e:
                result_str, is_error = str(e), True
            print(f"[Observation] {'⚠️ 錯誤：' if is_error else ''}{result_str}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": tool_results})

    return "⚠️ 超過 max_turns，agent 可能卡住了"


# ---------------------------------------------------------------------------
# HTTP 服務（已經寫好）
# ---------------------------------------------------------------------------

class FoodAgentHandler(BaseHTTPRequestHandler):
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

        question = body.get("input", {}).get("question", "")
        try:
            answer = run_food_agent_task(question)
            self._send_json(200, {"status": "completed", "output": {"answer": answer}})
        except Exception as e:
            self._send_json(200, {"status": "failed", "error": str(e)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[food-agent] {fmt % args}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print(f"問題：{TEST_QUESTION}\n")
        print(run_food_agent_task(TEST_QUESTION))
        return

    server = ThreadingHTTPServer(("localhost", PORT), FoodAgentHandler)
    print(f"美食 agent 啟動：http://localhost:{PORT}")
    print(f"  GET  http://localhost:{PORT}/agent-card")
    print(f"  POST http://localhost:{PORT}/tasks")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n關閉美食 agent")
        server.shutdown()


if __name__ == "__main__":
    main()
