"""階段 6：LangGraph 練習——同一件事，換框架做。

概念：階段 1–5 刻意不用框架，是為了看見 ReAct 迴圈、subagent 委派這些機制的
裸實作。這一階段用 LangGraph 的 StateGraph/node/edge 重做一次「多專才協作」，
重點不是做出新功能，而是**對照**——同一份資料、同一個測試題，手刻版 vs 框架版，
才看得出框架幫你省了什麼、又用什麼抽象藏住了細節。

測試題直接複用階段 5 的題目：
    「這週末去台南吃資料集裡的名店，會下雨嗎？怎麼搭車？」

這題天生需要三種能力，剛好對應三個 node（PLAN.md 原本寫「美食評論員、交通規劃師、
預算控管」，但這題完全沒提到預算，硬塞預算 node 只是為了湊數，所以這裡改成貼題目
本身需要的三個能力）：
    - 美食評論員：階段 3 的 RAG，只索引台南店家文件，選店＋推薦理由
    - 天氣查詢：階段 5 data/weather.json 的模擬資料
    - 公車查詢：階段 5 data/bus_routes.json 的模擬資料
再加一個彙整 node，把三份報告合成最終答案。

跟階段 5 A2A 的關鍵差異：這裡是**同一支程式、同一個 process**裡的多節點協作
（LangGraph 的 graph 內部委派），不是兩個獨立部署、只靠 HTTP 溝通的服務。
節點之間怎麼傳資料、要不要共用 state，跟 A2A 的「agent card + 任務生命週期」
是完全不同的信任邊界——這正是驗收標準要你回答的對照題。

安裝（跟階段 1–5 分開，見 requirements-langgraph.txt）：
    pip install -r requirements-langgraph.txt
執行：
    python3 06_langgraph.py

驗收標準：
    [ ] 能指出 StateGraph 的 node/edge 對應到階段 2、4 手刻的哪一段程式碼
        （提示：run_subagent() 的每次呼叫 ≈ 一個 node；orchestrator 的委派順序 ≈ edge）
    [ ] 能回答：框架版省了什麼（狀態管理、訊息組裝）、又犧牲了什麼可見性
        （提示：階段 4 手刻版你能一眼看到 messages list 怎麼組；LangGraph 的
        state 怎麼在 node 之間傳遞，藏在框架內部多少？）
    [ ] 用同一測試題跑手刻版（階段 4 的 run_orchestrator 風格）vs 這支 LangGraph 版，
        比較兩者的 API 呼叫次數或 token 用量
    [ ] 能回答：這裡的「同程式多節點」跟階段 5 的「跨服務 A2A」，本質差在哪？
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

import anthropic

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
client = anthropic.Anthropic()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"

TEST_QUESTION = "這週末去台南吃資料集裡的名店，會下雨嗎？怎麼搭車？"
CITY = "台南"

# ---------------------------------------------------------------------------
# 資料與工具（沿用階段 3 RAG + 階段 5 模擬資料，已經寫好——
# 這階段的重點不是重新設計檢索或資料，是後面的 graph 組裝）
# ---------------------------------------------------------------------------

ALL_SHOPS: list[dict] = json.loads((DATA_DIR / "shops.json").read_text(encoding="utf-8"))
TAINAN_SHOPS = [s for s in ALL_SHOPS if s["city"] == CITY]
TAINAN_SHOP_NAMES = {s["name"] for s in TAINAN_SHOPS}

WEATHER: dict = json.loads((DATA_DIR / "weather.json").read_text(encoding="utf-8"))
BUS_ROUTES: dict = json.loads((DATA_DIR / "bus_routes.json").read_text(encoding="utf-8"))

_PUNCTUATION = set("，。、（）「」〈〉！？：；\n\t ")


def _tokenize(text: str) -> list[str]:
    chars = [c for c in text if c not in _PUNCTUATION]
    if len(chars) < 2:
        return chars
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def _load_tainan_docs() -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        if path.stem in TAINAN_SHOP_NAMES:
            docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


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


def _embed_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = _tokenize(query)
    tf: dict[str, int] = {}
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1
    total = len(tokens) or 1
    return {term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


TAINAN_DOCS = _load_tainan_docs()
TAINAN_IDF, TAINAN_DOC_VECTORS = _build_tfidf_index(TAINAN_DOCS)


def retrieve_food_docs(query: str, k: int = 3) -> list[dict]:
    """RAG 檢索：只在台南店家文件裡搜尋（沿用階段 3 的 TF-IDF）。"""
    query_vec = _embed_query(query, TAINAN_IDF)
    scores = [(name, _cosine_similarity(query_vec, vec)) for name, vec in TAINAN_DOC_VECTORS.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        {"shop": name, "score": round(score, 4), "excerpt": TAINAN_DOCS[name][:400]}
        for name, score in scores[:k]
    ]


def get_weather(city: str = CITY) -> dict:
    """查某縣市天氣（沿用階段 5 的模擬資料）。"""
    if city not in WEATHER:
        raise LookupError(f"沒有「{city}」的天氣資料，目前只支援：{list(WEATHER.keys())}")
    return WEATHER[city]


def search_bus_routes(city: str = CITY, keyword: str = "") -> dict:
    """查某縣市公車路線（沿用階段 5 的模擬資料）。"""
    if city not in BUS_ROUTES:
        raise LookupError(f"沒有「{city}」的公車路線資料，目前只支援：{list(BUS_ROUTES.keys())}")
    routes = BUS_ROUTES[city]
    if keyword:
        routes = [r for r in routes if keyword in r["route_name"] or keyword in r["note"]]
    return {"routes": routes}


# ---------------------------------------------------------------------------
# LangGraph state 與 node（由你手刻——這是這階段真正的練習）
# ---------------------------------------------------------------------------

class TripState(TypedDict):
    """在 node 之間傳遞的共享狀態。

    對照階段 4：這裡的 state 取代了手刻版裡「每個子 agent 各自獨立的 messages +
    orchestrator 手動抽取/組裝字串」的做法。想一下：state 全部 node 共享，
    跟階段 4 刻意做的「子 agent 互相看不到彼此 context」相比，隔離性有沒有變差？
    """
    question: str
    food_report: str
    weather_report: str
    bus_report: str
    final_answer: str


# 每個 node 各自的耗時（by node name）。獨立於 TripState 之外，用一個模組層級
# 的 dict 記錄——各 node 只寫自己的 key，平行執行時不會互相覆蓋（跟 TripState
# 裡「各 node 只寫自己的欄位」是同一個設計原則），純粹是計時用，不進 graph state。
NODE_TIMINGS: dict[str, float] = {}


def _timed(name: str):
    """裝飾器：記錄 node 執行耗時到 NODE_TIMINGS[name]，並印出來。"""
    def decorator(fn):
        def wrapper(state: TripState) -> dict:
            start = time.perf_counter()
            result = fn(state)
            elapsed = time.perf_counter() - start
            NODE_TIMINGS[name] = elapsed
            print(f"[{name}] 耗時 {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


@_timed("food_critic")
def food_critic_node(state: TripState) -> dict:
    """呼叫 retrieve_food_docs() + LLM，選出台南名店並說明理由。

    對照階段 4 的 run_food_critic()：那邊是手刻的 ReAct 迴圈（呼叫 LLM →
    判斷要不要呼叫工具 → 組 messages，可能來回好幾輪）。這裡故意只呼叫一次
    retrieve_food_docs、再呼叫一次 LLM 做摘要，不做多輪迴圈——這正是要突顯
    「LangGraph 的 node 本身不會自動幫你做 ReAct 迴圈」：迴圈邏輯還是要
    自己寫，框架省的是「怎麼把多個步驟串起來」那一層，不是單一步驟內的
    reasoning/action 機制。
    """
    print("[food_critic] 檢索台南店家文件…")
    docs = retrieve_food_docs(state["question"], k=3)
    excerpt_text = "\n\n".join(f"【{d['shop']}】{d['excerpt']}" for d in docs)

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"以下是台南店家介紹文件節錄：\n\n{excerpt_text}\n\n"
                f"請根據這些文件，回答「{state['question']}」裡跟美食相關的部分："
                "推薦哪幾家名店、各自的推薦理由。"
            ),
        }],
    )
    report = "".join(b.text for b in response.content if b.type == "text")
    print(f"[food_critic] 完成\n{report}\n")
    return {"food_report": report}


@_timed("weather")
def weather_node(state: TripState) -> dict:
    """呼叫 get_weather()，整理成一段文字放進 state["weather_report"]。

    這個 node 完全不需要 LLM——單純的資料查詢＋格式化字串，凸顯 node 不是
    「一定要包一次 LLM 呼叫」，它只是 graph 裡的一個步驟。
    """
    print("[weather] 查詢天氣…")
    forecast = get_weather(CITY)
    lines = [
        f"{p['period']}：{p['weather']}，降雨機率 {p['rain_probability']}%，"
        f"氣溫 {p['min_temp']}–{p['max_temp']}°C"
        for p in forecast["forecast"]
    ]
    report = f"{forecast['location']}天氣：" + "；".join(lines)
    print(f"[weather] 完成 → {report}")
    return {"weather_report": report}


@_timed("bus")
def bus_node(state: TripState) -> dict:
    """呼叫 search_bus_routes()，整理成文字放進 state["bus_report"]。"""
    print("[bus] 查詢公車路線…")
    result = search_bus_routes(CITY)
    lines = [
        f"{r['route_name']}（{r['from']} → {r['to']}）" + (f"，{r['note']}" if r["note"] else "")
        for r in result["routes"]
    ]
    report = f"{CITY}公車路線：" + "；".join(lines)
    print(f"[bus] 完成 → {report}")
    return {"bus_report": report}


@_timed("synthesize")
def synthesize_node(state: TripState) -> dict:
    """把 food_report / weather_report / bus_report 三段文字交給 LLM 彙整成最終答案。

    對照階段 4 的 synthesize_itinerary()：邏輯幾乎一樣（組 prompt、呼叫一次
    LLM），差別只在於這裡的輸入是從 state 讀，不是函式參數直接傳入——
    這就是 LangGraph state 取代手動傳遞回傳值的地方。
    """
    print("[synthesize] 彙整最終答案…")
    prompt = f"""使用者問題：{state['question']}

【美食評論員報告】
{state['food_report']}

【天氣報告】
{state['weather_report']}

【公車報告】
{state['bus_report']}

請把以上三份報告整合成一段完整回答，直接回應使用者的問題。"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    final_answer = "".join(b.text for b in response.content if b.type == "text")
    print("[synthesize] 完成")
    return {"final_answer": final_answer}


def build_graph():
    """組出 StateGraph：food_critic → weather → bus → synthesize → END。

    這裡選擇序列（而非下面註解的平行 fan-out/fan-in），是為了跟階段 4
    run_orchestrator() 的委派順序一比一對照——想看平行版怎麼寫，見下方
    「平行化延伸」的註解與骨架。

    驗收時想一下：food_critic → weather → bus 這裡是刻意排成序列（跟階段 4
    的委派順序一致，方便對照），但這三個 node 彼此不依賴對方的輸出——
    LangGraph 其實可以讓它們平行執行。要不要改成平行，也是你可以做的延伸實驗。

    平行化延伸（TODO，選做）：把上面序列的三條 edge 改成從入口平行 fan-out 到
    food_critic / weather / bus，三者都完成後才 fan-in 到 synthesize：

        graph.add_edge("__start__", "food_critic")
        graph.add_edge("__start__", "weather")
        graph.add_edge("__start__", "bus")
        graph.add_edge("food_critic", "synthesize")
        graph.add_edge("weather", "synthesize")
        graph.add_edge("bus", "synthesize")

    這三個 node 各自只寫自己的 state key（food_report/weather_report/bus_report），
    平行執行不會有傳統 race condition（各自等待 API 回應、不共用記憶體）。
    但如果你之後手癢把某個 node 改成寫入同一個 key（例如三個都塞進同一個
    state["logs"] list），LangGraph 合併平行分支時預設會覆蓋或報錯——這才是
    框架特有的合併衝突，不是傳統意義的 race condition，需要用
    `Annotated[list, operator.add]` 這類 reducer 明確告訴 LangGraph 怎麼合併。
    手刻版（階段 4 的 run_orchestrator）完全不會遇到這個問題，因為你自己
    一步步組字串、合併邏輯全部顯式寫在程式裡。
    """
    graph = StateGraph(TripState)
    graph.add_node("food_critic", food_critic_node)
    graph.add_node("weather", weather_node)
    graph.add_node("bus", bus_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("food_critic")
    graph.add_edge("food_critic", "weather")
    graph.add_edge("weather", "bus")
    graph.add_edge("bus", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def build_parallel_graph():
    """組出平行版 StateGraph：food_critic / weather / bus 從 START 平行 fan-out，
    三者都完成後才 fan-in 到 synthesize。

    跟 build_graph()（序列版）用同一組 node 函式，差別只在 edge 的接法——
    這正是要你實測前面討論過的重點：這三個 node 各自只寫自己的 state key
    （food_report/weather_report/bus_report），平行執行不會有傳統 race
    condition。想製造 LangGraph 特有的「合併衝突」，可以自己另外實驗把某個
    node 改成寫入同一個 key，觀察 LangGraph 在沒有 reducer 時的報錯或覆蓋行為。
    """
    graph = StateGraph(TripState)
    graph.add_node("food_critic", food_critic_node)
    graph.add_node("weather", weather_node)
    graph.add_node("bus", bus_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "food_critic")
    graph.add_edge(START, "weather")
    graph.add_edge(START, "bus")
    graph.add_edge("food_critic", "synthesize")
    graph.add_edge("weather", "synthesize")
    graph.add_edge("bus", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run(mode: str = "sequential") -> dict[str, float]:
    """依 mode 選擇序列版或平行版 graph 執行同一道測試題,方便分別呼叫比較。

    mode="sequential" → build_graph()          （food_critic → weather → bus → synthesize）
    mode="parallel"   → build_parallel_graph()  （food_critic / weather / bus 同時觸發 → synthesize）

    回傳：{node 名稱: 耗時秒數, ..., "total": 總耗時}，方便呼叫端比較不同 mode。
    """
    NODE_TIMINGS.clear()  # 每次呼叫重置，避免上一次 run 的計時殘留

    builder = {"sequential": build_graph, "parallel": build_parallel_graph}[mode]
    print(f"Using model: {MODEL}")
    print(f"Graph mode: {mode}")
    app = builder()

    total_start = time.perf_counter()
    result = app.invoke({
        "question": TEST_QUESTION,
        "food_report": "",
        "weather_report": "",
        "bus_report": "",
        "final_answer": "",
    })
    total_elapsed = time.perf_counter() - total_start

    print(result["final_answer"])

    print(f"\n{'=' * 60}")
    print(f"耗時明細（{mode}）")
    print(f"{'=' * 60}")
    for name in ("food_critic", "weather", "bus", "synthesize"):
        print(f"  {name:<12} {NODE_TIMINGS.get(name, 0):.2f}s")
    print(f"  {'—' * 20}")
    print(f"  {'total':<12} {total_elapsed:.2f}s")
    print(
        "  （sequential 的 total ≈ 四個 node 耗時加總；"
        "parallel 的 total ≈ max(food_critic, weather, bus) + synthesize，"
        "兩者若差不多，代表這組資料裡平行化省不了多少時間——見前面分析）"
    )

    print(f"\n{'=' * 60}")
    print("驗收檢查（人工核對）")
    print(f"{'=' * 60}")
    print("[ ] 能指出 node/edge 對應到階段 2、4 手刻的哪一段程式碼？")
    print("[ ] 能回答：框架版省了什麼、又犧牲了什麼可見性？")
    print("[ ] 跟手刻版比較過 API 呼叫次數或 token 用量？")
    print("[ ] 能回答：同程式多節點 vs 階段 5 跨服務 A2A，本質差在哪？")
    if mode == "parallel":
        print("[ ] 平行版跟序列版相比,總耗時有沒有變短?(三個 node 各自等 API 回應,")
        print("      平行執行理論上耗時取決於最慢的那個 node,而不是三者加總)")

    return {**NODE_TIMINGS, "total": total_elapsed}


def main() -> None:
    mode = "parallel" if "--parallel" in sys.argv else "sequential"
    run(mode)


if __name__ == "__main__":
    main()
