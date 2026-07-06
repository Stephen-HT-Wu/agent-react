# 六階段練習流程圖解說

本文件用流程圖說明 [`01_cot.py`](01_cot.py) 到 [`06_langgraph.py`](06_langgraph.py) 六個練習的**資料怎麼流、誰呼叫誰、跟上一階差在哪**。

| 階段 | 檔案 | 核心機制 | 資料來源 |
|------|------|----------|----------|
| 1 | `01_cot.py` | Chain-of-Thought | `shops.json`（整包塞 prompt） |
| 2 | `02_react.py` | ReAct 工具迴圈 | `shops.json`（工具查詢） |
| 3 | `03_rag.py` | RAG 檢索增強 | `data/docs/*.md` |
| 4 | `04_subagent.py` | Subagent 委派 | RAG + `shops.json` + 距離工具 |
| 5 | `05_a2a_*.py` | Agent-to-Agent | 本地 RAG + 遠端 HTTP 服務 |
| 6 | `06_langgraph.py` | LangGraph StateGraph | 同階段 3 + 5 的模擬資料 |

---

## 階段 1：CoT（Chain-of-Thought）

**測試題：** 幫我用資料集裡的店，排台南一日美食行程（週三出發，預算 800 元）。

**重點：** 同一題問兩次，差別只在 prompt 有沒有要求「先推理、後結論」。沒有工具、沒有檢索，全部台南店家直接塞進 prompt。

```mermaid
flowchart TD
    Start([執行 01_cot.py]) --> Load[讀 shops.json<br/>篩選 city=台南]
    Load --> Format[格式化成文字清單<br/>店名 / 餐點 / hours / 價位 / note]
    Format --> A[build_prompt_a<br/>直接要答案]
    Format --> B[build_prompt_b<br/>要求逐項檢查後再排行程]
    A --> LLM_A[Claude API<br/>單次 messages.create]
    B --> LLM_B[Claude API<br/>單次 messages.create]
    LLM_A --> OutA[印出 A 版行程]
    LLM_B --> OutB[印出 B 版行程]
    OutA --> Check[人工驗收]
    OutB --> Check
    Check --> Trap1{老城鹹粥<br/>有沒被排到晚餐?}
    Check --> Trap2{府前冰果室<br/>週三公休有排除?}
    Check --> Trap3{預算有加總?}
```

### 推理檢查清單（B 版 CoT 引導）

```mermaid
flowchart LR
    Q[使用者問題] --> T1[1. 營業時間 vs 用餐時段]
    T1 --> T2[2. 公休日 vs 出遊日期]
    T2 --> T3[3. 地理動線]
    T3 --> T4[4. 預算加總]
    T4 --> T5[5. 輸出最終行程]
```

**跟下一階的差別：** CoT 只能重新排列 prompt 裡已有的資訊；無法回答「現在幾點、這家店此刻開不開」這類需要即時查詢的問題。

---

## 階段 2：ReAct（Reason + Act）

**測試題：** 現在晚上 11 點在逢甲，資料集裡的宵夜哪家還開著而且最近？

**重點：** 手刻 `while` 迴圈，模型在 Thought → Action → Observation 之間循環，直到不再呼叫工具。

```mermaid
flowchart TD
    Start([run_react_loop]) --> Init[messages = 使用者問題]
    Init --> Loop{turn < max_turns?}
    Loop -->|是| API[Claude API<br/>system + tools + messages]
    API --> Stop{stop_reason<br/>== tool_use?}
    Stop -->|否| Answer[印出最終文字<br/>結束迴圈]
    Stop -->|是| Thought[印出 Thought 文字]
    Thought --> Action[遍歷 tool_use blocks]
    Action --> Dispatch{工具名稱}
    Dispatch -->|search_shops| S1[查 shops.json<br/>依 city / keyword 篩選]
    Dispatch -->|is_open| S2[查營業時間<br/>含跨夜判斷]
    Dispatch -->|distance| S3[Haversine 算距離<br/>支援地標如逢甲夜市]
    S1 --> Obs[Observation 寫回 messages]
    S2 --> Obs
    S3 --> Obs
    Obs --> Loop
    Loop -->|否| Timeout[超過 max_turns 警告]
```

### 典型解題路徑（測試題）

```mermaid
flowchart LR
    S[search_shops<br/>city=台中] --> C[候選宵夜店清單]
    C --> O[is_open<br/>time=23:00]
    O --> F[篩掉未營業<br/>如河南夜宵生魚]
    F --> D[distance<br/>起點=逢甲夜市]
    D --> R[選最近且開著的店]
```

**陷阱：** 河南夜宵生魚名氣大但 00:00 才開——只做關鍵字比對、不呼叫 `is_open()` 就會答錯。

---

## 階段 3：RAG（Retrieval-Augmented Generation）

**測試題：** 資料集裡的碗粿老店中，哪一家只賣碗粿和魚羹、均一價 35 元？出自哪一篇店家介紹？

**重點：** 同一題跑 A / B / C 三版，對照「沒文件 → 全文 → 檢索 top-3」的差異。

```mermaid
flowchart TD
    Start([執行 03_rag.py]) --> Load[load_all_docs<br/>讀 20 篇 md]
    Load --> Index[build_tfidf_index<br/>bigram → TF-IDF 向量]
    Index --> Q[TEST_QUESTION]

    Q --> A[A 版 build_prompt_no_context<br/>不給任何文件]
    Q --> B[B 版 build_prompt_full_context<br/>20 篇全文塞 prompt]
    Q --> R[retrieve_top_k k=3<br/>餘弦相似度排序]
    R --> C[C 版 build_prompt_retrieved<br/>只塞 top-3 文件]

    A --> LLM[Claude API ask]
    B --> LLM
    C --> LLM
    LLM --> Compare[人工比對 A/B/C<br/>幻覺 vs 正確出處]
```

### 檢索管線（C 版）

```mermaid
flowchart LR
    Doc[每篇 md 內文] --> Tok[tokenize<br/>中文 bigram]
    Tok --> TFIDF[TF × IDF<br/>文件向量]
    Q2[使用者問題] --> QVec[embed_query]
    QVec --> Cos[cosine_similarity<br/>與每篇比對]
    TFIDF --> Cos
    Cos --> Top[取 top-k 店名]
    Top --> Prompt[組裝 prompt + LLM]
```

**跟階段 2 的差別：**

| | `search_shops`（階段 2） | RAG（階段 3） |
|---|---|---|
| 查詢方式 | 結構化欄位比對 | 語意/字面相似度 |
| 適合 | 「台中有哪些店」 | 「均一價 35 元的細節在哪篇」 |
| 資料 | `shops.json` | `docs/*.md` |

---

## 階段 4：Subagent（多專才委派）

**測試題：** 兩天一夜台中美食之旅，預算 3000，不騎車用大眾運輸。

**重點：** 一個 orchestrator 依序委派三位子 agent，各自有獨立 `messages`、不同工具集；子 agent 互相看不到對方 context。

```mermaid
flowchart TD
    Start([run_orchestrator]) --> FC[① 美食評論員<br/>run_food_critic]
    FC --> FCloop[ReAct 迴圈<br/>工具: retrieve_food_docs RAG]
    FCloop --> FCrep[food_report 文字報告]
    FCrep --> Extract[extract_shop_names]
    Extract --> BC[② 預算控管<br/>run_budget_check]
    BC --> BClop[ReAct 迴圈<br/>工具: check_budget / shops.json]
    BClop --> BCrep[budget_report]
    BCrep --> Conflict{within_budget?}
    Conflict -->|否| Arb[③ 仲裁<br/>再呼叫美食評論員<br/>要求替換高價店]
    Arb --> BC2[重新預算檢查]
    BC2 --> TP
    Conflict -->|是| TP[④ 交通規劃師<br/>run_transport_planner]
    TP --> TPrep[transport_report]
    TPrep --> Syn[⑤ synthesize_itinerary<br/>orchestrator 彙整]
    Syn --> Final[兩天一夜行程摘要]
```

### Context 隔離示意

```mermaid
flowchart TB
    subgraph Orchestrator[行程總監 — 不掛工具]
        O1[收到 task]
        O2[只收各 agent 最終報告 str]
        O3[仲裁 + 彙整]
    end

    subgraph FC[美食評論員]
        FCm[messages 獨立]
        FCt[RAG 工具]
    end

    subgraph BG[預算控管]
        BGm[messages 獨立]
        BGt[price 工具]
    end

    subgraph TP[交通規劃師]
        TPm[messages 獨立]
        TPt[distance 工具]
    end

    O1 --> FC
    FC -->|food_report| O2
    O2 --> BG
    BG -->|budget_report| O3
    O3 --> TP
    TP -->|transport_report| O3
```

**刻意衝突：** 評論員不知道預算，可能把河南夜宵生魚等高價宵夜排進去 → 預算控管判定超標 → orchestrator 仲裁。

---

## 階段 5：A2A（Agent-to-Agent）

**測試題：** 這週末去台南吃資料集裡的名店，會下雨嗎？怎麼搭車？

**重點：** 兩個獨立 HTTP 服務（port 8500 / 8600），美食 agent 執行時才向旅遊 agent 要 agent card，動態組合工具清單。

### 系統架構

```mermaid
flowchart LR
    User[使用者 / test 模式] --> Food[美食 agent<br/>05_a2a_food_agent.py<br/>:8600]
    Food -->|GET /agent-card| Travel[旅遊 agent<br/>05_a2a_travel_agent.py<br/>:8500]
    Food -->|POST /tasks| Travel
    Food --> RAG[本地 03_rag<br/>recommend_food_itinerary]
    Travel --> W[weather.json]
    Travel --> B[bus_routes.json]
```

### 美食 agent 核心迴圈

```mermaid
flowchart TD
    Start([run_food_agent_task]) --> Card[fetch_agent_card<br/>GET localhost:8500/agent-card]
    Card --> Merge[capabilities_to_tools<br/>遠端能力 + LOCAL_TOOLS]
    Merge --> Loop{ReAct 迴圈<br/>同階段 2}
    Loop --> Decide{工具名稱}
    Decide -->|recommend_food_itinerary| Local[RAG 本地執行]
    Decide -->|get_weather 等| Remote[call_remote_capability<br/>POST /tasks]
    Remote --> Travel[旅遊 agent 執行<br/>回傳 output]
    Local --> Obs[Observation]
    Travel --> Obs
    Obs --> Loop
    Loop -->|不再 tool_use| Answer[彙整回覆]
```

### 旅遊 agent 請求處理

```mermaid
flowchart TD
    GET[GET /agent-card] --> CardJSON[回傳 AGENT_CARD<br/>name / description / capabilities]
    POST[POST /tasks] --> Parse[解析 capability + input]
    Parse --> Handler{CAPABILITIES 對照表}
    Handler -->|get_weather| W[get_weather]
    Handler -->|search_bus_routes| B[search_bus_routes]
    W --> Resp[status: completed<br/>output: {...}]
    B --> Resp
```

**跟階段 4 的差別：**

| | Subagent（階段 4） | A2A（階段 5） |
|---|---|---|
| 部署 | 同一支 Python 程式 | 兩個 HTTP 服務 |
| 能力發現 | 寫程式時就知道 | 執行時讀 agent card |
| 通訊 | 函式呼叫 | JSON-over-HTTP |
| 信任邊界 | 同 process | 跨服務 |

---

## 階段 6：LangGraph（框架對照）

**測試題：** 同階段 5——這週末去台南吃資料集裡的名店，會下雨嗎？怎麼搭車？

**重點：** 用 LangGraph 的 `StateGraph` 把「美食 / 天氣 / 公車 / 彙整」做成 node + edge，對照階段 4 手刻 orchestrator 與階段 5 跨服務 A2A。**本檔案 node 實作為 TODO 骨架。**

### 規劃中的 Graph（序列版）

```mermaid
flowchart LR
    Start([__start__]) --> FC[food_critic_node<br/>RAG + LLM]
    FC --> W[weather_node<br/>get_weather]
    W --> B[bus_node<br/>search_bus_routes]
    B --> S[synthesize_node<br/>LLM 彙整]
    S --> End([END])
```

### 共享 State（TripState）

```mermaid
flowchart TD
    subgraph State[TripState 在 node 間傳遞]
        Q[question]
        FR[food_report]
        WR[weather_report]
        BR[bus_report]
        FA[final_answer]
    end

    FC2[food_critic_node] -->|寫入| FR
    W2[weather_node] -->|寫入| WR
    B2[bus_node] -->|寫入| BR
    S2[synthesize_node] -->|讀取 FR/WR/BR<br/>寫入| FA
```

### 延伸：平行 fan-out / fan-in

三個 node 彼此不依賴輸出，可改為平行執行後再匯入 synthesize：

```mermaid
flowchart TD
    Start([__start__]) --> FC[food_critic]
    Start --> W[weather]
    Start --> B[bus]
    FC --> Syn[synthesize]
    W --> Syn
    B --> Syn
    Syn --> End([END])
```

### 三階段對照（同一測試題）

```mermaid
flowchart TB
    subgraph S4[階段 4 手刻 Subagent]
        S4a[run_subagent × N]
        S4b[orchestrator 函式順序呼叫]
    end

    subgraph S5[階段 5 A2A]
        S5a[本地 RAG]
        S5b[HTTP 遠端天氣/公車]
        S5c[ReAct 動態選工具]
    end

    subgraph S6[階段 6 LangGraph]
        S6a[同 process 多 node]
        S6b[StateGraph 管理 state]
        S6c[edge 定義執行順序]
    end

    S4 -->|同程式委派| S6
    S5 -->|能力拆分思路| S6
```

| 維度 | 階段 4 手刻 | 階段 5 A2A | 階段 6 LangGraph |
|------|-------------|------------|------------------|
| 編排方式 | Python 函式順序 | ReAct + HTTP | node / edge |
| 狀態 | 各自 messages + 字串報告 | HTTP 回應 | 共享 TripState |
| Context 隔離 | 刻意隔離 | 服務邊界隔離 | 預設共享 state |
| 可見性 | 高（全在程式裡） | 中（HTTP 可 log） | 低（框架內部） |

---

## 六階段演進總覽

```mermaid
flowchart LR
    D[(data/)] --> S1[1 CoT<br/>prompt 推理]
    D --> S2[2 ReAct<br/>結構化工具]
    D --> S3[3 RAG<br/>文件檢索]
    S2 --> S4[4 Subagent<br/>多專才委派]
    S3 --> S4
    S3 --> S5[5 A2A<br/>跨 HTTP 服務]
    S2 --> S5
    S3 --> S6[6 LangGraph<br/>框架編排]
    S5 -.對照.-> S6
    S4 -.對照.-> S6
```

**閱讀建議：**

1. 先跑 [`01_cot.py`](01_cot.py) → [`02_react.py`](02_react.py)，理解「只有 prompt」vs「能查資料」。
2. 再跑 [`03_rag.py`](03_rag.py)，理解「結構化查詢」vs「文件檢索」。
3. [`04_subagent.py`](04_subagent.py) 把 2 + 3 拆成不同專才；[`05_a2a_food_agent.py`](05_a2a_food_agent.py) 把能力拆到不同服務。
4. 最後做 [`06_langgraph.py`](06_langgraph.py)，對照手刻版省了什麼、藏了什麼。

更多實測結果與機制筆記見 [NOTES.md](NOTES.md)；各階段驗收標準見 [PLAN.md](PLAN.md)。
