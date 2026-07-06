# 做中學：台灣美食 Agent 五階段練習計畫

素材主軸:以本專案練習資料集裡的虛構店家為資料來源,做一個「台灣美食行程 AI」。
五個階段各對應一個概念,由淺入深:CoT → ReAct → RAG → Subagent → A2A。
每階段都有「產出物」和「驗收標準」——驗收過了才進下一階段。

已完成階段的實際結果（跑出來發生了什麼，不是預期）記錄在 [NOTES.md](NOTES.md)。

---

## 階段 0:準備資料集(半天)

所有練習共用一份資料。手工整理 20–40 筆練習用店家,存成 `data/shops.json`:

```json
{
  "name": "老城鹹粥",
  "city": "台南",
  "dish": "虱目魚鹹粥",
  "episode": "練習資料：台南早餐",
  "price_range": "100-200",
  "hours": "04:30-12:00",
  "lat": 22.989, "lng": 120.199,
  "note": "凌晨開賣、賣完收攤"
}
```

另外為每家店寫一段 100–300 字的「店家介紹摘要」存成 `data/docs/*.md`(RAG 階段用)。
手工整理資料本身就是練習——你會被迫思考 schema 設計:哪些欄位給工具查詢用(結構化),哪些給檢索用(非結構化)。

---

## 階段 1:CoT(Chain-of-Thought)——只用 prompt,不寫工具

**概念**:讓模型把推理過程寫出來,再下結論。這是後面所有東西的地基(ReAct 的 "Reasoning" 就是 CoT)。

**練習**:寫一支 `01_cot.py`,同一個問題問兩次:
- A 版:直接問「幫我排台南一日美食行程」
- B 版:要求模型先列出考量(營業時間衝突、地理動線、早餐/午餐/宵夜分配、預算加總),逐項推理後才給行程

把 `shops.json` 的台南店家直接塞進 prompt(這階段刻意不做檢索)。

**驗收標準**:
- [ ] 找出至少一個 A 版排錯、B 版排對的案例(例如把 04:30 收攤的鹹粥排在晚餐)
- [ ] 能用一句話說明為什麼 CoT 有幫助(hint: 讓錯誤在中間步驟就能被檢查)

**延伸**:試 few-shot CoT——給一個「示範推理」的例子,觀察輸出格式怎麼被帶著走。

---

## 階段 2:ReAct(Reason + Act)——手刻 agent loop,不用框架

**概念**:Thought → Action(呼叫工具)→ Observation → 再 Thought,循環到能回答為止。

**練習**:寫 `02_react.py`,手刻 while 迴圈 + tool use API,提供 3 個工具:
1. `search_shops(city, dish_keyword)` — 查 shops.json
2. `is_open(shop_name, time)` — 判斷某時間有沒有開
3. `distance(shop_a, shop_b)` — 用 lat/lng 算直線距離

測試問題:「現在晚上 11 點在逢甲,資料集裡的宵夜哪家還開著而且最近?」——這題必須連續用三個工具才答得出來,單靠 prompt 答不了。

**驗收標準**:
- [ ] 印出完整的 Thought/Action/Observation 軌跡,能指著軌跡解釋每一步
- [ ] 故意讓一個工具回傳錯誤(店名打錯),觀察 agent 會不會自我修正重查
- [ ] 能回答:ReAct 跟階段 1 的差別是什麼?(CoT 只能用 prompt 裡有的資訊;ReAct 能主動取得新資訊)

**延伸**:加一個 `max_turns` 上限,觀察 agent 卡住鬼打牆的樣子——這是理解 agent 失敗模式的重要一課。

---

## 階段 3:RAG——當資料塞不進 prompt

**概念**:把文件切塊 → 向量化 → 依問題檢索最相關的幾塊 → 塞進 prompt 回答。

**練習**:寫 `03_rag.py`,用階段 0 的 `data/docs/*.md`:
1. 先做「假 RAG」:全部文件塞 prompt(基準線)
2. 再做真 RAG:embedding + 餘弦相似度檢索 top-3(文件少,不需要向量資料庫,numpy 就夠)
3. 要求回答必須附出處(哪一篇介紹、哪家店)

測試問題:「資料集裡哪家店的魚皮加蛋?是誰推薦的?」

**驗收標準**:
- [ ] 做一次 ablation:同一題「有檢索 vs 沒檢索(也不塞全文)」對比,沒檢索的版本應該出現幻覺或答不出來
- [ ] 找出一個檢索失敗的問題(語意相近但撈錯文件),並說明原因
- [ ] 能回答:RAG 跟階段 2 的 `search_shops` 工具差在哪?(結構化查詢 vs 語意檢索;其實 RAG 檢索也可以包成 ReAct 的一個工具——試試看)

**延伸**:把 RAG 檢索包成工具接進階段 2 的 ReAct loop,變成 agentic RAG。

---

## 階段 4:Subagent——一個 orchestrator 帶多個專才

**概念**:主 agent 把子任務委派給各自有獨立 context、獨立工具的子 agent,再彙整結果。重點是「context 隔離」——子 agent 看不到彼此的對話。

**練習**:寫 `04_subagent.py`,一個「行程總監」orchestrator + 三個子 agent:
- **美食評論員**:只有 RAG 工具(階段 3),負責選店和推薦理由
- **交通規劃師**:只有 distance/routing 工具(階段 2),負責排動線
- **預算控管**:只有 shops.json 的價格資料,負責檢查總花費

任務:「兩天一夜台中美食之旅,預算 3000,不騎車用大眾運輸」。
可參考你 `ds-agent-system-demo` 的 agents/ 架構,但這次自己手刻委派邏輯。

**驗收標準**:
- [ ] 每個子 agent 的 system prompt 和工具集都不同,且互相看不到對方的 context
- [ ] 觀察一次「子 agent 結論衝突」(評論員推的店超出預算),orchestrator 怎麼仲裁
- [ ] 能回答:什麼時候該用 subagent、什麼時候一個 agent 掛多工具就好?(hint: context 汙染、工具太多選錯、平行化)

---

## 階段 5:A2A(Agent-to-Agent)——跨服務的 agent 互通

**概念**:Subagent 是同一個程式內的委派;A2A 是**兩個獨立部署的 agent 服務**透過協定(agent card、任務生命週期)互相發現、溝通。

**練習**:跑兩個獨立的 HTTP 服務:
1. **美食 agent**:把階段 4 的成果包成服務,提供 agent card(宣告能力:美食推薦、行程規劃)
2. **旅遊資訊 agent**:改造你現成的 `taiwan-travel-ai`(它已經有天氣 cwa.py、交通 tdx.py!)包成第二個服務

流程:使用者問美食 agent「這週末去台南吃資料集裡的名店,會下雨嗎?怎麼搭車?」→ 美食 agent 發現自己沒有天氣/交通能力 → 讀對方的 agent card → 發任務給旅遊 agent → 彙整回覆。
先用自訂的簡單 JSON-over-HTTP 實作一次,再對照 Google A2A 協定規格,看它多定義了什麼(串流、任務狀態、認證)、為什麼需要。

**驗收標準**:
- [ ] 兩個服務分開啟動、分開的 codebase,只靠 HTTP 溝通
- [ ] 美食 agent 是從 agent card **動態得知**對方能力,不是寫死的 if/else
- [ ] 能回答:A2A 和 Subagent 的本質差異?(信任邊界、跨組織、能力發現 vs 同程式內委派)

---

## 階段 6:LangGraph——同一件事,換框架做

**概念**:前面五階段刻意不用框架,是為了看見 ReAct 迴圈、subagent 委派這些機制的裸實作。這一階段用 LangGraph 的 `StateGraph`/node/edge 重做一次「ReAct + Subagent」,重點不是做出新功能,而是**對照**——同一份資料、同一個測試題,手刻版 vs 框架版,才看得出框架幫你省了什麼、又用什麼抽象藏住了細節。

**練習**:寫 `06_langgraph.py`,用 LangGraph 重建多專才協作:
- 沿用階段 3 的 RAG(限縮台南店家)、階段 5 的天氣/公車模擬資料
- 四個 node:美食評論員(RAG)、天氣查詢、公車查詢、彙整——測試題完全沒提到預算,所以沒有沿用階段 4 的預算控管 node
- 用 `StateGraph` 的 edge 決定 orchestrator 怎麼委派與彙整,取代手刻的委派邏輯

**測試題**:直接複用階段 5 的題目:「這週末去台南吃資料集裡的名店,會下雨嗎?怎麼搭車?」——但這次不開兩個 HTTP 服務,改成單一 graph 內的多節點協作,順便對照「A2A(跨服務)vs LangGraph 內的 multi-node(同程式)」的差異。

**驗收標準**:
- [ ] 能指出 LangGraph 的 `StateGraph`/node/edge 對應到階段 2、4 手刻的哪一段程式碼
- [ ] 能回答:框架版省了什麼(例如狀態管理、工具呼叫的重試/錯誤處理)、又犧牲了什麼可見性
- [ ] 用同一測試題跑手刻版(階段 2+4)vs LangGraph 版,比較兩者的 token 用量或呼叫次數
- [ ] **平行化延伸**:把美食評論員/天氣/公車三個 node 從序列改成平行執行的 edge,觀察 state 合併行為。這三個 node 各自寫入獨立的 state key(`food_report`/`weather_report`/`bus_report`),平行執行不會有傳統 race condition(各自等待 I/O、不共用同一塊記憶體),但如果不小心讓兩個 node 寫「同一個 key」,LangGraph 合併平行分支時預設會覆蓋或報錯——這才是框架特有的衝突,要用 `Annotated[T, reducer]` 明確定義合併策略。能回答:為什麼手刻版(階段 4 的 `run_subagent`)完全不會遇到這個問題?(提示:手刻版是你自己一步步組字串、自己決定誰先誰後,合併邏輯全部顯式寫在 `run_orchestrator` 裡;框架把「平行分支怎麼合併」這件事變成隱式規則,不寫 reducer 就用預設值)

**環境隔離**:LangChain/LangGraph 依賴樹較重,另開 `requirements-langgraph.txt`(或獨立 venv),避免污染階段 1–5 的乾淨環境。

**Note(實測結果)**:`06_langgraph.py` 加了每個 node 的計時(`NODE_TIMINGS` + `_timed` 裝飾器),實際各跑一次序列版(`python3 06_langgraph.py`)和平行版(`--parallel`),結果:

| Node | 序列版 | 平行版 |
|---|---|---|
| food_critic | 4.51s | 4.21s |
| weather | 0.00s | 0.00s |
| bus | 0.00s | 0.00s |
| synthesize | 4.31s | 5.08s |
| **total** | **8.82s** | **9.29s** |

平行版並沒有比較快,差異落在誤差範圍內。原因:`weather`/`bus` 是純本地查詢(~0 秒),跟耗時的 `food_critic`(呼叫一次 LLM)平行執行沒有意義——三個 node 裡只有一個慢,把不慢的兩個跟它平行化省不了時間。真正的瓶頸是兩次必經、彼此有依賴關係的 LLM 呼叫(`food_critic` → `synthesize`),不管 `weather`/`bus` 的 edge 怎麼排都繞不開。**結論:平行化不是萬靈丹,要先確認瓶頸是不是真的分散在多個平行分支上,而不是集中在單一 node。**

---

## 建議節奏

| 階段 | 預估時間 | 前置 |
|---|---|---|
| 0 資料集 | 半天 | — |
| 1 CoT | 半天 | 0 |
| 2 ReAct | 1–2 天 | 1 |
| 3 RAG | 1–2 天 | 0 |
| 4 Subagent | 2–3 天 | 2+3 |
| 5 A2A | 2–3 天 | 4 |
| 6 LangGraph | 1–2 天 | 2+4 |

原則:每階段先不用框架(LangChain 等)手刻一次,理解機制後,想換框架再換——做中學的重點是看見裸的機制。階段 6 就是「換框架」這一步,對照才是重點,不是重新發明功能。
