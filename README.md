# agent-react

以《食尚玩家》介紹過的店家為素材，**手刻**一條從 prompt 到 multi-agent 的學習路徑：CoT → ReAct → RAG → Subagent → A2A。刻意不用 LangChain 等框架，目的是看見每個機制的裸實作。

完整練習規劃見 [PLAN.md](PLAN.md)；各階段實際跑出來的結果與機制筆記見 [NOTES.md](NOTES.md)。

## 五階段一覽

| 階段 | 檔案 | 核心概念 |
|------|------|----------|
| 0 | `data/` | 結構化店家資料 + 非結構化介紹文 |
| 1 | [`01_cot.py`](01_cot.py) | Chain-of-Thought：先推理、後結論 |
| 2 | [`02_react.py`](02_react.py) | ReAct：Thought → Action → Observation 工具迴圈 |
| 3 | [`03_rag.py`](03_rag.py) | RAG：TF-IDF 檢索 + 有/無文件對照 |
| 4 | [`04_subagent.py`](04_subagent.py) | Subagent：orchestrator 委派三位專才（context 隔離） |
| 5 | [`05_a2a_food_agent.py`](05_a2a_food_agent.py) · [`05_a2a_travel_agent.py`](05_a2a_travel_agent.py) | A2A：兩個 HTTP 服務，agent card 動態發現能力 |

## 環境設定

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

在專案根目錄建立 `.env`：

```bash
ANTHROPIC_API_KEY=sk-ant-...
# 選填，預設 claude-opus-4-8
ANTHROPIC_MODEL=claude-haiku-4-5
```

## 執行方式

### 階段 1–4（單支腳本）

```bash
python3 01_cot.py
python3 02_react.py
python3 03_rag.py
python3 04_subagent.py
```

### 階段 5（兩個服務）

```bash
# terminal 1：旅遊資訊 agent（port 8500）
python3 05_a2a_travel_agent.py

# terminal 2：美食 agent 測試題
python3 05_a2a_food_agent.py test
```

測試題：「這週末去台南吃食尚玩家名店，會下雨嗎？怎麼搭車？」

美食 agent 會從旅遊 agent 的 `GET /agent-card` 動態取得天氣、公車能力，再透過 `POST /tasks` 委派；本地則用階段 3 的 RAG 回答美食問題。

### 資料驗證

```bash
python3 scripts/validate_data.py
```

## 專案結構

```
agent-react/
├── 01_cot.py              # 階段 1：CoT A/B 對照
├── 02_react.py            # 階段 2：手刻 ReAct + 三個工具
├── 03_rag.py              # 階段 3：TF-IDF RAG + A/B/C 對照
├── 04_subagent.py         # 階段 4：行程總監 + 三位子 agent
├── 05_a2a_food_agent.py   # 階段 5：美食 agent（客戶端）
├── 05_a2a_travel_agent.py # 階段 5：旅遊 agent（服務端）
├── data/
│   ├── shops.json         # 結構化店家（工具查詢用）
│   ├── docs/*.md          # 節目介紹摘要（RAG 用）
│   ├── weather.json       # 階段 5 模擬天氣
│   └── bus_routes.json    # 階段 5 模擬公車路線
├── scripts/
│   └── validate_data.py
├── PLAN.md                # 練習計畫與驗收標準
└── NOTES.md               # 機制解說與實測筆記
```

## 資料設計

兩種資料分工（見 [data/README.md](data/README.md)）：

- **`shops.json`** — 結構化欄位（`hours`、`lat/lng`、`price_range`）給 ReAct 工具與預算控管
- **`data/docs/*.md`** — 自由文字介紹給 RAG 語意檢索

同一個「台南美食 AI」問題，有時該用結構化查詢，有時該用文件檢索——資料 schema 本身就是練習的一部分。

## 延伸閱讀

- [PLAN.md](PLAN.md) — 各階段目標、測試題、驗收標準
- [NOTES.md](NOTES.md) — TF-IDF vs embedding、subagent 仲裁、A2A agent card、產品化風險等筆記
- [Google A2A 規格](https://google.github.io/A2A/) — 對照階段 5 簡化協定（狀態機、串流、認證）

## License

學習用途專案；食尚玩家相關內容請遵守原報導出處與合理使用範圍。
