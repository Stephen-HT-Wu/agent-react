# 練習筆記：階段 1（CoT）與階段 2（ReAct）

這份文件記錄兩個練習實際做完後的結果，不是預期會發生什麼、是真的發生了什麼。
規劃緣由與後續階段見 [PLAN.md](PLAN.md)；資料集見 [data/README.md](data/README.md)。

---

## 階段 1：CoT（Chain-of-Thought）—— [01_cot.py](01_cot.py)

### 這個練習在測什麼

同一個台南一日遊排程問題，問兩次：

- **A 版**：直接把店家資料和需求丟給模型，不引導任何推理過程
- **B 版**：要求模型先逐項檢查（營業時間 → 公休日 → 動線 → 預算），再輸出結論

CoT 的核心機制是**自回歸生成**：模型每個 token 都基於前面已生成的內容繼續推。
直接要答案，模型會一步跳到結論；要求先寫推理步驟，那些中間步驟就變成生成
最終答案時「看得到」的上下文——讓錯誤有機會在半路被攔截，而不是悶頭生成一個
自信但邏輯有洞的答案。

刻意**不開 API 的 `thinking` 參數**：現在的模型有內建推理能力，開了它會替你
完成推理，A/B 的差異就會被模型自己的隱藏推理蓋過，看不出「prompt 引導出的
推理」本身有沒有用。

### 程式碼對照：CoT 在哪裡

CoT **沒有程式邏輯**，它就是 prompt 裡的一段文字——這是這個技巧的本質，
也是跟階段 2 最大的差別：

| 位置 | 對應 |
|---|---|
| [`build_prompt_a()`](01_cot.py) 第 56-64 行 | 無 CoT：只有店家資料 + `TASK`，沒有任何推理引導字句 |
| [`build_prompt_b()`](01_cot.py) 第 67-83 行 | CoT：第 69 行 `"先推理、後結論。"` + 第 72-78 行的五條編號檢查清單，就是 CoT 技巧的全部 |
| [`ask()`](01_cot.py) 第 86-94 行 | 兩版都只呼叫**一次** `client.messages.create()` |

沒有「檢查 `stop_reason`」，也沒有「下一輪」——因為根本不需要。CoT 的推理和
結論在**同一次生成**裡就完成了，`ask()` 拿到回應直接抽出 text block 回傳，
不會有第二次 API 呼叫。這是判斷「這是不是 CoT」的關鍵訊號：**只要整個技巧
在一次 `messages.create()` 呼叫裡就講完，就是 CoT；需要迴圈、需要看
`stop_reason` 決定下一步的，就是階段 2 的 ReAct。**

### 埋的陷阱

1. 阿堂鹹粥 `04:30-12:00`（賣完收攤）——排到晚餐就是錯的
2. 莉莉水果店週三公休——TASK 指定「這個週三出發」，若還排進去，代表沒有
   真的把公休日資訊拿來跟出遊日期核對，只是把它當裝飾性提醒

### 實際跑出來的結果

跑了三次（`claude-opus-4-8` 兩次、`claude-haiku-4-5` 一次），**A 版三次全部答對**，
沒有踩到任何一個陷阱——營業時間都排對、公休日都正確排除、預算加總都正確。

### 結論

**這組資料集（5 家店、單一縣市）的複雜度低於現行任何一個 Claude 模型的能力
閾值，連最小的 Haiku 都靠常識排對，CoT 沒有機會展現差異。** 這本身是一個
有效的發現，而不是練習失敗：

> CoT 的邊際效益會隨任務複雜度下降、模型能力上升而遞減。

要真的逼出 A 版犯錯，需要把任務複雜度拉高（全部 20 家店、跨縣市、兩天一夜），
但這個練習真正要練的是「理解 CoT 為什麼有效」的機制性理解，而不是一定要抓到
一個具體失敗案例——這點已經達成，所以在這裡收尾，往階段 2 走。

### 跑法

```bash
source venv/bin/activate
python3 01_cot.py
```

---

## 階段 2：ReAct（Reason + Act）—— [02_react.py](02_react.py)

### 這個練習在測什麼

手刻 agent loop（刻意不用 LangChain 之類的框架——目的是看見 Thought → Action →
Observation 這個機制的裸實作，知道框架幫你省了什麼，而不是把它當黑盒子用）。

三個工具，`call_tool()` 統一派發：

| 工具 | 功能 |
|---|---|
| `search_shops(city, dish_keyword)` | 依城市（和可選關鍵字）查店家清單 |
| `is_open(shop_name, time)` | 查某店在指定時間是否營業（正確處理跨午夜營業，如 17:00-01:00） |
| `distance(point_a, point_b)` | 算兩地直線距離（haversine），支援店名和「逢甲夜市」這種地標 |

測試題：「現在晚上 11 點在逢甲，食尚玩家介紹過的宵夜哪家還開著而且最近？」

這題跟階段 1 的陷阱性質不同——是**事實陷阱**而不是**推理陷阱**：CoT 再會
推理也生不出模型本來就不知道的事實（幾點、多遠），這種資訊只能靠工具查。

### 程式碼對照：Block、Thought/Action/Observation、檢查與下一輪在哪裡

**Block 是什麼**：API 回應（`response.content`）不是一整條字串，是一個
**清單**，清單裡每個元素叫一個 block，各自有 `type`。這次練習只會遇到：

| `block.type` | 誰產生的 | 在程式碼裡的意義 |
|---|---|---|
| `"text"` | 模型 | 模型自己寫的文字。**跟 `[Thought]`／`[Final Answer]` 不是一對一對應**——兩者用的是同一種 block，差別只在「印出來的當下，這輪是不是要呼叫工具」（見下方檢查點） |
| `"tool_use"` | 模型 | 模型決定呼叫某個工具，帶 `block.name`（工具名）和 `block.input`（參數）→ `[Action]` |
| `"tool_result"`（不是模型給的） | **你自己組出來**送回去給模型 | 工具實際執行的結果 → `[Observation]`，本質上跟前兩種是同一個 block 概念，只是方向相反 |

一次回應的 `response.content` 常常同時裝著一段 Thought 文字 + 一或多個
tool_use，這就是為什麼 [`run_react_loop()`](02_react.py) 第 220-222、
233-234 行要用 `for block in response.content:` 逐一檢查 `block.type`，
而不是直接假設回應只有一種東西。

**檢查點（Reason 完，決定要不要 Act）**：

```python
# 02_react.py 第 224 行
if response.stop_reason != "tool_use":
```

`stop_reason` 是 API 回應裡的欄位。值是 `"tool_use"`，代表模型這輪認為
資訊不夠，決定呼叫工具；不是 `"tool_use"`（通常是 `"end_turn"`），代表模型
判斷資訊已經夠回答了。**這一行就是 ReAct 迴圈裡「Reason → 決定要不要 Act」
的分岔點**——階段 1 的 CoT 完全沒有這種判斷，因為它只有一次生成、不需要決定
「這輪夠不夠」。

**下一輪（Observation 餵回下一次 Reason）**：

```python
# 02_react.py 第 230-244 行
messages.append({"role": "assistant", "content": response.content})   # 把這輪模型說的話存進歷史
tool_results = []
for block in response.content:
    if block.type == "tool_use":
        ...
        tool_results.append({...})       # 組出 tool_result block（= Observation）
messages.append({"role": "user", "content": tool_results})            # 送回去
# 迴圈跑到這裡結束，回到第 214 行 for turn in range(max_turns): 的下一次
```

這段做完之後，`for turn in range(max_turns):`（第 214 行）自然進入下一次
迭代，帶著剛剛累積的 `messages` 歷史（包含這輪的 Thought/Action 和剛組好的
Observation）重新呼叫一次 `client.messages.create()`——模型在下一輪的
Thought，就是基於這個 Observation 繼續推理。**這一整圈（呼叫 API → 檢查
`stop_reason` → 執行工具 → 組 Observation → append 進歷史 → 迴圈重來）
就是 ReAct 手刻迴圈的全部**，跟 CoT 的「一次呼叫講完」正好相反。

### 手刻迴圈踩過的 bug（過程本身就是重點）

寫第一版時把整段步驟說明直接塞進 `raise NotImplementedError("...")`
的字串參數裡——這是純文字，函式被呼叫時還是會直接丟例外，不會真的執行。
（這跟階段 1 寫 `build_prompt_a/b` 時犯的是同一種錯：把「計畫」和「會執行的
程式碼」搞混。）

翻成真正的邏輯後，還踩了三個會讓程式掛掉或行為錯誤的 bug：

1. **`anthropic.messages.create()` 應該是 `client.messages.create()`**——
   `anthropic` 是匯入的 SDK 模組本身，不是客戶端實例
2. **`tool_result` 沒包成 API 要求的格式，也沒帶 `tool_use_id`**——
   少了這個 ID，模型不知道這個結果對應到哪一次工具呼叫，下一輪 API 呼叫會 400
3. **多個工具呼叫各自 append 一則訊息，而不是彙整成一則**——
   Claude 預設允許一輪呼叫多個工具（parallel tool use），所有對應的
   `tool_result` 必須包在同一則 user 訊息裡送回去

修完後又意外掉了 `client = anthropic.Anthropic()` 這行初始化（改的時候被
連帶刪掉），補回去才真正跑通。

### 實際跑出來的結果

跑通後產生了完整的 Thought/Action/Observation 軌跡，而且發生了兩件超出
原本設計的事：

**1. 自我修正是真實發生的，不是手動注入的**

模型第一次呼叫 `distance()` 時自己猜了 `point_a="逢甲"`（而不是資料裡登記的
「逢甲夜市」），工具丟出 `LookupError`，錯誤訊息裡帶了 fuzzy match 建議
「你是不是要找『逢甲夜市』？」；模型收到這個 Observation 後，下一輪馬上改用
正確的地標名稱重查——完整驗證了驗收標準要求的「工具報錯後 agent 會不會自我
修正」，而且是模型自己犯的錯、自己看著錯誤訊息修好的，比人工故意打錯字更有
說服力。

**2. 最終答案是錯的，但錯誤的原因很有教學價值**

Agent 一開始就用 `dish_keyword="逢甲"` 縮小搜尋範圍，只撈到店名/餐點/備註
欄位裡剛好寫著「逢甲」兩個字的 3 家店（官芝霖、逢甲紅燒當歸鴨、丸南），
完全沒發現旺伯臭豆腐和米丹 MiDan 也在同一個商圈——因為它們的備註寫的是
「文華路5巷」「福上巷」，沒有字面上的「逢甲」。

實際核對五家店在晚上 11 點的狀態：

| 店名 | 23:00 開著 | 距逢甲夜市 |
|---|---|---|
| **旺伯臭豆腐** | ✅ | **0.13 km（真正最近）** |
| 官芝霖大腸包小腸 | ✅ | 0.19 km（agent 最終選的答案） |
| 逢甲紅燒當歸鴨 | ✅ | 0.26 km |
| 米丹MiDan | ✅ | 0.32 km |
| 丸南生魚片 | ❌（半夜12點才開，11點還沒到） | 0.65 km |

Agent 正確避開了「丸南生魚片」這個知名度陷阱（有確實呼叫 `is_open()` 驗證，
沒有單憑「傳奇宵夜」的印象就選它），但選出的「官芝霖」不是真正最近的——
**迴圈機制完全正確，錯的是工具呼叫的搜尋策略**：候選集合在第一步就被錯誤
地縮小了，後面所有推理都是在不完整的候選池裡做對的事。

### 結論

> ReAct 解決的是「模型不知道」的問題（給它工具去查即時資訊），
> 但不會自動解決「模型問錯問題」的問題（工具呼叫的參數、搜尋策略下錯，
> 一樣會導出錯誤答案）。

這比原本設計的「丸南陷阱」更完整地回答了驗收標準最後一題——ReAct 跟階段 1
CoT 的差別：CoT 只能重新排列模型已經知道的資訊；ReAct 讓模型能主動取得新
資訊，但工具本身的設計（例如 `search_shops` 用關鍵字比對而非地理範圍查詢）
還是會限制 agent 能查到什麼。

### 跑法

```bash
source venv/bin/activate
python3 02_react.py
```

想重現「工具報錯後自我修正」的行為，也可以不透過完整迴圈、直接單獨測試工具的
錯誤處理：

```bash
python3 -c "
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location('r', '02_react.py')
m = module_from_spec(spec); spec.loader.exec_module(m)
print(m.call_tool('is_open', {'shop_name': '旺伯臭都腐', 'time': '23:00'}))
"
```

### 已知的後續改進方向（沒有動手做，留給有興趣時再試）

把 `SYSTEM_PROMPT` 改成引導模型優先用 `city="台中"`（不加關鍵字）撈出全部
候選再篩選，看看修正搜尋策略後答案會不會變成正確的旺伯臭豆腐。這也是階段 4
（Subagent）會遇到的問題的縮影：工具設計得好不好，直接決定 agent 能不能查到
對的答案。
