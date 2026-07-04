# 練習筆記：階段 1（CoT）、階段 2（ReAct）、階段 3（RAG）、階段 4（Subagent）

這份文件記錄各階段練習的機制解說與實際跑出來的結果（不是預期會發生什麼、是真的發生了什麼）。
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

---

## 階段 3：RAG（Retrieval-Augmented Generation）—— [03_rag.py](03_rag.py)

### RAG 的底層邏輯

RAG 不管用什麼技術實作，底層都是兩步：

```
1. Retrieval（檢索）  問題 → 從知識庫找出最相關的幾段
2. Generation（生成）  把撈到的內容塞進 prompt → 叫 LLM 回答
```

展開來就是 **Retrieve → Augment → Generate**：

> **先找相關片段，再讓模型根據資料回答**——不是讓模型憑記憶瞎猜。

[`03_rag.py`](03_rag.py) 對應關係：

| RAG 步驟 | 程式碼 |
|---|---|
| 知識庫 | `data/docs/*.md` → [`load_all_docs()`](03_rag.py) |
| 向量化 | [`tokenize()`](03_rag.py) + [`build_tfidf_index()`](03_rag.py) |
| 檢索 | [`retrieve_top_k()`](03_rag.py) + [`cosine_similarity()`](03_rag.py) |
| 組 prompt | [`build_prompt_retrieved()`](03_rag.py) |
| 生成 | [`ask()`](03_rag.py) → Claude |

### 這個練習在測什麼

用 `data/docs/*.md`（20 篇店家介紹文）做三種版本對照：

| 版本 | 函式 | 給模型的資料 |
|---|---|---|
| A 版 | [`build_prompt_no_context()`](03_rag.py) | 完全不給文件（幻覺測試基準線） |
| B 版 | [`build_prompt_full_context()`](03_rag.py) | 假 RAG——全部 20 篇整包塞進 prompt |
| C 版 | [`build_prompt_retrieved()`](03_rag.py) | 真 RAG——只塞 `retrieve_top_k()` 撈到的 top-3 |

測試題：「食尚玩家介紹過的碗粿老店裡，哪一家只賣碗粿和魚羹兩種東西、均一價35元？出自食尚玩家的哪篇報導？」

正解：富盛號碗粿，出自〈台南碗粿內行老饕必吃５家〉。「均一價35元」和「哪篇報導」都是
只有讀過文件才答得出來的細節——沒有文件的話，模型要嘛編一個聽起來合理的價格和篇名
（幻覺），要嘛老實說不知道。

### B 版 vs C 版：假 RAG 與真 RAG 的差別

B 版和 C 版的 prompt 組裝邏輯幾乎一樣，**唯一差別是資料來源**：

```python
# B 版：全部文件
doc_content = "\n".join(
    f"### {name}\n{text}" for name, text in all_docs.items()
)

# C 版：只取檢索結果（retrieved = [(店名, 相似度分數), ...]）
doc_content = "\n\n".join(
    f"### {name}\n{all_docs[name]}" for name, score in retrieved
)
```

`main()` 裡 C 版的完整流程：

```python
retrieved = retrieve_top_k(TEST_QUESTION, k=3)   # ① 檢索
print(ask(build_prompt_retrieved(TEST_QUESTION, retrieved, ALL_DOCS)))  # ② 組 prompt + 生成
```

若 C 版跟 B 版答案一致，代表 top-3 沒有漏掉關鍵文件；若不一致，代表檢索出了問題。

### 檢索機制：從文字到向量（TF-IDF）

在算餘弦相似度之前，先把文字變成向量。這裡刻意用手刻的 **TF-IDF**（不是神經網路
embedding），因為文件只有 20 篇，重點是看見「查詢和文件都變成向量、算相似度、排序
取前幾名」這個檢索機制本身。

#### 1. `tokenize()`：切成 bigram

沒有斷詞函式庫，就兩個字一組當「詞」：

```
"富盛號碗粿" → ["富盛", "盛號", "號碗", "碗粿"]
```

簡單，但「早上生意」和「清晨五點半」不會有字重疊——這就是額外挑戰題會撈錯的原因。

#### 2. TF（Term Frequency）：這個詞在這篇裡多常出現

```
TF("碗粿", 富盛號文件) = 「碗粿」出現次數 / 這篇總 bigram 數
```

出現越多次，這篇跟這個詞越相關。

#### 3. IDF（Inverse Document Frequency）：這個詞有多「稀有」

```python
idf = math.log(文件總數 / 出現過這詞的文件數) + 1
```

- 「碗粿」很多店都有 → IDF 低（不太能區分）
- 「均一價」只有少數文件有 → IDF 高（很有區分力）

#### 4. TF-IDF = TF × IDF

每篇文件變成一個 **dict**（稀疏向量），key 是 bigram，value 是權重：

```python
DOC_VECTORS["富盛號碗粿"] = {
    "碗粿": 0.15,
    "魚羹": 0.12,
    "均一": 0.08,
    ...
}
```

問題也會用同樣方式變成 `query_vec`（[`embed_query()`](03_rag.py)），跟文件向量在
**同一個向量空間**裡比較。

### 餘弦相似度：在量什麼？

[`cosine_similarity()`](03_rag.py) 第 130-138 行：

```python
def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

公式：

\[
\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|}
\]

想像兩個向量是兩支從原點指出去的箭頭：

```
        文件 B
          ↗
         /
        /  θ（夾角）
       /________→ 問題
      文件 A
```

**餘弦相似度 = cos(θ)**，看的是**方向**像不像，不是長度。

| 夾角 | cos(θ) | 意思 |
|---|---|---|
| 0°（同方向） | 1.0 | 非常像 |
| 90°（垂直） | 0.0 | 無關 |

為什麼用餘弦、不用直接把權重加總？長文章 bigram 多，向量「比較長」；餘弦會把
長度除掉，只比**哪些詞重要、方向是否一致**，避免「字多就贏」。

逐行對照：

| 程式碼 | 在做什麼 |
|---|---|
| `common = set(vec_a) & set(vec_b)` | 兩個向量**共同有的 bigram**（稀疏向量只存非零維度） |
| `dot = sum(...)` | **點積**：共同詞的權重相乘再相加，共同的重要詞越多分數越高 |
| `norm_a`, `norm_b` | 向量的**長度**（L2 norm） |
| `return dot / (norm_a * norm_b)` | 點積除以兩個長度 → 0～1 之間的分數 |

### `retrieve_top_k()`：把檢索串起來

```python
def retrieve_top_k(query: str, k: int = 3) -> list[tuple[str, float]]:
    query_vec = embed_query(query, IDF)                                    # 問題 → 向量
    scores = [(name, cosine_similarity(query_vec, vec))                   # 跟每篇文件比
              for name, vec in DOC_VECTORS.items()]
    scores.sort(key=lambda x: x[1], reverse=True)                          # 由高到低排序
    return scores[:k]                                                      # 取 top-k
```

用測試題走一遍：

1. `tokenize` → `["碗粿", "老店", "魚羹", "均一", ...]`
2. `embed_query` → `query_vec`
3. 對 20 篇文件各算一次 `cosine_similarity`
4. **富盛號碗粿** 有「碗粿」「魚羹」「均一」等重疊 → 分數高
5. `sort` 後取 top-3

### 額外挑戰題：TF-IDF 的天生弱點

問題：「哪一家嘉義雞肉飯**只做早上生意**，**去晚一點就撲空**？」

正解是阿溪火雞肉飯（05:30-13:00）。[`data/docs/阿溪火雞肉飯.md`](data/docs/阿溪火雞肉飯.md)
寫的是「清晨五點半就開賣、下午一點打烊」「晚來就賣完」「早上限定」——人讀得懂
「只做早上生意」≈ 清晨開、下午打烊，「去晚一點就撲空」≈ 晚來吃不到；但這題刻意
用同義不同字面的說法問，跟文件實際用字完全沒有重疊：

| 問法用的字 | 文件用的字 | TF-IDF 能 match 嗎？ |
|---|---|---|
| 早上生意 | 清晨、五點半、早上限定 | ❌ 「早上生意」這組 bigram 不存在於文件 |
| 去晚一點撲空 | 晚來、賣完 | ❌ 字不同 |

同時，問題裡的「嘉義」「雞肉飯」會跟**很多**嘉義雞肉飯文件重疊——民主、郭家、阿霞
都會被拉高，阿溪反而因為「早鳥、清晨五點半」這種獨特用字，跟問題的用字對不上。

結果 `retrieve_top_k()` 撈到的 top-3 是民主、郭家、阿霞，阿溪火雞肉飯直接沒進榜。

為什麼同一件事換個說法問就找不到？可以拆成三層：

```
1. 餘弦相似度     → 只比向量方向像不像（數學本身沒問題）
2. TF-IDF 向量    → 向量來自「哪些詞出現」，不是「意思是什麼」
3. bigram tokenize → 連詞都切得很碎，字面重疊更難發生
```

**不只是 bigram 太僵化，而是整個 TF-IDF 檢索本來就不是語意檢索。** bigram 讓同義
改寫更難 match，但就算換成更好的中文斷詞，沒有 embedding 就還是不懂 paraphrase
（換句話說）。

### TF-IDF 不是真正的 Embedding

TF-IDF 比較像「用統計規則把文字變成數字」，不是「從大量語料裡學出語意」。

| | TF-IDF | 真正的 Embedding |
|---|---|---|
| 怎麼來的 | 手刻公式（詞頻 × 逆文件頻率） | 神經網路從大量文字**學出來** |
| 向量型態 | 稀疏（幾萬維，大多數是 0） | 稠密（例如 384、1536 維，每維都有值） |
| 每一維代表什麼 | 一個**具體的詞/bigram** | 沒有固定人類可讀意義，是抽象特徵 |
| 「清晨」和「早上」 | 兩個完全不同的維度 | 向量方向接近 |
| 懂同義改寫嗎 | ❌ | ✅（程度取決於模型） |

TF-IDF 是**詞袋模型（bag-of-words）**：文件 ≈ 一袋詞，每個詞一個權重。它不知道
「清晨」≈「早上」、「賣完」≈「撲空」、「只做早上生意」≈「五點半開賣、下午打烊」。
餘弦相似度只是在兩袋詞裡找**共同 key 的權重乘積**——沒有共同 key，語意再像也沒用。

#### 向量空間的直覺對照

TF-IDF 的世界（每個詞一根獨立軸）：

```
「清晨」軸 ████
「早上」軸 ████        ← 兩根完全不同的軸，餘弦相似度 = 0
「賣完」軸 ████
「撲空」軸 ████
```

Embedding 的世界（語意相近的東西聚在一起）：

```
        · 清晨五點半開賣
       · 只做早上生意          ← 這群聚在一起
      · 早上限定

                          · 深夜十二點才開門  ← 離很遠
```

### 真正的 Embedding 怎麼學出語意相似度？

核心想法：

> **在真實文字裡，意思相近的詞/句子，會出現在相似的上下文裡。**

模型透過大量閱讀，把「常一起出現、用法類似」的東西映射到向量空間裡相近的位置——
不是有人告訴它「清晨=早上」，是它從用法裡自己推出來的。

#### 詞級：Word2Vec（Skip-gram，2013）

```
給模型看：「我每天早上吃 ___ 」
任務：預測旁邊缺的那個詞
```

讀了幾十億句之後，「清晨」「早上」「黎明」常出現在同樣的句子結構裡，「賣完」
「撲空」「吃不到」也常出現在類似情境——這些詞的向量會被調整到方向接近。

#### 句級 / 文件級：現代 RAG 常用（Transformer）

現代 RAG 多半把整句或整段變成一個向量（OpenAI embedding、Voyage、BGE 等）。
常見訓練方式：

| 方法 | 在做什麼 |
|---|---|
| **對比學習** | 語意相近的配對（「清晨五點半開賣」↔「只做早上生意」）拉近；不相近的推開 |
| **遮罩語言模型**（BERT 類） | 遮住某些詞叫模型猜回來；要猜對必須理解上下文語意 |
| **檢索任務微調** | 直接用「問題 ↔ 相關文件」配對微調，專門為 RAG 檢索優化 |

#### 換成 Embedding 後，程式架構不變

RAG 骨架（檢索 → 塞 prompt → 生成）不變，只換「文字怎麼變向量」這一層：

```python
# 現在（TF-IDF）
query_vec = embed_query(query, IDF)

# 實務（神經網路 embedding）
query_vec = voyage_client.embed(query)   # 或 openai.embeddings.create(...)
```

檢索準不準，取決於 embedding 懂不懂語意；[`cosine_similarity()`](03_rag.py) 這層
數學通常還是用餘弦，換的是向量從哪裡來。

### 練習版 vs 實務版 RAG

你練的是**機制本身**；實務會換更強的零件，但流程不變：

| 環節 | 這個練習版 | 實務常見版 |
|---|---|---|
| 切塊 | 一篇 doc = 一家店 | 長文件切成很多 chunk |
| 向量化 | TF-IDF（字面相似） | Embedding 模型（語意相似） |
| 相似度 | 餘弦相似度 | 多半還是餘弦，或點積 |
| 儲存 | 記憶體裡的 dict | 向量資料庫（Pinecone、Chroma…） |
| 檢索量 | top-3 | top-k，常加 rerank |
| 生成 | 一次 `ask()` | 同上，或接進 agent loop |

**餘弦相似度**在實務裡也很常見；常換的是 **TF-IDF → 神經網路 embedding**。

### 跟前面階段的關係

```
Stage 1 CoT   → 模型在腦內推理（沒有外部資料）
Stage 2 ReAct → 模型呼叫「結構化工具」（is_open、distance、search_shops）
Stage 3 RAG   → 模型拿到「非結構化文件」裡撈出來的片段
```

RAG 跟階段 2 的 `search_shops` 差在哪？

| | `search_shops`（階段 2） | RAG（階段 3） |
|---|---|---|
| 查詢方式 | 結構化——精確比對 `city`、`dish_keyword` 欄位 | 語意檢索——不知道確切欄位，只知道問題大意 |
| 資料型態 | `shops.json` 的結構化欄位 | `data/docs/*.md` 的自由文字 |
| 典型問題 | 「台中有哪些宵夜？」 | 「只做早上生意、去晚就撲空的是哪家？」 |

`search_shops(city="嘉義")` 查得到嘉義的店，但查不到「只做早上生意、去晚就撲空」
這種**語意描述**——這就要 RAG。

延伸方向：把 `retrieve_top_k()` 包成 ReAct loop 裡的第四個工具，變成 **agentic RAG**
（模型自己決定什麼時候該檢索文件）。

### 跑法

```bash
source venv/bin/activate
python3 03_rag.py
```

手動測試檢索（不用呼叫 API）：

```bash
python3 -c "
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location('r', '03_rag.py')
m = module_from_spec(spec); spec.loader.exec_module(m)
q = '哪一家嘉義雞肉飯只做早上生意，去晚一點就撲空？'
print(m.retrieve_top_k(q))
"
```

---

## 階段 3 延伸：從有限文件到自然語言查詢

RAG（再加上後面的 agent）做的是這層轉換：

```
之前：20 篇 md（人要自己翻、自己找）
之後：自然語言問 → 檢索相關段落 → LLM 組成回答
```

| 以前 | RAG 之後 |
|---|---|
| 記得文件寫什麼才能答 | 「碗粿均一價 35 元是哪一家？」直接問 |
| 一篇一篇讀 | 問「逢甲宵夜傳奇店」會撈到丸南生魚片 |
| 搜尋靠關鍵字 | 問法可以口語、很長、很多種變化 |

**「無限查詢」**指的是問法可以一直變，不是答案無限多——能答的範圍仍被那批資料框住；
檢索撈錯或模型亂編時還是會錯（階段 3 A 版就是在對照「沒有這層轉換」時的幻覺）。

到階段 4，使用者只要說「兩天一夜台中美食，預算 3000，不騎車」，背後仍是有限篇數的
介紹文 + `shops.json`，但不必自己讀完、加總、排路線。

---

## 階段 4：Subagent—— [04_subagent.py](04_subagent.py)

### 這個練習在測什麼

一個 **orchestrator（行程總監）** 委派三位子 agent，各自獨立 context、獨立工具集：

| 子 agent | 工具 | 職責 |
|---|---|---|
| 美食評論員 | `retrieve_food_docs`（RAG，只索引台中 docs） | 選店、推薦理由（**不知道預算**） |
| 預算控管 | `get_taichung_price_table` / `estimate_food_cost` / `check_budget` | 檢查花費（總預算 3000，先扣住宿+交通 2000） |
| 交通規劃師 | `distance` / `estimate_transit` / `plan_route` | 從台中車站出發排動線（不騎車） |

**Context 隔離**：[`run_subagent()`](04_subagent.py) 每次呼叫都新建自己的 `messages`；
子 agent 互相看不到對話。Orchestrator 只傳各 agent 的「最終報告」和抽出的店名清單，
不傳 tool 軌跡。

### Orchestrator 流程

```
① 委派美食評論員（不知預算）
② 委派預算控管（只收到店名清單）
③ 若餐飲預估 > 1000 → 衝突仲裁，請評論員替換最貴的店（如丸南）
④ 委派交通規劃師（只收到最終店名）
⑤ 行程總監彙整兩天一夜行程
```

刻意設計的衝突：評論員若把**丸南生魚片**排第一，5 家店輪流吃 6 餐會讓丸南吃兩次
（價格上限 300×2），餐飲預估 1050 > 剩餘 1000 → 觸發仲裁。

### 強模型還需要 Subagent 嗎？

商用強模型（ChatGPT、Claude 等）讓「**單一 agent + 多工具**」能覆蓋更多場景，
但 subagent 解的不只是「模型不夠聰明」：

| 問題 | 強模型能改善嗎？ | Subagent 的價值 |
|---|---|---|
| 推理與規劃 | 通常較好 | — |
| 工具 ≤ 5、任務單一 | 單 agent 常夠用 | — |
| Context 汙染（RAG 片段、失敗 tool 堆滿歷史） | 部分改善 | 子 agent 只回摘要 |
| 工具太多選錯 | 部分改善 | 強制縮小各 agent 的工具集 |
| 權限／資料隔離 | 不能 | 預算 agent 不該看到完整 RAG 軌跡 |
| 平行化、分工維護 | 不能 | 美食 ∥ 天氣、模組化 |

> Subagent 是**架構選擇**，不是模型能力的補品。任務簡單用單 agent；多專家、要隔離、
> 工具很多時，仍值得拆。

### 跑法

```bash
source venv/bin/activate
python3 04_subagent.py
```

---

## 產品化風險：服務會被競爭對手「蒸餾」嗎？

這類 RAG / agent 問答服務要**預設會被試著抄**——技術上可行，但能抄走多少取決於暴露了什麼。

### 常見手法

| 手法 | 在做什麼 |
|---|---|
| **API 蒸餾** | 大量呼叫問答 API，用「問題 → 你的回答」微調便宜模型 |
| **知識庫重建** | 系統性提問，從回答與 citation 拼回文件內容 |
| **流程複製** | 觀察工具設計、agent 分工、仲裁策略，自己重寫 |
| **直接爬資料** | 若回應或客戶端露出 `docs` 全文，乾脆下載 |

RAG 的本質是把資料塞進 prompt——若回答**逐字露出檢索段落**，等於在漏題庫。

### 本專案若上線，哪些特別危險？

- **文件內容**：C 版若回傳大段 `data/docs` 原文，問幾百次可能拼回 20 篇。
  （本專案素材多來自食尚玩家公開報導，機密價值有限，但「整理方式與 schema」仍有價值。）
- **問答行為**：行程怎麼仲裁、怎麼排，可被樣本蒸餾成「便宜版助手」，不必拿到原始碼。
- **較難直接抄的**：即時 API 合約、使用者偏好歷史、品牌信任、持續更新的獨家資料。

### 常見防護

| 層級 | 做法 |
|---|---|
| 輸出 | 摘要化回答；出處只給標題+連結，不全文貼 doc |
| API | Rate limit、認證、異常流量偵測 |
| 護城河 | 價值放在即時資料、交易閉環、UX，而非靜態可被問出來的文字 |

強模型讓對手用**更少樣本**就能蒸餾出夠用的仿品——防線是限制暴露 + 把核心價值放在
知識庫以外的層次，而不是假裝 RAG 包一層就沒人學得走。

