"""階段 3：RAG（Retrieval-Augmented Generation）練習。

概念：把文件切塊 → 向量化 → 依問題檢索最相關的幾塊 → 塞進 prompt 回答。

用 data/docs/*.md（20 篇食尚玩家店家介紹文）做三種版本的對照：
  A 版：完全不給文件（幻覺測試基準線）
  B 版：假 RAG——全部 20 篇文件整包塞進 prompt（文件夠少才行得通）
  C 版：真 RAG——TF-IDF + 餘弦相似度，只撈 top-3 塞進 prompt

注意：Claude API 本身沒有 embedding 端點（Anthropic 官方建議搭配 Voyage AI 用
神經網路 embedding）。這裡刻意不引入新的 API/套件，改用手刻的 TF-IDF 當「向量化」
——文件只有 20 篇，數學不需要多精緻，重點是看見「查詢和文件都變成向量、算相似度、
排序取前幾名」這個檢索機制本身，而不是追求最好的語意理解。

測試題（PLAN.md 原題是「魚皮加蛋」，但這個細節沒有真的寫進 data/docs/，
已經換成資料裡實際存在的細節）：
  「食尚玩家介紹過的碗粿老店裡，哪一家只賣碗粿和魚羹兩種東西、均一價35元？
   出自食尚玩家的哪篇報導？」
  正解：富盛號碗粿，出自〈台南碗粿內行老饕必吃５家〉。
  這題要求的「均一價35元」和「哪篇報導」都是具體到只有讀過文件才答得出來的
  細節——沒有文件的話，模型要嘛編一個聽起來合理的價格和篇名（幻覺），
  要嘛老實說不知道。

額外挑戰題（已經實測驗證會撈錯，留給你自己手動重現）：
  「哪一家嘉義雞肉飯只做早上生意，去晚一點就撲空？」
  正解是阿溪火雞肉飯（05:30-13:00，文件裡寫「清晨五點半就開賣」「晚來就賣完」），
  但這題刻意用同義不同字面的說法問（「只做早上生意」「去晚一點就撲空」），
  跟文件實際用字（清晨、晚來、賣完）完全沒有重疊——結果 retrieve_top_k() 撈到的
  top-3 是民主、郭家、阿霞，阿溪火雞肉飯直接沒進榜。這就是 TF-IDF 的天生弱點：
  它只看字面上有沒有重疊的詞，換句話說完全不懂語意，只要問法換個說法就會失準。

執行前：
  pip install anthropic python-dotenv
  export ANTHROPIC_API_KEY=...   （或放進 .env）
執行：
  python3 03_rag.py

驗收標準：
  [ ] 做一次 ablation：A 版（沒檢索也不塞全文）應該出現幻覺或答不出來，
      B 版和 C 版應該答對且一致
  [ ] 用額外挑戰題手動測試 retrieve_top_k()，找出 TF-IDF 撈錯文件的案例，
      說明為什麼撈錯（提示：TF-IDF 只看字面重疊，不懂真正的語意）
  [ ] 能回答：RAG 跟階段 2 的 search_shops 工具差在哪？
      （提示：search_shops 是結構化查詢——精確比對 city/dish_keyword 欄位；
       RAG 是語意檢索——不知道確切欄位、只知道問題大意時用。
       其實 RAG 檢索也可以包成 ReAct 的一個工具，試試看把 retrieve_top_k()
       包成階段 2 loop 裡的第四個工具）

延伸：把 RAG 檢索包成工具接進階段 2 的 ReAct loop，變成 agentic RAG。
"""

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

DOCS_DIR = Path(__file__).parent / "data" / "docs"

TEST_QUESTION = "食尚玩家介紹過的碗粿老店裡，哪一家只賣碗粿和魚羹兩種東西、均一價35元？出自食尚玩家的哪篇報導？"
FOLLOW_UP_QUESTION = "哪一家嘉義雞肉飯只做早上生意，去晚一點就撲空？"


# ---------------------------------------------------------------------------
# 檢索機制（已經寫好，這階段的重點不是 TF-IDF 數學，是後面的 prompt 組裝）
# ---------------------------------------------------------------------------

_PUNCTUATION = set("，。、（）「」〈〉！？：；\n\t ")


def load_all_docs() -> dict[str, str]:
    """讀 data/docs/*.md 全部文件，回傳 {店名: 內文}。"""
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def tokenize(text: str) -> list[str]:
    """把中文字切成 bigram（兩字一組）當作詞——沒有斷詞函式庫，
    但對這個規模的檢索練習已經夠用，缺點留給你在額外挑戰題裡體會。
    """
    chars = [c for c in text if c not in _PUNCTUATION]
    if len(chars) < 2:
        return chars
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def build_tfidf_index(docs: dict[str, str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """對每篇文件算 TF-IDF 向量。回傳 (idf 表, {店名: 詞->權重 的向量})。"""
    doc_tokens = {name: tokenize(text) for name, text in docs.items()}
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


def embed_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    """用文件語料算出的 idf 表，把查詢也變成同一個向量空間裡的向量。"""
    tokens = tokenize(query)
    tf: dict[str, int] = {}
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1
    total = len(tokens) or 1
    return {term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """算兩個稀疏向量（用 dict 表示）的餘弦相似度。"""
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


ALL_DOCS = load_all_docs()
IDF, DOC_VECTORS = build_tfidf_index(ALL_DOCS)


def retrieve_top_k(query: str, k: int = 3) -> list[tuple[str, float]]:
    """回傳跟 query 最相關的 k 篇文件，[(店名, 相似度分數), ...]，由高到低排序。"""
    query_vec = embed_query(query, IDF)
    scores = [(name, cosine_similarity(query_vec, vec)) for name, vec in DOC_VECTORS.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def ask(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# Prompt 組裝（由你手刻——這是這階段真正的練習）
# ---------------------------------------------------------------------------

def build_prompt_no_context(question: str) -> str:
    """A 版：完全不給文件，測試模型會不會憑空回答（幻覺測試基準線）。

    TODO 由你實作：
    直接把 question 包成一句話的 prompt。可以加一句「如果不確定就誠實說不知道，
    不要編造答案」，方便觀察模型是選擇誠實還是選擇幻覺。
    """
    return f"請回答以下問題：{question}。如果不確定就誠實說不知道，不要編造答案。"


def build_prompt_full_context(question: str, all_docs: dict[str, str]) -> str:
    """B 版：假 RAG——把全部文件塞進 prompt（這階段的基準線，資料量小才行得通）。

    TODO 由你實作：
    把 all_docs 裡每篇文件的內容串起來（例如用「### 店名\\n內文」分隔每篇），
    加上 question，並要求回答附出處（哪家店、出自哪篇報導——報導名稱在每篇
    文件最後一行「出處：...」）。
    """
    
    doc_content = "\n".join([f"### {name}\n{text}" for name, text in all_docs.items()])
    return f"""請參考以下文件，回答以下問題：{question}。
    如果不確定就誠實說不知道，不要編造答案。
    回答要附出處（哪家店、出自哪篇報導——報導名稱在每篇
    文件最後一行「出處：...」""" + doc_content

def build_prompt_retrieved(
    question: str, retrieved: list[tuple[str, float]], all_docs: dict[str, str]
) -> str:
    """C 版：真 RAG——只把 retrieve_top_k() 撈到的文件塞進 prompt。"""
    doc_content = "\n\n".join(
        f"### {name}\n{all_docs[name]}" for name, _ in retrieved
    )
    return f"""請參考以下文件，回答以下問題：{question}
如果不確定就誠實說不知道，不要編造答案。
回答要附出處（哪家店、出自哪篇報導——報導名稱在每篇文件最後一行「出處：...」）。

{doc_content}"""

def main() -> None:
    print("=" * 60)
    print("A 版：沒有檢索、也不塞全文（幻覺測試基準線）")
    print("=" * 60)
    print(ask(build_prompt_no_context(TEST_QUESTION)))

    print()
    print("=" * 60)
    print("B 版：假 RAG（全部 20 篇文件整包塞進 prompt）")
    print("=" * 60)
    print(ask(build_prompt_full_context(TEST_QUESTION, ALL_DOCS)))

    print()
    print("=" * 60)
    print("C 版：真 RAG（TF-IDF + 餘弦相似度，撈 top-3）")
    print("=" * 60)
    retrieved = retrieve_top_k(TEST_QUESTION, k=3)
    print("檢索到的文件：", [(name, round(score, 3)) for name, score in retrieved])
    print(ask(build_prompt_retrieved(TEST_QUESTION, retrieved, ALL_DOCS)))

    print()
    print("=" * 60)
    print("驗收檢查（人工核對）")
    print("=" * 60)
    print("[ ] A 版有沒有出現幻覺（編造一個不存在的價格或報導名稱）？")
    print("[ ] B 版跟 C 版的答案是否一致？（一致代表 top-3 沒有漏掉關鍵文件）")
    print("[ ] C 版有沒有附上正確出處（富盛號碗粿 + 〈台南碗粿內行老饕必吃５家〉）？")
    print()
    print(f"[ ] 額外挑戰：retrieve_top_k('{FOLLOW_UP_QUESTION}') 撈到的 top-3")
    print("    有沒有包含阿溪火雞肉飯（正解）？已驗證這題會撈錯（阿溪掉出榜外），")
    print("    想想看：為什麼同一件事換個說法問，TF-IDF 就找不到了？")
    print(f"    print(retrieve_top_k('{FOLLOW_UP_QUESTION}'))")


if __name__ == "__main__":
    main()
