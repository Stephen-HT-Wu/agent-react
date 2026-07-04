"""階段 1：CoT (Chain-of-Thought) 練習。

同一個行程規劃問題問兩次：
  A 版：直接要答案（無 CoT 基準線）
  B 版：要求先逐項推理再給答案（CoT）

目標：找出至少一個 A 版排錯、B 版排對的案例。
已埋好的陷阱：
  1. 老城鹹粥 04:30-12:00（賣完收攤）——排到晚餐就是錯的。
  2. 府前冰果室週三公休——TASK 指定「這個週三出發」，
     若還把府前冰果室排進行程，就是沒真的核對公休日對上出遊日期。

執行前：
  pip install anthropic
  export ANTHROPIC_API_KEY=...
執行：
  python3 01_cot.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

import anthropic

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
print(f"Using model: {MODEL}")
DATA = Path(__file__).parent / "data" / "shops.json"

client = anthropic.Anthropic()

def load_tainan_shops() -> str:
    """把台南店家整理成塞進 prompt 的文字。

    這階段刻意不做檢索——全部資料直接進 prompt，
    等階段 3 資料塞不下時你就會體會到為什麼需要 RAG。
    """
    shops = [s for s in json.loads(DATA.read_text(encoding="utf-8")) if s["city"] == "台南"]
    lines = []
    for s in shops:
        lines.append(
            f"- {s['name']}（{s['district']}）：{s['dish']}，"
            f"營業 {s['hours']}，價位 {s['price_range']} 元。{s['note']}"
        )
    return "\n".join(lines)

#TASK = "幫我用這些資料集裡的店，排一個台南一日美食行程（早餐到宵夜），預算 800 元。"
TASK = "幫我用這些資料集裡的店，排一個台南一日美食行程（早餐到宵夜），這趟訂在這個週三出發，預算 800 元。"

def build_prompt_a(shops_text: str) -> str:
    """A 版：直接問，不引導推理。"""
    return f"""安排台南一日遊行程。
預算 800 元。
店家資料：
{shops_text}
TASK：
{TASK}
"""

def build_prompt_b(shops_text: str) -> str:
    """B 版：CoT——要求模型先列出考量、逐項推理，最後才給行程。"""
    return f"""先推理、後結論。
安排台南一日遊行程。
預算 800 元。
依序檢查：
1. 每家店的營業時間，對應到哪個用餐時段才合理
2. 每家店的公休日，是否剛好對到 TASK 指定的出遊日期，
   若公休就必須從行程中排除或替換掉該店
3. 地理動線（同區的店排在一起）
4. 預算加總不超過上限
5. 以上都檢查完，才輸出最終行程
店家資料：
{shops_text}
TASK：
{TASK}
"""

def ask(prompt: str) -> str:
    # 刻意不開 thinking：這個練習要觀察的是「prompt 引導出的推理」，
    # 模型內建的 thinking 會替你完成推理，A/B 對比就失真了。
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")

def main() -> None:
    shops_text = load_tainan_shops()

    print("=" * 60)
    print("A 版（無 CoT）")
    print("=" * 60)
    print(ask(build_prompt_a(shops_text)))

    print()
    print("=" * 60)
    print("B 版（CoT）")
    print("=" * 60)
    print(ask(build_prompt_b(shops_text)))

    print()
    print("=" * 60)
    print("驗收檢查（人工核對）")
    print("=" * 60)
    print("[ ] 老城鹹粥（04:30-12:00）有沒有被排到下午或晚上？")
    print("[ ] 保安豬心肺粉（17:00-00:00）有沒有被排到白天？")
    print("[ ] 府前冰果室（週三公休）有沒有還是被排進這趟週三的行程？")
    print("[ ] 預算加總有沒有超過 800？（A 版常常不加總就宣稱沒超過）")
    print("[ ] 動線合不合理？（中西區的店互相都在步行距離內）")
    print("跑個 3-5 次：單次 A 版可能剛好答對，多跑幾次看錯誤率的差異。")

if __name__ == "__main__":
    main()
