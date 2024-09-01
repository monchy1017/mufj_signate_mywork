import torch
from transformers import pipeline

# GPUが利用可能かどうかを確認
device = 0 if torch.cuda.is_available() else -1

# 感情分析パイプラインの初期化（特定のモデルを指定）
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# サンプルテキスト
texts = [
    "This app is great! 👍 I use it every day and love it. 👍",
    "Terrible experience. The app crashes all the time.",
    "Good functionality but needs improvement in UI. 👍",
    "Not bad, but could be better.",
    "Worst app ever. Do not download!",
]

# 感情分析
for text in texts:
    result = classifier(text)
    print(text)
    print(result)
