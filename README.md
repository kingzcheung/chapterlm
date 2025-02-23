# chapterlm - 识别文章章节模型

最近在为做一个小说阅读排版，需要识别文章章节，所以就训练了这个模型。很多小说的章节名并不统一，基于还非常的混乱。正常的正则表达式难以适应。使用小模型来识别虽然有一些重，但是效果比较出色。

### 安装依赖

```bash
uv async
```

### 训练

```bash
uv run train.py
```

### 预测
```python
from transformers import AlbertForSequenceClassification, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "rkingzhong/chapterlm"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(model)
# 推理示例
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        print(outputs)
    return torch.argmax(outputs.logits).item()

text = "一，黄家药铺"

pred = predict(text)

print("是章节名" if pred == 1 else "不是章节名")
print(f"pred:{pred}")
```
