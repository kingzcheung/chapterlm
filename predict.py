from transformers import AlbertForSequenceClassification, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "./results/checkpoint-9386"  # 中文ALBERT-Tiny（仅18MB）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# print(model)
# 推理示例
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        print(outputs)
    return torch.argmax(outputs.logits).item()

text = "TinyBERT 的结构如下图："

pred = predict(text)

print("是章节名" if pred == 1 else "不是章节名")
print(f"pred:{pred}")