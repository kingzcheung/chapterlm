from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# 加载BERT模型和分词器
model_name = "bert-base-chinese"  # 中文BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 数据预处理
class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def preprocess_data(self, data):
        if isinstance(data, str):
            return data
        elif isinstance(data, list) and all(isinstance(i, str) for i in data):
            return data
        else:
            return f"{data}"

    def __getitem__(self, idx):
        text = self.texts[idx]
        item = self.preprocess_data(text)
        label = self.labels[idx]
        encoding = self.tokenizer(
            item,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        if label not in ["0", "1"]:
            label = "0"
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(int(label), dtype=torch.long)
        }
