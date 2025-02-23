from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer,AlbertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import Dataset

# 加载BERT模型和分词器
# model_name = "./bert-base-chinese"  # 中文BERT
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

pretrained = './albert-tiny-chinese'
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AlbertForSequenceClassification.from_pretrained(pretrained,num_labels=2)


# 加载数据
data = pd.read_csv("dataset/fake.csv")
texts = data["text"].tolist()
labels = data["label"].tolist()


# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
print(train_texts[0])
# 创建数据集
train_dataset = Dataset(train_texts, train_labels, tokenizer)
test_dataset = Dataset(test_texts, test_labels, tokenizer)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,  # 最多保存2个检查点
    save_steps=500,      # 每500步保存一次
    load_best_model_at_end=True,  # 训练结束时加载最佳模型
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,  # 将分词器传入Trainer
)

# 训练模型
trainer.train()