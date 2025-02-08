
import pandas as pd
from pathlib import Path


def create_csv_data(file: str):
    """
    按行读文本文件，并生成csv文件
    """
    with open(file, 'r') as f:
        data = f.readlines()
        data = [i.strip() for i in data]
        # 生成csv文件，并在第二列添加1列，内容为0
        # 以追加的形式写入
        return data
        
    
if __name__ == '__main__':
    
    # 使用panda 打开文件没有文件就创建
    df = pd.read_csv("dataset/fake.csv", header=None, names=["text", "label"])
    
    data = []
    for p in Path("./data").iterdir():
        if not p.name.endswith(".nb"): continue
        d = create_csv_data(str(p))
        data += d
        
    data_with_labels = [[text, "0"] for text in data]

    dfdata = pd.DataFrame(data_with_labels, columns=["text", "label"])
    
    df = pd.concat([df, dfdata], ignore_index=True)
    # 保存
    df.to_csv("dataset/fake.csv", index=False)
        
        
        