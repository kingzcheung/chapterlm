from bs4 import BeautifulSoup
import pandas as pd

def main():
    with open("data/raw.html",'r') as f:
        html_doc = f.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    s = soup.find_all('a')
    
    # 追加的形式写入
    with open("data/chapter.txt",'a') as f:
        for i in s:
            f.write(i.get_text()+'\n')
    
    
def merge():
    with open("data/chapter.txt",'r') as f:
        lines = f.readlines()
    df = pd.read_csv("dataset/fake.csv", header=None, names=["text", "label"])
    # 合并 lines 到 fake.csv，并在
    data_with_labels = [[text, "1"] for text in lines]
    dfdata = pd.DataFrame(data_with_labels, columns=["text", "label"])
    df = pd.concat([df, dfdata], ignore_index=True)
    # 保存
    df.to_csv("dataset/fake.csv", index=False)

if __name__ == "__main__":
    merge()
