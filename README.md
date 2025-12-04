# My CS336 Spring 2025 Assignment 1: Basics Notes and Solutions



For a full **problem solutions** of this assignment, see the notion website at [CS336_Assignment1_Solutions](https://eager-alibi-97f.notion.site/assignment-1-Basics-298f0625020f807c9db1e336d1fe4263?source=copy_link)

For a full **code Implementation** of this assignment, see the library `cs336_basics` 

For a full **notes** of this assignment, please wait me to update my notion websites hear.



## Setup

### Full files

We suggest that you should `git clone` the full code from [the official cs336-basics repository](https://github.com/stanford-cs336/assignment1-basics#). Then, you can replace same files using our repository. 

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```



Also you can Download [our owt and TinyStories Tokenization Results in Google Drive](https://drive.google.com/drive/folders/1ZdZdMzgWSapEUs4l4QgzG0J5fn5LtPoj?usp=sharing)

---

# 📂 项目文件结构

```
project_name/
│
├── README.md                
├── requirements.txt         # Python 包依赖
│
├── cs336/                   # the core of this project
│   ├── models/              # llm/functions/optim/model.py
│   ├── datasets/            # loading.py
│   ├── tokenization/        # bpe/prepare/tokenizer.py
│   ├── utils/               # checkpoint/config/set.py
│
├── configs/                 # train and generation config
│   ├── config.yaml
│   └── generation_config.yaml
│
│── main.py          
│── docs/                    

```
