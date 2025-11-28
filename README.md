# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](docs/cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

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

# Structure
下面给你一个**适用于深度学习科研项目的文件夹组织方案**，它既适合多人协作，也方便长期维护、结果追踪与论文复现。结构遵循清晰、可扩展、可复现三大原则。

---

# 🧭 整体组织思路（黄金原则）

### **1. 项目分层：代码 / 数据 / 实验 / 文档 分开**

把不同职能的内容强制分到不同层级，减少混乱。

### **2. 每次实验自动产生独立记录（实验即目录）**

训练脚本自动创建一个实验文件夹，保存配置、结果、指标和模型权重。

### **3. 配置文件（configs）独立**

使实验的可复现性更高，修改配置无需改代码。

### **4. 原始数据与处理后数据分离**

避免误删和数据泄露；大型数据集不放在 Git 内。

### **5. 工具/脚本统一管理**

任何处理脚本都放 tools 下，避免散落到处都是。

---

# 📂 推荐项目文件结构（示例）

```
project_name/
│
├── README.md                # 项目说明、环境安装方法、指令示例
├── requirements.txt         # Python 包依赖
├── setup.sh                 # 环境/数据配置脚本（可选）
│
├── src/                     # 代码源文件（整个项目的逻辑核心）
│   ├── models/              # 模型架构（.py）
│   ├── datasets/            # 数据加载器、Transforms
│   ├── trainers/            # 训练循环、验证逻辑
│   ├── utils/               # 公共工具函数（logging、metrics 等）
│   └── main.py              # 程序入口（train/test）
│
├── configs/                 # 实验配置文件（YAML）
│   ├── baseline.yaml
│   └── ablation_lr.yaml
│
├── data/                    # 数据不放 Git（用 .gitignore）
│   ├── raw/                 # 原始数据
│   └── processed/           # 预处理后数据
│
├── experiments/             # 每次实验独立文件夹（自动创建）
│   ├── exp_001_baseline/
│   │   ├── config.yaml      # 当时的配置快照
│   │   ├── log.txt
│   │   ├── tensorboard/     # TensorBoard 日志
│   │   ├── checkpoints/     # 模型权重
│   │   └── results.json     # 最终指标
│   └── exp_002_lr1e-4/
│
├── logs/                    # 全局日志（可选）
│
├── tools/                   # 实用脚本（不属于主代码逻辑）
│   ├── preprocess.py        # 数据预处理
│   ├── visualize.py         # 可视化工具
│   └── export_model.py      # 模型导出 ONNX/TorchScript
│
└── docs/                    # 文档、论文草稿、实验记录
    ├── paper/
    ├── notes.md
    └── figures/
```

---

# 🔍 各模块的功能说明

## 1. `src/` — 项目代码核心

保持模块化，让模型、数据、训练彼此独立，便于复用。

**推荐结构：**

* `datasets/`：不要把数据写死在脚本里，用 config 指定路径。
* `models/`：每个模型一个文件，例如 `resnet.py`
* `trainers/`：统一训练框架，使得比对模型只需换 config。

---

## 2. `configs/` — YAML 配置文件（强烈推荐）

保持所有实验的参数在这里，比如：

```
model: resnet50
optimizer:
  type: Adam
  lr: 1e-4
dataset:
  name: CIFAR10
train:
  epochs: 100
```

这允许：

* 快速切实验
* 可复现
* 自动记录实验配置

---

## 3. `experiments/` — 实验自动记录中心（高效科研的关键）

每做一次实验：

👉 自动创建：

```
exp_003_new_aug/
├── config.yaml
├── log.txt
├── checkpoints/
├── results.json
└── tensorboard/
```

这样你可以**随时回溯实验设定，不会忘记用的是什么参数。**

（可以用 MLflow、Sacred、Weights & Biases 进一步自动化）

---

## 4. `data/` — 管理数据

数据不进 Git，把其路径写进 `.gitignore`。

```
data/
  raw/
  processed/
```

方便多人组织，也防止误删。

---

## 5. `tools/` — 常用脚本

避免把临时脚本乱丢到仓库根目录。

如：

* 数据清洗
* 可视化脚本
* 模型转换脚本
* 结果统计脚本

---

## 6. `docs/` — 文献与项目文档

把论文、实验记录和可视化结果放这里。

你也可以用：

* `docs/paper/`：论文写作
* `docs/notes.md`：自己的研究笔记
* `docs/figures/`：实验图片

---

# 🌟 加分建议（让项目更专业）

### ⭐ 使用 `.gitignore` 忽略大文件

例如：

```
data/
experiments/
logs/
*.pth
*.ckpt
```

---

### ⭐ 用 DVC 管理数据（专业科研团队强烈推荐）

DVC 提供：

* 数据版本管理
* 模型文件管理
* 与 Git 自动关联

---

### ⭐ 用 Conda 或 Docker 固定环境

给每个实验带来真正可复现性。

```
environment.yml
Dockerfile
```

---

### ⭐ 自动生成实验编号

例如：

```bash
python train.py --config config.yaml --exp-name baseline
```

脚本自动生成：

```
experiments/exp_2025-01-18_14-33-01_baseline/
```

---

# 📘 总结

一个成熟的深度学习科研工程应做到：

### ✓ **代码、数据、实验、文档彻底拆分**

### ✓ **每次实验自动有记录与结果快照**

### ✓ **配置文件管理实验参数**

### ✓ **数据版本明确**

### ✓ **可复现、可扩展、可协作**

按这个结构组织后：

* 项目更整洁
* 实验可溯源
* 更容易写论文
* 便于合作与未来复现

---

如果你愿意，我可以：

📌 **为你的具体项目生成一个完整可直接使用的文件夹模板**
📌 **帮你写一个自动创建实验目录的 Python 训练脚本**
📌 **帮你制作一个 config + log 自动化框架（类似 W\&B 的轻量替代）**

需要吗？


