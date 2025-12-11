# CS336 Assignment 1: Basics - Implementation & Notes

This repository contains my full implementation, notes, and experimental results for the **CS336 (Spring 2025) Assignment 1: Basics**.  



## ✨ Highlights

### 🔥 Train Everything on a Single RTX 3090 (24GB)

All experiments are completed on **one NVIDIA GeForce RTX 3090 (24GB)** within **≈1 hour of training time**.
 No need to worry about compute — simply follow our configuration and you can fully reproduce the results.

### Fully Open Resources

To help you finish and verify the assignment quickly, we provide *everything*:

- Full **code implementation**
- **Tokenized datasets**
- **Training logs and learning curves**
- **Generated stories**
- **Notes and Q&A solutions**

### Extra Features Beyond the Assignment Requirements

We implemented several practical features commonly used in modern LLM systems:

- Parallel tokenizer encoding
- Advanced generation configurations:
  - `top-k`
  - `do_sample`
  - `repetition_penalty`

### Training Efficiency

Using only **81,920,000 tokens** from TinyStories (¼ of the official requirement):

- Our model reaches **validation loss = 1.536**
  *(assignment target: 1.45)*

Larger training budgets will likely achieve even lower loss.

### LLM-Judge Evaluation

We evaluate generated stories with **Qwen3-max** as an LLM-based judge:

- Story quality score: **8.4 / 10**

 ```js
     Lily and Ben are friends. They like to play with toys and read books. One day, they find a big box in the backyard. The box has many toys inside.
    "Wow! Look at this!" Lily says. "We can unpack this box and pretend we are pirates or princesses."
    "I am so happy!" Ben says. "Let's play pirates. We can make anything together!"
    They put the box under the bed and look for treasure. They put some old toys, some stickers, and some crayons on paper. They write their stories and share them with each other. They have fun.
    They are friends.
    
 ```

  

------

## Resources

* Full Solutions

  * [Complete problem solutions (Notion)](https://eager-alibi-97f.notion.site/assignment-1-Basics-298f0625020f807c9db1e336d1fe4263)

  

* Codebase

  * Complete implementation packaged as: **`cs336_basics`**

    

* Notes (Updating)

  *  [Lec2: PyTorch & Resource Accounting](https://eager-alibi-97f.notion.site/Lec2-PyTorch-resource-accounting-2b1f0625020f80d781d1f570c63a3e5d)

  *  [Lec3:  most of the large LMs have in common](https://eager-alibi-97f.notion.site/Lec3-most-of-the-large-LMs-have-in-common-2c5f0625020f80e68b87e489c3f0110f?source=copy_link)

    

* Tokenized Data

  * [Google Drive (preprocessed TinyStories & OWT)](https://drive.google.com/drive/folders/1ZdZdMzgWSapEUs4l4QgzG0J5fn5LtPoj)

  

* Training Logs & Learning Curves ([wandb](https://wandb.ai/fshihao900/cs336-basics))

  

* Generation Configurations & Stories ([wandb table](https://wandb.ai/fshihao900/cs336-basics))

------

## Setup

### Full files

We suggest that you should `git clone` the full code from [the official cs336-basics repository](https://github.com/stanford-cs336/assignment1-basics#). Then, you can replace same files using our repository. 

### Download data

You can download the raw data using the commands below, or download our **pre-tokenized data** directly from [Google Drive](https://drive.google.com/drive/folders/1ZdZdMzgWSapEUs4l4QgzG0J5fn5LtPoj?usp=sharing) to save time.



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



------

## 📂 Project Structure

The core implementation is located within the `cs336` package.

```
project_name/
│
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── main.py                  # Entry point for training/evaluation
├── inference.py             # Entry point for generation and justification
│
├── cs336/                   # Core library package
│   ├── models/              # Model architecture (Transformer, LLM definitions)
│   ├── datasets/            # Data loading and processing
│   ├── tokenization/        # BPE tokenizer and preparation scripts
│   └── utils/               # Configuration, checkpointing, and logging utils
│
├── configs/                 # YAML Configuration files
│   ├── config.yaml          # Training hyperparameters
│   └── generation_config.yaml # Text generation settings
│
└── docs/                    # Additional documentation
```

