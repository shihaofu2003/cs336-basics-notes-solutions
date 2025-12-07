import json
from collections import defaultdict
import torch
import torch.nn as nn
import yaml
import re


from openai import OpenAI

from cs336_basics.models.llm import TransformerLM
from cs336_basics.tokenization.prepare import load_vocab_merges
from cs336_basics.tokenization.tokenizer import Tokenizer


def extract_final_score(text: str):
    # 匹配 "Final Average Score: 8.7/10" 这类格式，允许空格和大小写微小差异
    pattern = r"Final\s*Average\s*Score\s*:\s*(\d+\.\d|\d+)\s*/\s*10"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        return None


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load generation config
    with open('./configs/generation_config.yaml', 'r') as file:
        generate_config = yaml.safe_load(file)
        generate_config['eos_token_id'] = generate_config['eos_token_id'][0]
    print(generate_config)

    # tokenizer
    vocab_path = "data/TinyStories/TinyStories_train_vocab.json"
    merges_path = "data/TinyStories/TinyStories_train_vocab_merges.txt"
    vocab, merges = load_vocab_merges(vocab_path, merges_path)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    # LM model
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000
    ).to(device=device)

    # load weights
    weights_path = "experiments/lr_1e-2_1e-4/2025-12-05_23-59-40/checkpoints/2499step_1.93.pt"
    torch.serialization.add_safe_globals([defaultdict])
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict["model"])

    input_str = "<|endoftext|>\n\n"
    input_ids = tokenizer.encode(input_str)
    input_ids = torch.tensor(input_ids, dtype=torch.long).reshape(1, -1)

    batch, seq = input_ids.shape
    attn_mask = torch.tril(torch.ones(batch, 1, seq, seq, dtype=torch.bool))
    print(input_ids, attn_mask)

    output_ids = model.generate(
        input_ids.to(device=device),
        attn_mask.to(device=device),
        **generate_config
    )

    print(output_ids)
    print(output_ids.shape)

    output_str = tokenizer.decode(output_ids.tolist()[0])
    print("output_str: \n", output_str, "\n----------------------\n")

    prompt = f"""
    You are an expert evaluator for children's stories trained on the TinyStories dataset. Your task is to evaluate how well a generated story matches the style, quality, and characteristics of the best TinyStories models (like TinyStories-33M or better).

Rate the following story on a scale of 1 to 10 for EACH of the following 7 criteria (1 = very poor, 10 = excellent, indistinguishable from the best human-written TinyStories examples). After each score, write a one-sentence justification.

Story to evaluate:
{output_str}

Criteria:

1. Grammar & Fluency  
   (Is the English perfectly correct, natural, and easy to read aloud to a 3–7-year-old child?)

2. Coherence & Logical Flow  
   (Does the story have a clear beginning, middle, and end? Do events follow logically?)

3. Repetition & Looping  
   (Penalty: does the story repeat the same sentences or ideas excessively? Best TinyStories almost never repeat the same sentence. 10 = no harmful repetition)

4. Story Completeness & Satisfaction  
   (Does it feel like a complete little story? Or does it end with a proper conclusion, typically including the "<|endoftext|>" token?)

5. Vocabulary & Sentence Complexity  
   (Uses only simple, age-appropriate words and short sentences like real TinyStories? Almost no complex clauses.)

6. Creativity & Engagement  
   (Is it imaginative and fun for small children, with characters, emotions, and a tiny conflict/resolution?)

7. Overall TinyStories Authenticity  
   (If you saw only this text with no other context, would you believe it was written by a top-tier TinyStories model? 10 = yes, perfectly matches the official examples)

Finally, compute the average of the 7 scores (round to 1 decimal place) and write:
"Final Average Score: X.X/10"

Then write one short paragraph (3–5 sentences) summarizing the strengths and the most important weaknesses of this story compared to the very best TinyStories outputs.

Respond using only this structured format, no extra text."""

    client = OpenAI(
        api_key="sk-35efdaa15b56437c8f86f9ec2f2fd52c",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    justification = completion.choices[0].message.content
    print("Justification: \n", justification)

    score = extract_final_score(justification)
    print("score: ", score)

    content = {"generation_config": generate_config, "LM_content": output_str,
               "LM_justification": justification, "score": score}
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append(content)

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=4)






