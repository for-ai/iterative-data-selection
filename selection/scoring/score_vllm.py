from scorer import Llama_Scorer
import json
from tqdm import tqdm
import torch


model_name_or_path = "hkust-nlp/deita-quality-scorer"
data_path = "/mnt/data/data-selection/data/processed/sharegpt/sharegpt_data.jsonl"
output_path = "/mnt/data/data-selection/data/processed/sharegpt"
scorer = Llama_Scorer(model_name_or_path, is_vllm = True)
BATCH_SIZE = 64

# example input
# input_text = "word to describe UI with helpful tooltips" # Example Input
# output_text = "User-friendly or intuitive UI" # Example Output
# quality_score = scorer.infer_quality(input_text, output_text)
scores = []

with open(data_path) as f:
    lines = f.readlines()
for i in tqdm(range(0, len(lines), BATCH_SIZE)):
    batches = lines[i:i+BATCH_SIZE]
    messages = [json.loads(line)["messages"] for line in batches]
    len_conversation = [len(message) // 2 for message in messages]
    index = [i for i, count in enumerate(len_conversation) for _ in range(count)]
    local_index = [i for _, count in enumerate(len_conversation) for i in range(count)]
    inputs = [messages[j][2*i]["content"] for i, j in zip(local_index, index)]
    outputs = [messages[j][2*i+1]["content"] for i, j in zip(local_index, index)]
    quality_scores, complexity_scores = scorer.batch_infer(inputs, outputs)

    index, quality_scores, complexity_scores = torch.tensor(index, dtype=torch.int64), torch.tensor(quality_scores, dtype=torch.int64), torch.tensor(complexity_scores, dtype=torch.int64)
    dummy = torch.zeros(BATCH_SIZE, dtype=torch.int64)

    # gather reduce scores based on index
    evol_scores = quality_scores * complexity_scores
    eval_scores = dummy.scatter_add(0, index, evol_scores).tolist()
    quality_scores = dummy.scatter_add(0, index, quality_scores).tolist()
    complexity_scores = dummy.scatter_add(0, index, complexity_scores).tolist()

    scores += [{"eval_score": eval_score, "quality_score": quality_score, "complexity_score": complexity_score} for eval_score, quality_score, complexity_score in zip(eval_scores, quality_scores, complexity_scores)]

with open(output_path + "/scores.jsonl", "w") as f:
    for score in scores:
        f.write(json.dumps(score) + "\n")

    # quality_score = scorer.infer_quality(input_text, output_text)

# output the scores
# print(quality_score)
        
# nohup python examples/scoring/score_vllm.py > examples/scoring/score_vllm.log 2>&1 &