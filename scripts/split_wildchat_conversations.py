"""
This script is largely copied from the Vicuna repo: https://github.com/lm-sys/FastChat/blob/main/fastchat/data/split_long_messages.py
We fixed a bug in `split_one_sample`, which previously includes long messagess in the processed data. Now we skip these long messagess.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import transformers
from tqdm import tqdm


def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": str(sample["id"]) + "_" + str(start_idx),
        "messages": sample["messages"][start_idx:end_idx],
    }


tokenizer = max_length = None


def split_one_sample(sample):
    tokenized_lens = []
    messagess = sample["messages"]
    messagess = messagess[: len(messagess) // 2 * 2]
    for c in messagess:
        length = len(tokenizer(c["content"]).input_ids) + 6
        tokenized_lens.append(length)

    start_idx = 0
    cur_len = 0

    if len(messagess) % 2 != 0 or len(messagess) < 2:
        return []

    new_samples = []
    for i in range(0, len(messagess), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if cur_len + tmp_len > max_length:
            new_samples.append(make_sample(sample, start_idx, i))
            if tmp_len > max_length:  # if the current messages is too long, we should skip it
                start_idx = i + 2
            else:
                start_idx = i
            cur_len = 0
        elif i == len(messagess) - 2:
            new_samples.append(make_sample(sample, start_idx, i + 2))

        cur_len += tmp_len

    return new_samples


def split_all(content, begin, end, tokenizer_, max_length_):
    """
    Keep the maximum round of messagess within the max token length constraint
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    content = content[begin:end]
    new_content = []

    with ProcessPoolExecutor(max_workers=256) as executor:
        for result in tqdm(executor.map(split_one_sample, content), total=len(content)):
            new_content.extend(result)

    return new_content


# def filter_invalid_roles(content):
#     new_content = []
#     for i, c in enumerate(content):
#         roles = ["human", "gpt"]
#         if len(c["messages"]) <= 0:
#             continue

#         valid = True
#         for j, s in enumerate(c["messages"]):
#             if s["role"] != roles[j % 2]:
#                 valid = False
#                 break

#         if valid:
#             new_content.append(c)

#     return new_content


def main(args):
    content = []
    with open(args.in_file) as f:
        for line in f:
            content.append(json.loads(line))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    new_content = split_all(content, args.begin, args.end, tokenizer, args.max_length)
    # new_content = filter_invalid_roles(new_content)

    print(f"total: {len(content)}, new: {len(new_content)}")
    with open(args.out_file, "w") as f:
        for c in new_content:
            f.write(json.dumps(c) + "\n")
    # json.dump(new_content, open(args.out_file, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    main(args)