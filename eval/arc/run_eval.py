import argparse
import os
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
import logging
import datasets
from eval.mmlu.categories import subcategories, categories
from eval.utils import get_next_word_predictions, load_hf_lm_and_tokenizer, dynamic_import_function
from finetune.eval.utils import score_qa_task

choices = ["A", "B", "C", "D"]

key2idx = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}

def format_example(df, idx, include_answer=True):
    prompt = "Question: " + df.iloc[idx]['question'] + "\nAnswer:"
    choices = df.iloc[idx]['choices']
    # load choices from json if it's a string
    if isinstance(choices, str):
        choices = json.loads(choices)
    labels = list(choices['label'])
    text = choices['text']
    if include_answer:
        prompt += text[labels.index(df.iloc[idx]['answerKey'].strip())]
    return prompt

def gen_prompt(train_df, k=-1):
    prompt = ""
    # prompt = "The following are multiple choice questions (with answers).\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
        prompt += "\n\n"
    return prompt


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    # process the test_df and turn into dict
    # the targeted should be as follows: {
    #    "id": "test-0",
    #    "input": FEW-SHOT PROMPT + QUESTION,
    #   "output": IDX OF THE CORRECT ANSWER
    # }
    scoring_examples = []
    ori_train_prompt = gen_prompt(dev_df, args.ntrain)

    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = ori_train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"
        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"
                    
            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        
        prompts.append({
            "id": "test-{}".format(i),
            "input": prompt,
            "choices": list(test_df.iloc[i]['choices']["text"]),
            "label": key2idx[test_df.iloc[i]['answerKey'].strip()] if test_df.iloc[i]['answerKey'].strip() in key2idx else int(test_df.iloc[i]['answerKey'].strip()),
        })

    print("Prompt examples:", prompts[0])
    acc = score_qa_task(model, tokenizer, prompts, batch_size=batch_size)
    return acc
    # # get the answer for all examples
    # # adding a prefix space here, as that's expected from the prompt
    # # TODO: should raise a warning if this returns more than one token
    # answer_choice_ids = [tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in key2idx]
    # pred_indices, all_probs = get_next_word_predictions(
    #     model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    # )

    # # get the metrics
    # cors = []
    # groud_truths = [key2idx[k] if k in key2idx else k for k in list(test_df['answerKey'])]

    # for i in range(len(pred_indices)):
    #     # prediction = choices[pred_indices[i]]
    #     prediction = pred_indices[i]
    #     ground_truth = groud_truths[i]
    #     cors.append(prediction == ground_truth)
        
    # acc = np.mean(cors)
    # cors = np.array(cors)

    # all_probs = np.array(all_probs)
    # print("Average accuracy {:.3f}".format(acc))
    # return cors, acc, all_probs

def main(args):

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit, 
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            use_fast_tokenizer=not args.use_slow_tokenizer,
            padding_side="right"
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.info("loading data...")
    hellaswag_eval_data = datasets.load_dataset("ai2_arc", "ARC-Challenge")

    dev_df = hellaswag_eval_data['validation'].to_pandas()
    test_df = hellaswag_eval_data['test'].to_pandas()

    if args.n_instances:
        test_df = test_df[:args.n_instances]

    if args.model_name_or_path:
        acc = eval_hf_model(args, model, tokenizer, dev_df, test_df, batch_size=args.eval_batch_size)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "acc": acc,
            },
            f
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mmlu/llama-7B/"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
