import os
import random

from datasets import load_dataset, concatenate_datasets
from promptsource.templates import DatasetTemplates
from datasets import DatasetDict
import argparse
from config import DATASETS as p3_datasets
from tqdm import tqdm

random.seed(42)

all_train_datasets = []
all_test_datasets = []

def process_p3_datasets(dataset, dataset_name, category, prompt_template, num_workers=4):
    # only keep the answer_choices, inputs_pretokenized and targets_pretokenized columns
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['answer_choices', 'inputs_pretokenized', 'targets_pretokenized']])
    # rename the columns to be consistent with the template
    dataset = dataset.rename_column('answer_choices', 'choices')
    dataset = dataset.rename_column('inputs_pretokenized', 'input')
    dataset = dataset.rename_column('targets_pretokenized', 'output')

    # add dataset name to the dataset
    dataset = dataset.map(
        lambda example: {'dataset': dataset_name, 'category': category, 'prompt_template': prompt_template, **example},
        num_proc=num_workers
    )

    return dataset

def process_storycloze_datasets(dataset, prompt_template, prompt_template_obj, num_workers=4):
    dataset_name = 'storycloze'
    category = 'sc'
    dataset = dataset.map(
        lambda example: {'dataset': dataset_name, 'category': category, 'prompt_template': prompt_template, 'input': prompt_template_obj.apply(example)[0], 'output': prompt_template_obj.apply(example)[1], 'choices': [example['sentence_quiz1'], example['sentence_quiz2']]},
        num_proc=num_workers
    )
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['dataset', 'category', 'prompt_template', 'input', 'output', 'choices']])

    return dataset
    
def main(args):

    for name, dataset in tqdm(p3_datasets.items()):
        print(f"Processing {name}...")

        dataset_type = getattr(dataset, 'type', None)
        hf_dataset = getattr(dataset, 'dataset', None)

        if hf_dataset is not None:
            hf_subset = getattr(dataset, 'prompt_template', None)

            train_dataset = load_dataset(hf_dataset, hf_subset, split='train')
            test_dataset = load_dataset(hf_dataset, hf_subset, split='test')

            answer_choices = test_dataset['answer_choices'][0]
            if test_dataset['targets_pretokenized'][0] not in answer_choices:
                test_dataset = load_dataset(hf_dataset, hf_subset, split='validation')

            # filter the dataset to only contain train_split and test_split
            train_dataset = process_p3_datasets(train_dataset, name, dataset_type, hf_subset, args.num_workers)
            test_dataset = process_p3_datasets(test_dataset, name, dataset_type, hf_subset, args.num_workers)
        else:
            # This happens when the dataset is storycloze, since it's behind license and we can't load it directly from huggingface
            # we need to load it from local path instead

            local_path = getattr(dataset, 'dataset_path', None)
            assert local_path is not None, "dataset_path is not specified"

            prompt_source = getattr(dataset, 'prompt_source')
            prompt_template = getattr(dataset, 'prompt_template')
            
            prompt_template_obj = DatasetTemplates(prompt_source)[prompt_template]

            def sample_random_wrong_answer(example):
                sentence_quiz1 = example['sentence5']
                sentence_quiz2 = random.choice(sentence5_collections)
                answer_right_ending = 1
                while sentence_quiz2 == sentence_quiz1:
                    sentence_quiz2 = random.choice(sentence5_collections)
                
                # shuffle the order of the two sentences and keep the idx of the correct answer "sentence_quiz1"
                if random.random() < 0.5:
                    sentence_quiz1, sentence_quiz2 = sentence_quiz2, sentence_quiz1
                    answer_right_ending = 2
                return {'sentence_quiz1': sentence_quiz1, 'sentence_quiz2': sentence_quiz2, 'answer_right_ending': answer_right_ending}
                
            # add idx to the dataset
            train_dataset = load_dataset("csv", data_files=os.path.join(local_path, 'train.csv'))
            test_dataset = load_dataset("csv", data_files=os.path.join(local_path, 'validation.csv'))
            sentence5_collections = train_dataset['train']['sentence5']

            train_dataset = train_dataset['train'].map(
                lambda example: {**sample_random_wrong_answer(example), 'input_sentence_1': example['sentence1'], 'input_sentence_2': example['sentence2'], 'input_sentence_3': example['sentence3'], 'input_sentence_4': example['sentence4']},
                num_proc=args.num_workers
            )
            test_dataset = test_dataset['train'].map(
                lambda example: {'input_sentence_1': example['InputSentence1'], 'input_sentence_2': example['InputSentence2'], 'input_sentence_3': example['InputSentence3'], 'input_sentence_4': example['InputSentence4'], 'sentence_quiz1': example['RandomFifthSentenceQuiz1'], 'sentence_quiz2': example['RandomFifthSentenceQuiz2'], 'answer_right_ending': example['AnswerRightEnding']},
                num_proc=args.num_workers
            )

            train_dataset = process_storycloze_datasets(train_dataset, prompt_template, prompt_template_obj, args.num_workers)
            test_dataset = process_storycloze_datasets(test_dataset, prompt_template, prompt_template_obj, args.num_workers)
            
        all_train_datasets.append(train_dataset)
        all_test_datasets.append(test_dataset)

    # Merge all datasets
    train_dataset = concatenate_datasets(all_train_datasets)
    test_dataset = concatenate_datasets(all_test_datasets)

    # combine train_dataset and test_dataset as 'train' and 'test' split
    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    dataset.push_to_hub(args.push_to, private=True)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--push_to", type=str, default="simonycl/p3_0.5_dataset")
    arg_parser.add_argument('--num_workers', type=int, default=4)


    args = arg_parser.parse_args()
    main(args)