# Copyright (c) OpenMMLab. All rights reserved.
import transformers
from trainer import Trainer

from xtuner.apis import DefaultTrainingArguments, build_lora_model
from xtuner.apis.datasets import alpaca_data_collator, alpaca_dataset
import numpy as np

import json
import torch
from torch.utils.data import Dataset, DataLoader

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, partition="train"):
        self.ann = json.load(open(data_path))
        if partition == "train":
            self.ann = self.ann[:80]
        else:
            self.ann = self.ann[80:]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        prompt = PROMPT_DICT["prompt_no_input"].format_map(ann) 

        if isinstance(ann["output"], str):
            example = prompt + ann["output"]
        else:
            example = prompt + str(ann["output"])
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        
        example_mask = example.ge(0)
        
        example[~example_mask] = 0
        

        # 填充至指定长度
        max_length = 1064  # 指定的长度
        # fill_value = 0  # 指定的填充值
        example = pad(example, (0, max_length - len(example)), value=0)
        
        example_mask = pad(example_mask, (0, max_length - len(example_mask)), value=0)

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        label_mask = labels.ge(0)
        labels[~label_mask] = IGNORE_INDEX
        labels = pad(labels, (0, max_length - len(labels)), value=-100)

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }


# from typing import Dict, Sequence

# def collate_fn(batch):
#     # 将batch中的元素分解为序列和标签
#     sequences, labels = zip(*batch)
    
#     # 对序列进行填充
#     sequences = [pad(sequence, (0, 1062 - len(sequence))) for sequence in sequences]
    
#     return sequences, labels

# def default_collate_fn(instances: Sequence[Dict]):

#     input_ids, labels, attention_mask = [], [], []

#     for example in instances:
#         input_ids.append(torch.LongTensor(example['input_ids']))
#         attention_mask.append(torch.LongTensor(example['attention_mask']))
#         labels.append(torch.LongTensor(example['labels']))

#     if len(instances) > 1:
#         input_ids = [torch.from_numpy(np.pad(ii.numpy(), (0, 1062 - len(ii)), constant_values=0)) for ii in input_ids]
#         attention_mask = [torch.from_numpy(np.pad(am.numpy(), (0, 1062 - len(am)), constant_values=0)) for am in attention_mask]
#         labels = [torch.from_numpy(np.pad(ll.numpy(), (0, 1062 - len(ll)), constant_values=-100)) for ll in labels]


    
#     data_dict = {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'labels': labels
#     }

#     return data_dict




def train():
    # get DefaultTrainingArguments and to be updated with passed args
    parser = transformers.HfArgumentParser(DefaultTrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]

    # init model and dataset
    model, tokenizer = build_lora_model(
        model_name_or_path=training_args.model_name_or_path,
        return_tokenizer=True)
    # train_dataset = alpaca_dataset(
    #     tokenizer=tokenizer, path='/home/pgpu/code/train_data')
    # data_collator = alpaca_data_collator(return_hf_format=True)

    train_dataset = InstructionDataset('/home/pgpu/code/train_data/train_dataset.json', tokenizer)

    # for i in train_dataset:
    #     if len(i['input_ids']) != 418 or len(i['labels']) != 418 or len(i['attention_mask']) != 418:
    #         print("+++++++++++++++++++++++++++++++++++")
    #         print(len(i['input_ids']))
    #         print("+++++++++++++++++++++++++++++++++++")
    #         print(i)
    #         print("+++++++++++++++++++++++++++++++++++")

    # train_dataset = pad_sequence((i for i in train_dataset), batch_first=True)

    # for i in train_dataset:
    #     if len(i['input_ids']) != 1062 or len(i['labels']) != 1062 or len(i['attention_mask']) != 1062:
    #         print("+++++++++++++++++++++++++++++++++++")
    #         print(len(i['input_ids']))
    #         print("+++++++++++++++++++++++++++++++++++")
    #         print(i)
    #         print("+++++++++++++++++++++++++++++++++++")

    # build trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
        ) #data_collator=default_collate_fn

    # training
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir='/home/pgpu/code/output_model')


if __name__ == '__main__':
    train()
