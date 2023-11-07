import os
import torch
import tqdm
import json
from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler, DataLoader
import argparse
import numpy as np
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--model_name",
    default="roberta-base",
    type=str,
    help="Model type selected in the list: roberta-base ...",
)
parser.add_argument("--device", type=str, default="cuda", help="")
args = parser.parse_args()

def to_tensor_dataset(args, data, tokenizer):
    pad_token = 0
    labels = torch.stack(
        [torch.tensor([(d["label"])], dtype=torch.long) for d in data]
    ).squeeze()

    all_input_ids, all_attention_masks = [], []
    for d in data:
        inputs = tokenizer(d["text"])
        input_ids, attention_masks = inputs["input_ids"], inputs["attention_mask"]
        padding_length = args.max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_masks = attention_masks + ([0] * padding_length)
        input_ids = input_ids[: args.max_seq_length]
        attention_masks = attention_masks[: args.max_seq_length]

        assert (
            len(input_ids) == args.max_seq_length
        ), "Error with input length {} vs {}".format(
            len(input_ids), args.max_seq_length
        )
        assert (
            len(attention_masks) == args.max_seq_length
        ), "Error with input length {} vs {}".format(
            len(attention_masks), args.max_seq_length
        )

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_masks)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.int)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.int)
    if labels.shape == torch.Size([]):
        labels = labels.unsqueeze(0)
    dataset = TensorDataset(labels, all_input_ids, all_attention_masks)
    return dataset

MODEL_PATH = "models/Concrete_RobertaClsf_2023-05-21_19-01-16.pt"

class Ranker:
    def __init__(self):
        print(f"***Loading Model from {MODEL_PATH}***")
        model = torch.load(MODEL_PATH)
        self.model = model.to(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
        )

    def compare(self, t1, t2):
        text_pair = [
            {"label": -1, "text": t1 + " <sep> " + t2},
            {"label": -1, "text": t2 + " <sep> " + t1},
        ]
        pair_dataset = to_tensor_dataset(args, text_pair, self.tokenizer)
        score = self.run_model(pair_dataset)
        if score < 0.5:
            return 0  # first is more concrete
        else:
            return 1  # second is more concrete

    def run_model(self, dataset):
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)
        for batch in dataloader:
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
                outputs = self.model(**inputs)
            scores = (
                F.softmax(outputs.logits, dim=1)[:, 0]
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
            aver_score = (scores[0] + (1 - scores[1])) / 2
            return aver_score

    def rank(self, texts_list):  # input a list of texts
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            else:
                pivot = arr[0]
                less = []
                greater = []
                for t in arr[1:]:
                    cmp = self.compare(pivot, t)
                    if cmp == 0:
                        less.append(t)
                    elif cmp == 1:
                        greater.append(t)
                return quicksort(greater) + [pivot] + quicksort(less)

        return quicksort(texts_list)
        # most concrete -> lest concrete

    def rank_idx(self, texts_list):  # input a list of texts
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            else:
                pivot = arr[0]
                less = []
                greater = []
                for t in arr[1:]:
                    cmp = self.compare(texts_list[pivot], texts_list[t])
                    if cmp == 0:
                        less.append(t)
                    elif cmp == 1:
                        greater.append(t)
                return quicksort(greater) + [pivot] + quicksort(less)

        return quicksort(list(range(len(texts_list))))


def main():
    test_ranker = Ranker()
    print("Initialized Ranker.")

    outlines = []
    with open("path/to/outline.jsonl") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            item = json.loads(line)
            outlines.append(item)

    for test_text in outlines:
        res = test_ranker.rank_idx(test_text["text"])
        res_text = [test_text["text"][i] for i in res]
        res_text_leaf = [
            test_text["text"][i] for i in res if test_text["level"][i] != 0
        ]
        print("most concrete top 3: ", res_text[:3])
        print("most vague top 3: ", res_text[-3:])
        print("most vague top 3 leaves: ", res_text_leaf[-3:])
        print("check: ", [test_text["level"][i] for i in res])
