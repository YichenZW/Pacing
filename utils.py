import json
import nltk
import tiktoken
from transformers import RobertaTokenizer
import random
import torch
import tqdm
from torch.utils.data import TensorDataset

def load_jsonl(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        item = json.loads(line)
        item["dataset_id"] = idx
        data.append(item)
    print("     Successfully loaded {} lines".format(len(data)))
    return data


def find_closest_sublist(tgt_len, l):
    n = len(l)
    closest_sum = float("inf")
    closest_index = (0, 1)
    for i in range(n):
        sublist_sum = l[i]
        j = i + 1
        while j < n:
            if abs(sublist_sum - tgt_len) < abs(closest_sum - tgt_len):
                closest_sum = sublist_sum
                closest_index = (i, j)
            if sublist_sum == tgt_len:
                return closest_index
            elif sublist_sum < tgt_len:
                sublist_sum += l[j]
                j += 1
            else:
                break
    return closest_index


def trunc_text(text, trg_len):
    sentences = nltk.sent_tokenize(text)
    sen_len = [get_token_numbers(s) for s in sentences]
    sublist = find_closest_sublist(trg_len, sen_len)
    trunced = sentences[sublist[0] : sublist[1]]
    trunced = " ".join(trunced)
    return trunced


tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def get_token_numbers(t):
    return len(tokenizer.encode(t))


def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    num_tokens = 0
    for message in messages:
        num_tokens += (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def get_roberta_token_numbers(t):
    return len(roberta_tokenizer(t)["input_ids"])


def to_tensor_dataset(args, data, tokenizer):
    pad_token = 0
    labels = torch.stack(
        [torch.tensor([(d["label"])], dtype=torch.float32) for d in data]
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
    dataset = TensorDataset(labels, all_input_ids, all_attention_masks)
    return dataset


def rand_throw(text, target_len):
    sentences = nltk.sent_tokenize(text)
    length = get_token_numbers(text)
    former_len = length
    while length > target_len:
        delete_sent = random.choice(sentences)
        sentences.remove(delete_sent)
        length -= get_token_numbers(delete_sent)
    text = " ".join(sentences)
    print.info("***shorten {} raw tokens into {} tokens.".format(former_len, length))
    return text


def rand_throw_abs(text, target_len):
    # allow to be slightly longer than the target length
    sentences = nltk.sent_tokenize(text)
    length = get_token_numbers(text)
    former_len = length
    while length > target_len:
        old_sentences = sentences
        delete_sent = random.choice(sentences)
        sentences.remove(delete_sent)
        new_length = length - get_token_numbers(delete_sent)
        if new_length < target_len:
            if abs(new_length - target_len) < abs(length - target_len):
                length = new_length
                break
            else:
                sentences = old_sentences
                break
        length -= get_token_numbers(delete_sent)
    text = " ".join(sentences)
    print("***shorten {} raw tokens into {} tokens.".format(former_len, length))
    return text


def number_h(num):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
import numpy as np


def compute_metrics(preds, labels, num_label=2):
    assert len(preds) == len(labels)
    if num_label == 1:
        align_preds = []
        given_labels = [0.0, 0.5, 1.0]
        for p in preds:
            temp = np.argmin([abs(g - p) for g in given_labels])
            temp = given_labels[temp]
            align_preds.append(temp)
        # convert into int.
        le = preprocessing.LabelEncoder()
        le.fit(given_labels)
        final_preds = le.transform(align_preds)
        labels = le.transform(labels)
        acc = accuracy_score(final_preds, labels)

        partial = 1 - align_preds.count(0.5) / len(align_preds)
        true_par, false_neu, major_err = 0, 0, 0
        for p, l in zip(final_preds, labels):
            if p != le.transform([0.5]).item() and p == l:
                true_par += 1
            if p == le.transform([0.5]).item() and l != le.transform([0.5]).item():
                false_neu += 1
            if (p == le.transform([0]).item() and l == le.transform([1]).item()) or (
                p == le.transform([1]).item() and l == le.transform([0]).item()
            ):
                major_err += 1
        return {
            "accuracy": acc,
            "partial": partial,
            "true_partical": true_par / len(align_preds),
            "false_neutral": false_neu / len(align_preds),
            "major_error": major_err / len(align_preds),
        }

    if num_label == 2:
        binary_pred = []
        for p in preds:
            if p < 0.5:
                binary_pred.append(0)
            else:
                binary_pred.append(1)
        acc = accuracy_score(binary_pred, labels)
        f1 = f1_score(y_true=labels, y_pred=binary_pred)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


def histogram_word(data, bins=200, logger=None):
    hist, edges = np.histogram(data, bins=bins, range=(0, 1))
    bin_widths = edges[1:] - edges[:-1]
    logger.info("Histogram: currently ignore zeros.")
    for count, width in zip(hist, bin_widths):
        percent = 100.0 * count / len(data)
        if logger == None:
            logger.info(f"{edges[0]:.4f} - {edges[0]+width:.4f}: {percent:.4f}%")
        else:
            if percent != 0:
                logger.info(f"{edges[0]:.4f} - {edges[0]+width:.4f}: {percent:.4f}%")
        edges = edges[1:]