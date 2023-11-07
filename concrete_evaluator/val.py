import os
import logging
import torch
import json
import argparse
import random
import numpy as np
import datetime
import wandb

from utils import number_h, compute_metrics, histogram_word, to_tensor_dataset
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.optim import AdamW
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from dataset import load_dataset, build_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/{current_time}-ConcreteDec.log"
file_handler = logging.FileHandler(log_filename)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.getcwd(), "data"),
        type=str,
        # required=True,
        help="",
    )
    parser.add_argument(
        "--model_name",
        default="roberta-large",
        type=str,
        # required=True,
        help="Model type selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.getcwd(), "results"),
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", default=False, help="Whether to run training.")
    parser.add_argument(
        "--do_eval", default=True, help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", default=False, help="Whether to run test on the dev set."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=6e-6,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=28,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=31, help="random seed for initialization"
    )
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--load_best", type=bool, default=False, help="")
    parser.add_argument("--num_label", type=int, default=1, help="")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="BCEWithLogits",
        help="All the loss types are list as follows: BCEWithLogits, MSE, BCE",
    )
    parser.add_argument(
        "--seperate_by_book",
        type=bool,
        default=False,
        help="Give some books at some specific epoches to reduce the semantic shift",
    )
    parser.add_argument(
        "--sim_comp", type=bool, default=True, help="Using Contriever to pair data."
    )
    args = parser.parse_args()
    return args


args = init_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_jsonl(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        data.append(item)
    logger.info("   successfully loaded {} lines".format(len(data)))
    return data


if args.loss_type == "BCEWithLogits":
    loss_fn = torch.nn.BCEWithLogitsLoss()
elif args.loss_type == "MSE":
    loss_fn = torch.nn.MSELoss()


def eval(args, eval_dataset, model, tokenizer, mode, epoch=None):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    if epoch == None:
        logger.info("***** Running {} *****".format(mode))
    else:
        logger.info("*** Running {} Epoch {} ***".format(mode, epoch))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss, eval_step = 0.0, 0
    preds = None
    with logging_redirect_tqdm():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
                outputs = model(**inputs)  # torch.size([8, 2])
                scores_softmax = F.softmax(outputs.logits, dim=1)[:, 0].squeeze(-1)
                scores = outputs.logits[:, 0].squeeze(-1)
                loss = loss_fn(scores, batch[0])
            eval_step += 1
            if preds is None:
                preds = scores.detach().cpu().numpy()
                preds_softmax = scores_softmax.detach().cpu().numpy()
                labels = batch[0].detach().cpu().numpy()
            else:
                preds = np.append(preds, scores.detach().cpu().numpy(), axis=0)
                preds_softmax = np.append(
                    preds_softmax, scores_softmax.detach().cpu().numpy(), axis=0
                )
                labels = np.append(labels, batch[0].detach().cpu().numpy(), axis=0)
    preds = preds.reshape(-1)
    preds_softmax = preds_softmax.reshape(-1)
    histogram_word(preds_softmax, logger=logger)

    logger.info(preds[:10])
    logger.info(preds_softmax[:10])
    logger.info(labels[:10])

    logger.info(preds[-10:])
    logger.info(preds_softmax[-10:])
    logger.info(labels[-10:])
    result = compute_metrics(preds_softmax, labels, num_label=2)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        wandb.log({"val/{}".format(key): result[key]})
    logger.info("  %s = %s", "BCELossWithLogits", str(loss))
    wandb.log({"val/BCELoss": loss})
    return result, loss


def main():
    set_seed(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Current Device: %s", args.device)

    wandb.init(
        project="concrete",
        name="Concrete",
        reinit=True,
    )
    wandb.config.update(args, allow_val_change=True)
    wandb.define_metric("train/loss")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )
    model = torch.load("models/Concrete_RobertaClsf_st-nv.pt")
    model.to(args.device)

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

    eval_dataset = load_jsonl("human_eval/human_annotated/short55-too_detailed.jsonl")
    eval_dataset = to_tensor_dataset(args, eval_dataset, tokenizer)
    test_dataset = load_dataset("test", 5000, "mainstoryonly_d2")
    test_dataset = to_tensor_dataset(args, test_dataset, tokenizer)

    if args.do_eval:
        print(eval(args, eval_dataset, model, tokenizer, mode="eval"))

    if args.do_test:
        print(eval(args, test_dataset, model, tokenizer, mode="test"))


if __name__ == "__main__":
    main()
