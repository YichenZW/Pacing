import os
import logging
import torch
import json
import argparse
import random
import numpy as np
import datetime
import wandb

from utils import number_h, compute_metrics, histogram_word
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
        help="",
    )
    parser.add_argument(
        "--model_name",
        default="roberta-large",
        type=str,
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
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument(
        "--do_eval", default=True, help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", default=True, help="Whether to run test on the dev set."
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


MODEL_PATH = "models/Concrete_RobertaClsf_{}.pt".format(current_time)

if args.loss_type == "BCEWithLogits":
    loss_fn = torch.nn.BCEWithLogitsLoss()
elif args.loss_type == "MSE":
    loss_fn = torch.nn.MSELoss()


def train(args, model, tokenizer, eval_dataset):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info("Total Params: %s", number_h(total_params))
    logger.info("Total Trainable Params: %s", number_h(total_trainable_params))

    train_dataset_len = 1000
    t_total = train_dataset_len * args.num_train_epochs / 2
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset_len)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch")
    tot_loss, global_step = 0.0, 0
    best_valloss = 0.0
    best_acc = 0.0
    for idx, _ in enumerate(train_iterator):
        epoch_loss = 0.0

        logger.info(
            "= Learning Rate: {} =".format(
                optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        wandb.log({"train/lr": optimizer.state_dict()["param_groups"][0]["lr"]})

        # pair training dataset
        train_dataset = build_dataset(
            args,
            tokenizer,
            "train",
            train_dataset_len,
            to_jsonl=False,
            to_dataset=True,
            epoch=idx,
            seperate_by_book=args.seperate_by_book,
            similarity_comp=args.sim_comp,
        )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )
        wandb.log({"train/epoch": idx})
        with logging_redirect_tqdm():
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
                outputs = model(**inputs)
                logits = outputs[0]  # torch.size([8, 2])
                if args.loss_type == "MSE":
                    scores = F.softmax(logits, dim=1)[:, 0].squeeze(-1)
                    loss = loss_fn(scores, batch[0])  # torch.size([8])
                elif args.loss_type == "BCEWithLogits":
                    scores = logits[:, 0].squeeze(-1)
                    if scores.size() == torch.Size([]):
                        scores = scores.unsqueeze(0)
                    loss = loss_fn(scores, batch[0])
                loss.backward()
                tot_loss += loss.item()
                epoch_loss += loss.item()
                epoch_iterator.set_description(
                    "loss {}".format(round(epoch_loss / (step + 1), 4))
                )
                wandb.log({"train/loss": epoch_loss / (step + 1)})
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            # eval each batch
            if args.do_eval:
                res, valloss = eval(
                    args, eval_dataset, model, tokenizer, mode="eval", epoch=idx
                )
                if res["accuracy"] > best_acc:
                    logger.info(
                        "***Best Epoch, Saving Model Into {}***".format(MODEL_PATH)
                    )
                    best_acc = res["accuracy"]
                    torch.save(model, MODEL_PATH)

    return tot_loss / global_step


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
    result = compute_metrics(preds_softmax, labels, num_label=args.num_label)

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
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, hidden_dropout_prob=0.2, num_labels=2
    )

    model.to(args.device)

    # Using a already paired dataset for val set and text set
    eval_dataset = load_dataset("val", 5000, "mainstoryonly_d2")
    test_dataset = load_dataset("test", 5000, "mainstoryonly_d2")

    if args.do_train:
        train(args, model, tokenizer, eval_dataset)

    if args.do_test:
        if args.load_best:
            model = torch.load(MODEL_PATH)
        eval(args, test_dataset, model, tokenizer, mode="test")


if __name__ == "__main__":
    main()
