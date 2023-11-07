import os
import random
import math
import argparse
from utils import load_jsonl, trunc_text, to_tensor_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, DataLoader, SequentialSampler
import jsonlines
import json
from collections import Counter
random.seed(16)

Dataset_Name = "mainstoryonly_d2_sim_wo0.5"
MAIN_STORY_DIR = 'path/to/story'

MAIN_STORY_FILE_DICT = {
    "train": {"para": "paragraph_train_turbo_summary.jsonl",
              "chap": "chapter_train_turbo_summary.jsonl"},
    "val":   {"para": "paragraph_val_turbo_summary.jsonl",
              "chap": "chapter_val_turbo_summary.jsonl"},
    "test":  {"para": "paragraph_test_turbo_summary.jsonl",
              "chap": "chapter_test_turbo_summary.jsonl"},
}
MAX_SAMPLE_TIME = 200

contrieve_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
contriever = AutoModel.from_pretrained('facebook/contriever').cuda()

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", 
                        default=os.path.join(os.getcwd(), 'data'),
                        type=str,
                        help="")
    parser.add_argument("--model_name", default="roberta-base",
                        type=str,
                        help="Model type selected in the list: roberta-base")
    parser.add_argument("--output_dir",
                        default=os.path.join(os.getcwd(), 'results'),
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.",
                        )
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run test on the dev set.")
    parser.add_argument("--batch_size", default = 8, type = int, 
                        help = "Batch size per GPU/CPU for training.",)
    parser.add_argument("--learning_rate", default = 1e-5, type = float,
                        help = "The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default = 0.0, type = float,
                        help = "Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default = 1e-8,
                        type = float, help = "Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default = 1.0,
                        type = float, help = "Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument("--seed", type=int, default=31,
                        help="random seed for initialization")
    parser.add_argument("--device", type=str, default='cuda',
                        help="")   
    parser.add_argument("--load_best", type=bool, default=False,
                        help="")        
    parser.add_argument("--num_label", type=int, default=1,
                        help="")        
    parser.add_argument("--loss_type", type=str, default="BCEWithLogits",
                        help="All the loss types are list as follows: BCEWithLogits, MSE, BCE") 
    args = parser.parse_args()
    return args

args = init_args()

def sample_length(min_val = 25, max_val = 180):
    log_min_val = math.log(min_val)
    log_max_val = math.log(max_val)
    random_log_num = math.exp(random.uniform(log_min_val, log_max_val))
    random_int = int(round(random_log_num))
    return random_int

def load_dataset(task, size=5000, d_name=None):
    print("***Loading Fixed Dataset: {} {} {}***".format(task, size, d_name))
    dataset = torch.load('data/{}_{}_{}.pth'.format(task, size, d_name))
    print("***Loaded***")
    return dataset

visited_samples = []

def visit_record_formating(sample):
    if 'sub_index' in sample.keys():
        visited_item = {'sub_index': sample['sub_index'], 
                        'index': sample['index']}
    elif 'para_idx' in sample.keys():
        visited_item = {'para_idx': sample['para_idx'], 
                        'index': sample['index']}
    else:
        raise ValueError("Sample Key Error")
    return visited_item

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

# Less reload time.
task = 'train'
print("*Preloading Whole Training Raw Data*")
main_story_para_dir = os.path.join(MAIN_STORY_DIR, MAIN_STORY_FILE_DICT[task]["para"])
print("     {}, paralevel: {}".format(task, main_story_para_dir))
main_story_para = load_jsonl(main_story_para_dir)
main_story_chap_dir = os.path.join(MAIN_STORY_DIR, MAIN_STORY_FILE_DICT[task]["chap"])
print("     {}, chaplevel: {}".format(task, main_story_chap_dir))
main_story_chap = load_jsonl(main_story_chap_dir)

def build_dataset(args, tokenizer, task, size=1000, to_jsonl=True, to_dataset=True, overwrite=False, for_example=False, epoch=None, seperate_by_book=False, book_per_epochs=5, update_epoch=5, similarity_comp=False):
    """
    task should be one of the following str: 'train', 'val', 'test' 
    to_jsonl save a jsonline file for the dataset as a dict type.
    to_dataset transfer the raw dataset into dataset type (tokernize to embeding) then return
    for_example use when testing baseline api-model with few shots
    overwrite controls whether overwrite the exist same-name file or not.
    epoch epoch number during training.
    seperate by book means whether feeding a dataset by sequence of books.
    book per epochs means how many books feeded one time.
    update epoch, feed new books batch per update epochs.
    similar_comp, whether pairing base on similarity.
    """
    print("***Building {} {} Dataset, Loading Main Story***".format(task, size))
    global main_story_para
    global main_story_chap
    global visited_samples
    # loaded training dataset in advance
    if not task == 'train': 
        main_story_para_dir = os.path.join(MAIN_STORY_DIR, MAIN_STORY_FILE_DICT[task]["para"])
        print("     {}, paralevel: {}".format(task, main_story_para_dir))
        main_story_para = load_jsonl(main_story_para_dir)

        main_story_chap_dir = os.path.join(MAIN_STORY_DIR, MAIN_STORY_FILE_DICT[task]["chap"])
        print("     {}, chaplevel: {}".format(task, main_story_chap_dir))
        main_story_chap = load_jsonl(main_story_chap_dir)
    
    if not seperate_by_book:
        main_story = (None, main_story_chap, main_story_para)

        data_set = []
        data_set_labels = []

        empty_time = 0
        for sam_idx in tqdm(range(size), desc="Sample"):
            set_item = dict() 
            # Pick Level
            levels = [-1, -1]
            element = [None, None]
            for eidx in [0, 1]:
                levels[eidx] = random.randint(1, 2) # 1 means chapter-level, 2 means para-level

            if levels[0] == levels[1]:
                label = 0.5
            elif levels[0] > levels[1]:
                label = 0
            elif levels[0] < levels[1]:
                label = 1
            set_item['label'] = label

            # Pick element 0
            eidx = 0
            trg_len_e0 = sample_length()
            sample_id = random.choice(range(len(main_story[levels[eidx]])))
            sample = main_story[levels[eidx]][sample_id]
            sample_rec = visit_record_formating(sample)
            sample_time = 1
            while (sample['turbo_len'] < trg_len_e0 or sample_rec in visited_samples) and sample_time < MAX_SAMPLE_TIME : 
                sample_id = random.choice(range(len(main_story[levels[eidx]])))
                sample = main_story[levels[eidx]][sample_id]
                sample_rec = visit_record_formating(sample)
                sample_time += 1
            if sample_time == MAX_SAMPLE_TIME:
                continue
            sample_text = sample['text']
            trunc_sample_text = trunc_text(sample_text, trg_len_e0)
            
            element[0] = trunc_sample_text
            ele0_index = sample['index']
            
            visited_samples.append(sample_rec)

            #Pick element 1
            eidx = 1
            do_same_length = random.choice([True, False])
            if do_same_length:
                trg_len_e1 = trg_len_e0
            else:
                trg_len_e1 = sample_length()
            if not similarity_comp:
                sample = random.choice(main_story[levels[eidx]])
                sample_rec = visit_record_formating(sample)
                sample_time = 1
                while (sample['turbo_len'] < trg_len_e1 or sample_rec in visited_samples) and sample_time < MAX_SAMPLE_TIME : 
                    sample = random.choice(main_story[levels[eidx]])
                    sample_rec = visit_record_formating(sample)
                    sample_time += 1
                if sample_time == MAX_SAMPLE_TIME:
                    continue
                sample_text = sample['text']
                trunc_sample_text = trunc_text(sample_text, trg_len_e1)
            else: # similarity comparision
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                target_sentence = element[0]
                target_input = contrieve_tokenizer(target_sentence, padding=True, truncation=True, return_tensors='pt')
                target_input = target_input.to(device)
                contriever.eval()
                with torch.no_grad():
                    target_output = contriever(**target_input)
                target_embedding = mean_pooling(target_output[0], target_input['attention_mask'])
                similarity_scores = []

                constituency_large = [s for s in main_story[levels[eidx]] if (s['index']['bid']==ele0_index['bid'] and visit_record_formating(s) not in visited_samples)]
                if constituency_large == []:
                    empty_time += 1
                    print("Constituency is Empty for ", ele0_index['bid'], ". No. {} time for this epoch.".format(empty_time))
                    # Softer constraint, don't need come from same book
                    constituency_large = [s for s in main_story[levels[eidx]] if (visit_record_formating(s) not in visited_samples)]
                    if constituency_large == []:
                        print("Constituency is Empty Since Visited.")
                        break

                if len(constituency_large) <= 64:
                    constituency_len = len(constituency_large)
                    constituency = constituency_large
                else:
                    constituency_len = int(math.sqrt(len(constituency_large)))
                    constituency = random.choices(constituency_large, k=constituency_len)
                constituency_text = [trunc_text(s['text'], trg_len_e1) for s in constituency]
                for batch_id in range((constituency_len-1) // 16 + 1):
                    con_batch = constituency_text[batch_id*16: min((batch_id+1)*16, len(constituency_large))]
                    con_input = contrieve_tokenizer(con_batch, padding=True, truncation=True, return_tensors='pt')
                    con_input = con_input.to(device)
                    with torch.no_grad():
                        con_output = contriever(**con_input)
                    con_embeddings = mean_pooling(con_output[0], con_input['attention_mask'])
                    similarity_scores.extend([target_embedding @ embed for embed in con_embeddings])
                assert len(similarity_scores) == len(constituency), "Incorrect Similarity Score Length"

                max_text_id = similarity_scores.index(max(similarity_scores))
                trunc_sample_text = constituency_text[max_text_id]
                trunc_sample = constituency[max_text_id]
                sample_rec = visit_record_formating(trunc_sample)
                 
            element[1] = trunc_sample_text
            visited_samples.append(sample_rec)

            set_item['text'] = element[0] + ' <sep> ' + element[1]

            data_set.append(set_item)

    data_set_labels = [ds['label'] for ds in data_set]
    print("= Valid Data Num: ", len(data_set_labels))
    print("= Visited Samples: ", len(visited_samples))
    print("= Label Distribution: ", Counter(data_set_labels))
    print("= Empty Time: ", empty_time)
    assert empty_time < size * 0.95, "! Too many empty sample!"
    
    if to_jsonl:
        if for_example:
            return data_set
        if os.path.exists('data/{}_{}_{}.jsonl'.format(task, size, Dataset_Name)) and not overwrite and not for_example:
            print("Caution: Can't OVERWRITE the file. Exiting. If you want to do so, set the arguement to Ture.")
            return -1
        if os.path.exists('data/{}_{}_{}.jsonl'.format(task, size, Dataset_Name)) and overwrite:
            print("Caution: You are going to OVERWRITE file ", 'data/{}_{}_{}.jsonl'.format(task, size, Dataset_Name))
            user_in = input("Type Yes to continue")
            if user_in != "Yes":
                print("Didn't OVERWRITE the file. Exiting.")
                return -1
            with open('data/{}_{}_{}.jsonl'.format(task, size, Dataset_Name), 'w') as output:
                output.truncate(0)
        print("     1) Saving into Jsonline File")
        with jsonlines.open('data/{}_{}_{}.jsonl'.format(task, size, Dataset_Name), 'a') as output:
            for item in data_set:
                output.write(item)
        print("     1) Saved")
    if to_dataset:       
        print("     2) Transfering into Tensor Embedding Dataset File")
        data_set = to_tensor_dataset(args, data_set, tokenizer)
        print("     2) Transfered") 
    return data_set

def save_for_val(size = 5000):  
    tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,)
    val_set = build_dataset(args, tokenizer, task="val", size=size, to_jsonl=True, to_dataset=True, similarity_comp=True)
    if os.path.exists('data/val_{}_{}.pth'.format(size, Dataset_Name)):
        print("Caution: Not overwrite the file. Break!")
        return -1
    print("***Saving***")
    torch.save(val_set, 'data/val_{}_{}.pth'.format(size, Dataset_Name))
    print("***Saved to ", 'data/val_{}_{}.pth'.format(size, Dataset_Name), " ***")

def save_for_test(size = 5000):
    tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,)
    test_set = build_dataset(args, tokenizer, task="test", size=size, to_jsonl=True, to_dataset=True, similarity_comp=True)
    if os.path.exists('data/test_{}_{}.pth'.format(size, Dataset_Name)):
        print("Caution: Not overwrite the file. Break!")
        return -1
    print("***Saving***")
    torch.save(test_set, 'data/test_{}_{}.pth'.format(size, Dataset_Name))       
    print("***Saved to ", 'data/test_{}_{}.pth'.format(size, Dataset_Name), " ***") 

def check_for_eval():
    size = 5000
    tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,)
    test_set = build_dataset(args, tokenizer, task="val", size=size, to_dataset=False)

if __name__ == "__main__":
    save_for_val()
    save_for_test()