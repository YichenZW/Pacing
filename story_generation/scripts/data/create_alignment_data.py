import argparse
import csv
import os

from tqdm import tqdm

from story_generation.common.util import add_general_args
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer
from story_generation.common.data.split_paragraphs import split_paragraphs, group_chunks

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)
    parser.add_argument('--save-csv', type=str, required=True, help='save to this csv file')
    parser.add_argument('--max-chunk-length', type=int, default=200, help='maximum length of chunks when splitting paragraphs')
    args = parser.parse_args()

    dataset = load_dataset(args)
    summarizer = load_summarizer(args)
    tab_token = summarizer.tokenizer.encode('\t')[0]
    logit_bias = {tab_token:-100}
    long_texts = dataset.load_long_texts(split='train')
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, 'w') as wf:
        writer = csv.writer(wf)
        writer.writerow(['text1', 'text2'])
        for text in tqdm(long_texts):
            chunks = group_chunks(split_paragraphs(text, mode='sentence', max_sentence_length=args.max_chunk_length), max_chunk_length=args.max_chunk_length)
            prompts = [chunk.strip() + '\n\n\n\nOne-sentence summary:\n\n\n\n' for chunk in chunks]
            summaries = [s.strip() for s in summarizer(prompts, num_completions=3, logit_bias=logit_bias, model_string=args.gpt3_model)]
            summaries = [summaries[i:i+3] for i in range(0, len(summaries), 3)]
            summaries = [min(group, key=len) for group in summaries] # get min length summary for each
            writer.writerow(['\t'.join(chunks), '\t'.join(summaries)])
