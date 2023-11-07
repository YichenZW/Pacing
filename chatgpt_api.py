import argparse
import os
import time
import logging
import json
import tiktoken
import torch
from transformers import AutoTokenizer
import openai

from doc_generation_toolkit.common.summarizer.models.abstract_summarizer import (
    AbstractSummarizer,
)
from doc_generation_toolkit.common.data.split_paragraphs import cut_last_sentence

GPT3_END = "THE END."
PRETRAINED_MODELS = [
    "ada",
    "babbage",
    "curie",
    "davinci",
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
    "text-davinci-002",
    "text-davinci-003",
]

openai.api_key = os.environ["OPENAI_API_KEY"]

class ChatGPT3Summarizer(AbstractSummarizer):
    def __init__(self, args, logger):
        assert args.gpt3_model is not None
        self.model = args.gpt3_model
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.args = args
        self.controller = None
        self.summarize = {
            "num_queries": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "maximal_prompt_tokens": 0,
        }
        self.logger = logger

    @torch.no_grad()
    def __call__(
        self,
        texts,
        suffixes=None,
        max_tokens=None,
        top_p=None,
        temperature=None,
        retry_until_success=True,
        stop=None,
        logit_bias=None,
        num_completions=1,
        cut_sentence=False,
        model_string=None,
    ):
        assert type(texts) == list
        self.summarize["num_queries"] += len(texts)
        if logit_bias is None:
            logit_bias = {}
        if suffixes is not None:
            raise NotImplementedError
        if model_string is None:
            pass
        else:
            model_string = None
        if self.controller is None:
            return self._call_helper(
                texts,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                retry_until_success=retry_until_success,
                stop=stop,
                logit_bias=logit_bias,
                num_completions=num_completions,
                cut_sentence=cut_sentence,
                model_string=model_string,
            )
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _call_helper(
        self,
        texts,
        max_tokens=None,
        top_p=None,
        temperature=None,
        retry_until_success=True,
        stop=None,
        logit_bias=None,
        num_completions=1,
        cut_sentence=False,
        model_string=None,
    ):
        assert model_string in PRETRAINED_MODELS or model_string is None

        if logit_bias is None:
            logit_bias = {}

        outputs = []
        for i in range(len(texts)):
            text = texts[i]
            prompt = text

            retry = True
            num_fails = 0
            while retry:
                try:
                    context_length = len(self.tokenizer.encode(prompt))
                    self.summarize["total_prompt_tokens"] += context_length
                    self.summarize["maximal_prompt_tokens"] = max(
                        self.summarize["maximal_prompt_tokens"], context_length
                    )
                    if context_length > self.args.max_context_length:
                        if self.logger is None:
                            print(
                                "context length"
                                + " "
                                + str(context_length)
                                + " "
                                + "exceeded artificial context length limit"
                                + " "
                                + str(self.args.max_context_length)
                            )
                        else:
                            self.logger.warning(
                                "context length"
                                + " "
                                + str(context_length)
                                + " "
                                + "exceeded artificial context length limit"
                                + " "
                                + str(self.args.max_context_length)
                            )
                        time.sleep(
                            1
                        )  # similar interface to gpt3 query failing and retrying
                        assert False
                    if max_tokens is None:
                        max_tokens = min(
                            self.args.max_tokens,
                            self.args.max_context_length - context_length,
                        )
                    engine = self.model if model_string is None else model_string
                    if engine == "text-davinci-001":
                        engine = "text-davinci-002"
                    completion = openai.ChatCompletion.create(
                        model=engine,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                        if temperature is not None
                        else self.args.summarizer_temperature,
                        stop=stop,
                        logit_bias=logit_bias,
                        n=num_completions,
                    )
                    gpt3_pair = {
                        "prompt": prompt,
                        "completion": [
                            completion["choices"][j]["message"]["content"]
                            for j in range(num_completions)
                        ],
                    }
                    retry = False
                except Exception as e:
                    if self.logger is None:
                        print(str(e))
                    else:
                        self.logger.warning(str(e))
                    retry = retry_until_success
                    num_fails += 1
                    if num_fails > 20:
                        raise e
                    if retry:
                        if self.logger is None:
                            print(f"retrying... sleeping {num_fails} seconds...")
                        else:
                            self.logger.warning(
                                f"retrying... sleeping {num_fails} seconds..."
                            )
                        time.sleep(num_fails)
            outputs += [
                completion["choices"][j]["message"]["content"]
                for j in range(num_completions)
            ]
        if cut_sentence:
            for i in range(len(outputs)):
                if len(outputs[i].strip()) > 0:
                    outputs[i] = cut_last_sentence(outputs[i])
        engine = self.model if model_string is None else model_string
        self.summarize["total_output_tokens"] += sum(
            [len(self.tokenizer.encode(o)) for o in outputs]
        )
        return outputs


def load_model(temp=0.5, logger=None):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    args = argparse.Namespace()
    args.gpt3_model = "gpt-3.5-turbo"
    args.max_tokens = 4096  # output length
    args.max_context_length = 4096  # input length
    args.summarizer_temperature = temp
    args.summarizer_frequency_penalty = 0.0
    args.summarizer_presence_penalty = 0.0
    gpt3 = ChatGPT3Summarizer(args, logger)
    return gpt3


def load_model2classification(model="gpt-3.5-turbo"):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    args = argparse.Namespace()
    args.gpt3_model = model
    args.max_tokens = 1024  # output length
    args.max_context_length = 3985  # input length
    args.summarizer_temperature = 0
    gpt3 = ChatGPT3Summarizer(args)
    return gpt3


def determistic_simple_API(model, text, logit_bias=None):
    ChatList = [{"role": "user", "content": text}]
    if logit_bias == None:
        logit_bias = {}
    response = openai.ChatCompletion.create(
        model=model,
        messages=ChatList,
        temperature=0,
        logit_bias=logit_bias,
    )["choices"][0]["message"]["content"]
    return response


if __name__ == "__main__":
    # A text case
    texts = [
        """Premise: An ordinary high school student discovers that they possess an extraordinary ability to manipulate reality through their dreams.    As they struggle to control this power and keep it hidden from those who would exploit it, they are drawn into a dangerous conflict between secret organizations vying for control over the fate of the world.

Outline:

Point 2.1.2
Main plot: Alex struggles to control their power
Begin Event: Alex accidentally manipulates reality in their dream
End Event: Alex seeks guidance from Mr. Lee to control their power
Characters: Alex, Mr. Lee


Can you break down point 2.1.2 into less than 3 independent, chronological and same-scaled outline points? Also, assign each character a name. Please use the following template with "Main Plot", "Begin Event". "End Event" and "Characters". Do not answer anything else.

Point 2.1.2.1
Main plot: [TODO]
Begin Event: [TODO]
End Event: [TODO]
Characters: [TODO]

Point 2.1.2.2
Main plot: [TODO]
Begin Event: [TODO]
End Event: [TODO]
Characters: [TODO]

...
"""
    ]
    print(determistic_simple_API("gpt-3.5-turbo", texts[0]))
