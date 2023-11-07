import openai
import time
import os
import nltk
from nltk.tokenize import word_tokenize

def openai_call(engine, messages, **kwargs):
    fails = 0
    while True:
        try:
            if "gpt-3.5-turbo" in engine or "gpt-4" in engine:
                if type(messages) == str:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an intelligent AI assistant.",
                        },
                        {"role": "user", "content": messages},
                    ]
                result = openai.ChatCompletion.create(
                    model=engine, messages=messages, **kwargs
                )
                text = result["choices"][0]["message"]["content"]
            else:
                result = openai.Completion.create(
                    engine=engine,
                    prompt=messages,  # should be a string
                    logprobs=5,
                    **kwargs,
                )
                text = result.choices[0]["text"]
            break
        except Exception as e:
            fails += 1
            time.sleep(fails)
            print(e)
            print("failed {} times, retrying...".format(fails))
    return result, text


def eval(text1, text2):
    settings = "Here are two story excerpts.\n\n\n\n"
    questions = """


The shown stories are parts of whole stories. You shouldn't be concerned about the completeness of the plot. 

Answer the following question:

Overall, which story's plot is closer to the premise? A / B

Please answer with a single letter (A or B).
"""
    messages = (
        settings + "Story A:\n\n" + text1 + "\n\n\n\nStory B:\n\n" + text2 + questions
    )
    _, result = openai_call("gpt-4", messages, temperature=0.0)
    result_list = list(result)
    if len(result_list) == 1 and (
        set(result_list) == {"A"} or set(result_list) == {"B"}
    ):
        result_int_list = [s == "B" for s in result_list]
    else:
        print(f"False output format: {result}")
        result_int_list = [0.5]
    return result_int_list


def split_excerpt(while_text, excerpt_len=1000):
    passages = while_text.split("\n\n")
    excerpts = []
    temp_ex = []
    temp_len = 0
    for p in passages:
        temp_ex.append(p)
        temp_len += len(word_tokenize(p))
        if temp_len < excerpt_len:
            continue
        else:
            excerpts.append("\n\n".join(temp_ex))
            # print(len(word_tokenize(excerpts[-1])))
            temp_ex = []
            temp_len = 0
    return excerpts


def main():
    scores = 0
    counts = 0
    for pid in range(800, 900):
        baseline_dir = f"output/path/to/baseline_{pid}_story.pkl.final.txt.story"
        concoct_dir = f"output/path/to/concoct_{pid}_story.pkl.final.txt.story"
        if not (os.path.exists(baseline_dir) and os.path.exists(concoct_dir)):
            print(pid, "Missing files")
            continue
        with open(baseline_dir, "r") as file:
            baseline_str = file.read()
        with open(concoct_dir, "r") as file:
            concoct_str = file.read()
        baseline_excerpts = split_excerpt(baseline_str)
        concoct_excerpts = split_excerpt(concoct_str)
        minlen = min(len(baseline_excerpts), len(concoct_excerpts))
        baseline_excerpts = baseline_excerpts[:minlen]
        concoct_excerpts = concoct_excerpts[:minlen]
        for b_ex, c_ex in zip(baseline_excerpts, concoct_excerpts):
            temp = eval(b_ex, c_ex)
            counts += 1
            scores += temp[0]
            temp = eval(c_ex, b_ex)
            counts += 1
            scores += 1 - temp[0]
        print(counts)
        print(scores / counts)


if __name__ == "__main__":
    main()
