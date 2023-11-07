import argparse
import time
import json
import pickle

import openai
from tqdm import trange
from story_generation.edit_module.entity import *
from story_generation.plan_module.outline import *


def openai_call(engine, messages, **kwargs):
    fails = 0
    while True:
        try:
            if (
                "gpt-3.5-turbo" in engine
                or "gpt-4" in engine
                or "gpt-3.5-turbo-16k" in engine
            ):
                if type(messages) == str:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an intelligent AI assistant.",
                        },
                        {"role": "user", "content": messages},
                    ]
                result = openai.ChatCompletion.create(
                    model=engine,
                    messages=messages,
                    **kwargs,
                )
                text = result["choices"][0]["message"]["content"]
            else:
                result = openai.Completion.create(
                    engine=engine,
                    prompt=messages,  # should be a string prompt in this case
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


class OutlinePoint:
    def __init__(self, idx, outline, passage="[TODO]"):
        self.idx = idx
        self.plain_idx = None
        self.outline = outline["main_plot"]
        self.passage = passage

    def outline_str(self):
        return f"Point {self.plain_idx}: {self.outline}\n"

    def __str__(self):
        return f"Point {self.plain_idx}:\nOutline: {self.outline}\nChapter Content: {self.passage}\n\n"


def main():
    for pid in trange(800, 900):
        print(f"***generating {pid} story***")
        OUTLINE_DIR = f"output/path/to/{pid}_b3d3_plain.jsonl"
        OUTPUT_DIR = f"output/path/to/baseline_{pid}_story.pkl.final.txt"
        if not os.path.exists(OUTLINE_DIR):
            print(f"{pid} not exist.")
            continue
        with open(OUTLINE_DIR, "r") as file:
            outline = json.load(file)

        leaves = []

        def find_leaves(idx, outline):
            if "outline" not in outline.keys():
                leaves.append(OutlinePoint(idx, outline))
                return 0
            for idx, child in outline["outline"].items():
                find_leaves(idx, child)

        find_leaves(-1, outline)
        print(len(leaves))
        if len(leaves) <= 20:
            print(f"{pid} too short.")
            continue
        for p_idx, l in enumerate(leaves):
            l.plain_idx = p_idx + 1
        premise = "Premise:" + outline["premise"] + "\n"
        entire_outline = "".join([l.outline_str() for l in leaves])

        BATCH_SIZE_OUTLINE = 5
        for epoch in range(0, len(leaves) // BATCH_SIZE_OUTLINE + 1):
            print(
                f"...generating {BATCH_SIZE_OUTLINE*epoch}:{min(BATCH_SIZE_OUTLINE*(epoch+1), len(leaves))}..."
            )
            batch_leaves = leaves[: min(BATCH_SIZE_OUTLINE * (epoch + 1), len(leaves))]
            setting_prompt = (
                "Generate passages with the given premise, generated story, and entire outline.\n\n"
                + premise
                + "\n"
                + "Entire Outline:\n"
                + entire_outline
                + "\nExpand each outline point into passages. Each passage should have the same length (around 75 words). The story based on the outline are as follows, fill into the [TODO] blanks."
            )
            story_prompt = (
                "".join([str(l) for l in batch_leaves])
                + "Output passages for [TODO] blanks. Do not directly copy the outline to the passages."
            )

            prompt = setting_prompt + story_prompt
            retry_count, max_retries = 0, 3
            while retry_count < max_retries:
                try:
                    _, story = openai_call("gpt-3.5-turbo-16k", prompt)
                    story = story.split("Note:")[0]
                    passages = story.split("\n\n")
                    for p in passages:
                        lines = p.split("\n")
                        current_point = None
                        for line in lines:
                            if line.startswith("Point"):
                                match = re.search(r"Point (\d+):", line)
                                if match:
                                    current_point = int(match.group(1))
                                else:
                                    raise ValueError
                            elif line.startswith("Chapter Content"):
                                if (
                                    current_point
                                    and leaves[current_point - 1].passage == "[TODO]"
                                ):
                                    leaves[current_point - 1].passage = line.replace(
                                        "Chapter Content: ", ""
                                    )
                                else:
                                    raise ValueError
                    break
                except ValueError:
                    print(f"faced an value error, try {retry_count}...")
                    print(p)
                    retry_count += 1

        final_story = "".join([str(l) for l in leaves])
        with open(OUTPUT_DIR, "w") as file:
            file.write(final_story)
        with open(OUTPUT_DIR + ".story", "w") as file:
            file.write("\n\n".join([l.passage for l in leaves]))

if __name__ == "__main__":
    main()
