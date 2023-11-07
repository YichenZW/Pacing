import argparse
import time
import json
import pickle

import openai

from story_generation.edit_module.entity import *
from story_generation.plan_module.outline import *


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


def recursive_traverse(outline_dict, parent):
    node_dict = (
        outline_dict["outline_item"] if "outline_item" in outline_dict else outline_dict
    )
    node = OutlinePiece(node_dict["main_plot"], parent)
    node.selected_entities = node_dict["characters"].split(", ")
    if "outline" in outline_dict:
        for child_key in sorted(list(outline_dict["outline"].keys())):
            recursive_traverse(outline_dict["outline"][child_key], node)
    if parent is not None:
        parent.children.append(node)
    return node


def main(args):
    input_data = json.load(open(args.input_json))
    input_txt = open(args.input_txt).read()
    node_num = len(input_txt.split("\n\n"))
    output_data = {}

    output_data["premise"] = input_data["premise"]

    setting_prompt = (
        "Consider the following premise and story outline:\n\n"
        + input_txt
        + '\n\n\n\nPlease infer a setting for this story. Your response should begin with "This story is set in" and should be no more than 15 words long.'
    )
    setting = openai_call("gpt-3.5-turbo", setting_prompt, max_tokens=30)[1]
    output_data["setting"] = setting

    character_set = set()
    for line in input_txt.split("\n"):
        if line.startswith("Characters:"):
            characters = line.split(":", maxsplit=1)[1].strip().split(",")
            characters = [c.strip() for c in characters]
            character_set.update(characters)
    character_set = list(character_set)
    character_strings = {}
    characters = ""
    for i, character in enumerate(character_set):
        character_description_prompt = (
            "Consider the following premise and story outline:\n\n"
            + input_txt
            + "\n\n\n\nPlease describe the character "
            + character
            + '. Your response should begin with "'
            + character
            + ' is" and should be no more than 15 words long.'
        )
        character_description = openai_call(
            "gpt-3.5-turbo", character_description_prompt, max_tokens=30
        )[1]
        entity = Entity(character, description=character_description, is_character=True)
        character_strings[character] = entity
        characters += (
            str(i + 1)
            + ".\n\nFull Name: "
            + character
            + "\n\nCharacter Portrait: "
            + character_description
            + "\n\n"
        )
    characters = characters.strip()
    output_data["character_strings"] = character_strings
    output_data["characters"] = characters

    infer_attributes_string = "\n\n".join(
        [output_data["premise"], output_data["setting"]]
        + [entity.description for entity in character_strings.values()]
    )
    output_data["infer_attributes_string"] = infer_attributes_string

    input_data["outline_item"] = {"main_plot": "", "characters": ""}
    outline = recursive_traverse(input_data, None)

    output_data["outline"] = outline

    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)


if __name__ == "__main__":
    from tqdm import trange

    for idx in trange(800, 900):
        if (
            not os.path.exists(f"path/to/concoct_output/{idx}_b3d3_plain.jsonl")
        ) or (not os.path.exists(f"path/to/concoct_output/{idx}_b3d3.txt")):
            print(f"{idx} not exist.")
            continue
        parser = argparse.ArgumentParser(description="Convert outline to doc format")
        parser.add_argument(
            "--input-json",
            default=f"path/to/concoct_output/{idx}_b3d3_plain.jsonl",
            help="input json file",
        )
        parser.add_argument(
            "--input-txt",
            default=f"path/to/concoct_output/{idx}_b3d3.txt",
            help="input txt file",
        )
        parser.add_argument(
            "--output",
            default=f"path/to/concoct_output/{idx}_b3d3.doc.pkl",
            help="output file",
        )
        args = parser.parse_args()
        main(args)
