import jsonlines
import copy
import logging

logger = logging.getLogger()
from strsimpy.cosine import Cosine

cosine = Cosine(2)
import random

random.seed(10)
from tqdm import trange
from chatgpt_api import load_model
from ranker4outline import Ranker


class OutlineItem:
    def __init__(self, str_outline, parent):
        str_outline = str_outline.split("\n")
        try:
            idx, main_plot, characters = str_outline[:3]
            self.idx = idx
            self.main_plot = main_plot[len("Main plot: ") :]
            self.characters = characters[len("Characters: ") :]
        except:
            self.idx = None
            self.main_plot = None
            self.characters = None
            logger.warning("Initalize OutlineItem: str_outline is not valid")

        self.parent = parent

    def __str__(self):
        return f"""Point {self.idx}
Main plot: {self.main_plot}
Characters: {self.characters}

"""

    def str(self, max_depth=None):
        if max_depth == 0:
            return ""
        else:
            return str(self)

    def get_dict(self, max_depth=None):
        return {"main_plot": self.main_plot, "characters": self.characters}

    def get_list_ancestor(self):
        ancestor = []
        assert self.parent is not None, "Outline Item's Parent is None"
        # add parent and parent's brothers into the top of the list
        for p in self.parent.parent.son_outlines:
            if type(p) == OutlineItem:
                ancestor.append(p)
            elif type(p) == Outline:
                ancestor.append(p.outline_item)


class Outline:
    def __init__(
        self,
        son_outlines,
        parrallel_son_outlines=None,
        premise=None,
        outline_item=None,
        parrallel_outline_item=None,
    ):
        if premise is not None:
            self.premise = premise
            self.is_root = True
            self.parent = self
        else:
            self.is_root = False
            self.outline_item = outline_item
            self.parrallel_outline_item = parrallel_outline_item
            self.parent = None
        try:
            self.son_outlines = [OutlineItem(outline, self) for outline in son_outlines]
        except:
            self.son_outlines = None
        try:
            self.parrallel_son_outlines = (
                [OutlineItem(outline, self) for outline in parrallel_son_outlines]
                if parrallel_son_outlines is not None
                else self.son_outlines
            )
        except:
            self.parrallel_son_outlines = None

    def __str__(self):
        if self.is_root:
            result_str = f"Premise: {self.premise}\n\nOutline:\n\n"
            for outline in self.son_outlines:
                result_str += str(outline)
        else:
            result_str = str(self.outline_item)
            for outline in self.son_outlines:
                result_str += str(outline)
        return result_str

    def str(self, max_depth=None):
        if max_depth == 0:
            return ""
        if max_depth is not None:
            max_depth -= 1
        if self.is_root:
            result_str = f"Premise: {self.premise}\n\nOutline:\n\n"
            for outline in self.son_outlines:
                result_str += outline.str(max_depth=max_depth)
        else:
            result_str = str(self.outline_item)
            for outline in self.son_outlines:
                result_str += outline.str(max_depth=max_depth)
        return result_str

    # Simple version
    def get_prompt(self, idx):
        if self.is_root:
            result_str = f"Premise: {self.premise}\n\nOutline:\n\n"
            for outline in self.son_outlines:
                if type(outline) == OutlineItem:
                    if outline.idx != idx:
                        continue
                    result_str += str(outline)
                else:
                    result_str += outline.get_prompt(idx)
        else:
            result_str = str(self.outline_item)
            # if the outline_item is not exactly the idx, reject it
            if self.outline_item.idx != idx:
                result_str = ""
            # check if the idx is the prefix of the outline_item.idx
            if self.outline_item.idx == idx[: len(self.outline_item.idx)]:
                for outline in self.son_outlines:
                    if type(outline) == OutlineItem:
                        if outline.idx != idx:
                            continue
                        result_str += str(outline)
                    else:
                        result_str += outline.get_prompt(idx)
        return result_str

    # Detailed version
    def get_prompt_detail(self, idx):
        if self.is_root:
            result_str = f"Premise: {self.premise}\n\nOutline:\n\n"
            for outline in self.son_outlines:
                if type(outline) == OutlineItem:
                    result_str += str(outline)
                else:
                    result_str += outline.get_prompt_detail(idx)
        else:
            result_str = str(self.outline_item)
            if self.outline_item.idx == idx[: len(self.outline_item.idx)]:
                for outline in self.son_outlines:
                    if type(outline) == OutlineItem:
                        result_str += str(outline)
                    else:
                        result_str += outline.get_prompt_detail(idx)
        return result_str

    def get_dict_plain(self, max_depth=None):
        if max_depth == 0:
            return None
        if max_depth is not None:
            max_depth -= 1
        if self.is_root:
            result_dict = {"premise": self.premise}
            if max_depth != 0:
                outline_tree = {}
                for idx in range(len(self.son_outlines)):
                    outline = self.son_outlines[idx]
                    if type(outline) == OutlineItem:
                        outline_tree[outline.idx] = outline.get_dict()
                    else:
                        outline_tree[outline.outline_item.idx] = outline.get_dict_plain(
                            max_depth=max_depth
                        )
                result_dict["outline"] = outline_tree
        else:
            result_dict = {"outline_item": self.outline_item.get_dict()}
            if max_depth != 0:
                outline_tree = {}
                for idx in range(len(self.son_outlines)):
                    outline = self.son_outlines[idx]
                    if type(outline) == OutlineItem:
                        if (
                            self.parrallel_son_outlines is not None
                            and len(self.parrallel_son_outlines) > idx
                        ):
                            parrallel_outline = self.parrallel_son_outlines[
                                idx
                            ].get_dict()
                        else:
                            parrallel_outline = None
                        outline_tree[outline.idx] = outline.get_dict()
                    else:
                        outline_tree[outline.outline_item.idx] = outline.get_dict_plain(
                            max_depth=max_depth
                        )
                result_dict["outline"] = outline_tree
        return result_dict

    def is_leaf(self):
        return self.son_outlines == None


def generate_outline(premise, model):
    prompt = [
        f"""Premise: {premise}

Can you break down the premise into some independent,  same-scaled outline? Also, assign each character a name. Please use the following template with "Main Plot" and "Characters". Do not answer anything else.

Event 1
Main plot: [TODO]
Characters: [TODO]

Event 2
Main plot: [TODO]
Characters: [TODO]

...
"""
    ]
    # If you want to control the node numbers, you can consider changing 'some' into specific numbers. And also consider adding event entry in the template.
    outputs = model[0](prompt, num_completions=1)[0]
    events = [event for event in outputs.split("Event ")[1:]]
    outline = Outline(events, premise=premise)

    if outline.son_outlines is None:
        logger.warning("Root Outline is None. Failed.")
        return None
    logger.info(f"***Finished Root Outline. Expand into {len(events)} items.***")
    return outline


Outline_Ranker = Ranker()

# This function use detailed prompt to expand the outline, and the parallel outline is generated also using the detailed prompt.


def similarity_check(cand_son, parent):
    success = True
    if cand_son == "":
        return False
    for p in parent.son_outlines:
        if type(p) == Outline:
            sim = cosine.similarity_profiles(
                cosine.get_profile(cand_son),
                cosine.get_profile(p.outline_item.main_plot),
            )
            if sim >= 0.90:
                success = False
                logger.warning(
                    f"       ! {round(sim,2)} similar to {p.outline_item.idx}: {p.outline_item.main_plot}"
                )
                break
        elif type(p) == OutlineItem:
            sim = cosine.similarity_profiles(
                cosine.get_profile(cand_son), cosine.get_profile(p.main_plot)
            )
            if sim >= 0.90:
                success = False
                logger.warning(
                    f"       ! {round(sim,2)} similar to {p.idx}: {p.main_plot}"
                )
                break
    return success


# This function use detailed prompt to expand the outline, and the parallel outline is generated also use detailed prompt
def expand_outline_detail(root_outline, leave_outline_items, model, expand_t):
    if expand_t <= 0:
        return 0
    if leave_outline_items == []:
        logger.error("!!!No leaves left. Generation Finished.!!!")
        return -1

    # rerank outline.
    candidates = [outl.main_plot for outl in leave_outline_items]

    cand_rank_id, rank_logits = Outline_Ranker.rank_idx_conpletely_wlogits(
        candidates, logger
    )

    to_expand_id = cand_rank_id[-1]
    to_expand = leave_outline_items[to_expand_id]
    threshold = rank_logits[to_expand_id]
    step_down = min(0.001 * expand_t, (threshold - 0.5) / 2)
    # Here you can custom and slightly change the Concreteness Scheduler. The default scheduler might be not suitable as the OpenAI model update.

    # Expand
    retry = 3  # outline loop retry means change total idea.
    neg_outputs = None
    outline_item = to_expand
    parent_outline = outline_item.parent
    idx = outline_item.idx  # type string, 1 greater than the number of to_expand_id
    logger.info(
        f"***Expand {idx}: {outline_item.main_plot} & Threshold {threshold} & Step Down {step_down}"
    )

    while retry:
        logger.info(f"  =Outline Loop {3-retry} for {idx}")
        prompt_outline = f"""{root_outline.get_prompt_detail(idx)}"""
        prompt = f"""
Can you break down point {idx} into some independent, chronological and same-scaled outline points? Also, assign each character a name. Please use the following template with "Main Plot" and "Characters". Do not answer anything else.

Point {idx}.1
Main plot: [TODO]
Characters: [TODO]

Point {idx}.2
Main plot: [TODO]
Characters: [TODO]

...
"""
        # Should be align with the inital-prompting setting
        outputs = model[0]([prompt_outline + prompt], num_completions=1)[0]
        events = [event for event in outputs.split("Point ")[1:]]
        outer_suboutline_idx = [idx + "." + str(i) for i in range(1, len(events) + 1)]
        retry -= 1
        # sometimes output repeats the origin outline_item in the outline sons list
        for son in events:
            temp_idx = son.split("\n")[0]
            if outline_item.str() in son:
                events.remove(son)
            if temp_idx not in outer_suboutline_idx:
                events.remove(son)
        # should expand to more than 1 outline item
        if len(events) <= 1:
            outputs, events = None, None
            logger.info("  !Fail: Only One Invid Child.")
            continue
        # construct new outline type
        new_outline = Outline(
            events,
            parrallel_son_outlines=events,
            outline_item=outline_item,
            parrallel_outline_item=outline_item,
        )
        if new_outline.son_outlines == None:
            logger.info("  !Fail: New Outline Has No Child Or Doesn't Satisfy Format.")
            outputs, events, new_outline = None, None, None
            continue
        outer_outputs, outer_new_outline = outputs, new_outline

        old_outline_str = outline_item.main_plot

        # innear loop, polish vague subpoints.
        sub_retry = 3
        if (
            parent_outline.is_root
        ):  # the root expansion with a inital threshold is recommended
            threshold += 0.03
            sub_retry = 6
        successed = False

        while sub_retry:
            logger.info(f"    =Inner Loop {3-sub_retry} for {idx}")
            neg_idx = []
            less_concrete_subpoint_num = 0
            if new_outline == None:
                new_outline = outer_new_outline
                outputs = outer_outputs
            sim_check == False  # if the outline suffers the repeating problem, it is recommended to utilize similarity checking. But note that it will increase the failure and retry time.
            if sim_check:
                for son in new_outline.son_outlines:
                    sim_check = similarity_check(son.main_plot, parent_outline)
                    if not sim_check:
                        neg_idx.append(son.idx)
                        less_concrete_subpoint_num += 1
            suboutline_idx = [s.idx for s in new_outline.son_outlines]
            for son in new_outline.son_outlines:
                eve = son.main_plot
                logit = Outline_Ranker.compare_w_neighbors(t=eve, cand=candidates)
                if logit > max(0.5025, threshold - step_down):
                    logger.info(
                        f'    - VS "{eve}": {round(threshold, 4)} vs {round(logit, 4)}'
                    )
                    neg_idx.append(son.idx)
                    less_concrete_subpoint_num += 1
            if not less_concrete_subpoint_num:
                successed = True
                break
            elif less_concrete_subpoint_num >= 0.8 * len(
                suboutline_idx
            ):  # 0.8, a softer setting for rejection
                sub_retry = 0
                continue
            else:  # sub-expand, using insert to rewrite
                logger.info("    - Exist Vague Child " + " ".join(neg_idx))
                temp_outline = copy.deepcopy(new_outline)
                for temp_son in temp_outline.son_outlines:
                    if temp_son.idx in neg_idx:
                        temp_son.main_plot = "[INSERT]"
                        temp_son.characters = "[INSERT]"
                prompt_outline = f"""{root_outline.get_prompt_detail(idx)}"""
                prompt = f"""
Can you break down point {idx} into some independent, chronological and same-scaled outline points? Also, assign each character a name. Please use the following template with "Main Plot" and "Characters". Do not answer anything else.

"""
                # Should be align with the inital-prompting setting
                prompt += (
                    "Output: "
                    + temp_outline.get_prompt_detail(idx)
                    + f"""
Task: Fill in the \"[INSERT]\" in the Outline. Do not change any other parts except \"[INSERT]\". 
"""
                )
                outputs = model[1]([prompt_outline + prompt], num_completions=1)[0]
                events = [event for event in outputs.split("Point ")[1:]]
                sub_retry -= 1
                # fill in [insert]
                # check if insert all the neg_idx items
                inserted_idx = [son.split("\n")[0] for son in events]
                if not all(elem in inserted_idx for elem in neg_idx):
                    outputs, events = outer_outputs, None
                    logger.info(
                        "    !Sub-Fail: Not all the blank is inserted or wrongly inserted."
                    )
                    continue
                for son in events:
                    son_item = OutlineItem(son, new_outline)
                    if son_item.idx not in neg_idx:
                        continue
                    sonlist_idx = int(son_item.idx.split(".")[-1]) - 1
                    new_outline.son_outlines[sonlist_idx] = son_item

        if not successed:
            neg_outputs = outputs
            outputs, events, new_outline = None, None, None
            continue
        else:
            logger.info("    :)Success Inner Loop")
            break

    if outputs == None:
        logger.info(f":( FAIL Expand {idx}; Remain {expand_t} Times")
        logger.info(f"***Remain Leaves: {[l.idx for l in leave_outline_items]}")
        leave_outline_items.remove(to_expand)

        ret = expand_outline_detail(root_outline, leave_outline_items, model, expand_t)
        return ret

    for cid, child in enumerate(parent_outline.son_outlines):
        if child == outline_item:
            parent_outline.son_outlines[cid] = new_outline

    # new outline node's parent is outline node
    new_outline.parent = parent_outline
    # update leaves list
    leave_outline_items.remove(new_outline.outline_item)
    leave_outline_items.extend(new_outline.son_outlines)
    logger.info(f":) SUCCESS Expand {idx}; Remain {expand_t-1} Times")
    logger.info(f"***Remain Leaves: {[l.idx for l in leave_outline_items]}")
    res = expand_outline_detail(root_outline, leave_outline_items, model, expand_t - 1)
    return res


def checkpoint(premise, name, expand_t, func, depth=None, bandwidth=None):
    global logger
    OUTPUT_DIR = "data/path/to"
    log_file = f"{OUTPUT_DIR}/{name}_stnv.log"

    logger = logging.getLogger(f"Expand")
    logger.setLevel(logging.DEBUG)
    formatter1 = logging.Formatter("%(levelname)s||%(message)s - %(asctime)s")
    formatter2 = logging.Formatter("%(levelname)s||%(message)s")
    logger.handlers = []
    steam_handler = logging.StreamHandler()
    steam_handler.setFormatter(formatter1)
    logger.addHandler(steam_handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter2)
    logger.addHandler(file_handler)
    logger.info(f"Premise: {premise}")
    model = [load_model(temp=0.5, logger=logger), load_model(temp=1.0, logger=logger)]

    outline = None
    while outline is None:
        outline = generate_outline(premise, model)  # expand the root

    leaves = outline.son_outlines.copy()
    res = func(outline, leaves, model, expand_t=expand_t)

    with open(f"{OUTPUT_DIR}/{name}_stnv.txt", "w") as f:
        f.write(str(outline))
    with jsonlines.open(f"{OUTPUT_DIR}/{name}_stnv_plain.jsonl", "w") as f:
        f.write(outline.get_dict_plain())
    import pickle

    with open(f"{OUTPUT_DIR}/{name}.pickle", "wb") as file:
        pickle.dump(outline, file)
    file.close()
    return outline


def example_outline_generator(premises, expand_t=None):
    for i in trange(len(premises)):
        premise = premises[i]
        print(f"Premise {i+1}: {premise}")

        filename = f"ex_detailed_outline_{i+1}_e{expand_t}"
        outline = checkpoint(
            premise=premise,
            name=filename,
            expand_t=expand_t,
            func=expand_outline_detail,
        )
        print("Finish Expansion")


def load_premise(n):
    PREMISE_FILE = "data/WritingPrompts/train.wp_source_processed"  # or you can use custom datasets
    with open(PREMISE_FILE, "r") as file:
        lines = file.readlines()
    if n > 0:
        premises = random.sample(lines, k=n)
        return premises
    else:
        return lines


if __name__ == "__main__":
    premises = load_premise(n=-1)[-1000:]

    example_outline_generator(
        premises, expand_t=25
    )  # 25 for long outlines, 13 for short outlines in the setting in the paper.
