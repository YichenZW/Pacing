# Improving Pacing in Long-Form Story Planning

This repository includes the code implementation of the paper  <u>Improving Pacing in Long-Form Story Planning</u> by *Yichen Wang, Kevin Yang, Xiaoming Liu,* and *Dan Klein*. Corresponding datasets are in the [huggingface hub]([ZachW/GPT-BookSum · Datasets at Hugging Face](https://huggingface.co/datasets/ZachW/GPT-BookSum)).

### Installation

(1) Install Python 3.8.13 (Recommended). Other similar versions should also work fine.

(2) Clone the repo and move into it.

```shell
git clone https://github.com/YichenZW/Pacing
cd Pacing
```

(3) Install the required packages. You are recommended to do it in a Conda environment.

```shell
pip install -r requirements.txt
```

(4) Install the repo by running

```shell
pip install -e .
```

(5) Set your OpenAI API key (or sign up and generate one on the [OpenAI](https://openai.com/) website) 

```shell
echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
```

or simply add the following lines at the beginning of  `ChatGPT_API.py` and remove all the `openai.api_key = os.environ['OPENAI_API_KEY']` lines.

```python
import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
```

(6) Download the GPT_BookSum dataset from the [huggingface hub]([ZachW/GPT-BookSum · Datasets at Hugging Face](https://huggingface.co/datasets/ZachW/GPT-BookSum)). Save it into `data/`.

(7) Train and save the concreteness evaluator.

```shell
python concrete_evaluator/train.py
```

Note: For fair and quicker evaluation, it is recommended to run the `save_for_val()` and `save_for_test()` in the `dataset.py` first.

```shell
python dataset.py
```

(8) (Optional) Download the [WritingPrompts](https://paperswithcode.com/dataset/writingprompts) dataset if you want. Save it in the `data/WritingPrompts/`. You can also build the prompt dataset or input individual prompts by yourself.

(9) Run the outline generator.

```shell
python run_outline.py
```

### Evaluation

`gpt4_eval/gpt4_eval_single.py` is for evaluating the quality of outlines and story excerpts by gpt-4. For the detailed prompt settings for each attribute, please follow the appendix. `concrete_evaluator/val.py` is for evaluating the performance of the concreteness evaluator. For other human evaluation settings and interfaces, please refer to the appendix of the paper.

### Downstream Story Generation

We mainly follow the DOC generation framework to generate complete stories. The full description and instruction for DOC is at [this page]([GitHub - yangkevin2/doc-story-generation](https://github.com/yangkevin2/doc-story-generation).  Also, a copy at `story_generation/README.md`. We also utilize a simplified version of DOC to generate a fixed-length story passage for each outline item rather than dynamically varying the passage length as the vanilla DOC. Furthermore, we modify DOC to use ChatGPT rather than their original OPT-175B.

Install the environment first via `pip install -r story_generation/requirements.txt`. Then install this repo with `pip install -e story_generation/.`.

a) using ChatGPT version DOC:

```shell
cd story_generation
CUDA_VISIBLE_DEVICES=0 python -u scripts/main.py {{{ALPA_ARGS}}} --controller longformer_classifier longformer_classifier fudge_controller --loader alignment coherence fine_coherence --controller-load-dir doc_data/ckpt/relevance_reranker doc_data/ckpt/coherence_reranker doc_data/ckpt/detailed_controller --controller-model-string allenai/longformer-base-4096 allenai/longformer-base-4096 facebook/opt-350m --load-outline-file output/plan.pkl --no-editor --include-future-context --control-strength 1 1 0 --control-strength-substep-increment 3 --save-complete-file output/story.pkl --log-file output/story.log
```

setting `--draft-model-string` to `gpt-3.5-turbo` if you what to use ChatGPT instead of original OPT-175B. `ALPA_ARGS` is only needed if you want to use the free public Alpa OPT-175B API.

For other detailed descriptions, please refer to [DOC repo]([GitHub - yangkevin2/doc-story-generation](https://github.com/yangkevin2/doc-story-generation), Section Main Story Generation Command.

If you want to generate a batch of stories, you can refer to the scripts `story_generation/scripts/gen_baseline.sh` and `story_generation/scripts/gen_concoct.sh`.

b) using the simplified version DOC with ChatGPT:

```shell
python story_generation/direct_gen.py
```



### Citations

[TODO]
