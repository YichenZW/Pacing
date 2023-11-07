for ((id = 800; id <= 1000; id++)); do
    input_dir="concoct_outputs/concoct_short/ex_detailed_outline_${id}_e13_stnv.doc.pkl"
    final_story_dir="output/short/concoct_${id}_story.pkl.final.txt"
    output_pkl="output/short/concoct_${id}_story.pkl"
    output_log="output/short/concoct_${id}_story.log"
    if [[ -e "$input_dir" && ! -e "$final_story_dir" ]]; then
        echo "$id"
        CUDA_VISIBLE_DEVICES=1 python -u scripts/main.py --extension-method gpt3 --controller longformer_classifier longformer_classifier fudge_controller --loader alignment coherence fine_coherence --controller-load-dir doc_data/ckpt/relevance_reranker doc_data/ckpt/coherence_reranker doc_data/ckpt/detailed_controller --controller-model-string allenai/longformer-base-4096 allenai/longformer-base-4096 facebook/opt-350m --load-outline-file "$input_dir" --no-editor --include-future-context --control-strength 1 1 0 --control-strength-substep-increment 3 --save-complete-file "$output_pkl" --log-file "$output_log"
    else
        echo "$input_dir does not exist."
    fi
done