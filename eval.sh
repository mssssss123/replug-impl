
export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/trex_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --top_n 5  \
    --rerank \
    --user_chat_template  > minicpm_replug_trex.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/trex_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --top_n 5  \
    --rerank \
    --llama_style \
    --user_chat_template  > llama_replug_trex.out  2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/wow_dev.trec \
    --max_new_tokens 32  \
    --metric f1  \
    --task wow  \
    --top_n 5  \
    --rerank \
    --user_chat_template  > minicpm_replug_wow_new.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/wow_dev.trec \
    --max_new_tokens 32  \
    --metric f1  \
    --task wow  \
    --top_n 5  \
    --rerank \
    --llama_style \
    --user_chat_template  > llama_replug_wow_new.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/tqa_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task tqa  \
    --top_n 5  \
    --rerank \
    --user_chat_template  > minicpm_replug_tqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/tqa_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task tqa  \
    --top_n 5  \
    --rerank \
    --llama_style \
    --user_chat_template  > llama_replug_tqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/nq_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --top_n 5  \
    --rerank \
    --user_chat_template  > minicpm_replug_nq.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/nq_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --top_n 5  \
    --rerank \
    --llama_style \
    --user_chat_template  > llama_replug_nq.out  2>&1 &





export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/marco_qa_dev.trec \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --top_n 5  \
    --rerank \
    --case_num 3000  \
    --user_chat_template  > minicpm_replug_marco.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/marco_qa_dev.trec \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --top_n 5  \
    --rerank \
    --llama_style \
    --case_num 3000  \
    --user_chat_template  > llama_replug_marco.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/hotpotqa_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --top_n 5  \
    --rerank \
    --user_chat_template  > minicpm_replug_hotpotqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl \
    --passage_score_file /data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/hotpotqa_dev.trec \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --top_n 5  \
    --rerank \
    --llama_style \
    --user_chat_template  > llama_replug_hotpotqa.out  2>&1 &