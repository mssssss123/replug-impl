import argparse
import json
import os

import numpy as np
from scipy.special import softmax
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_utils import load_file, process_input_data, postprocess_output, test_kilt_em, match, read_csv_to_list, chunk_list
from collections import defaultdict
import operator


CHOICE_TASK =['arc','hellaswag','socialiqa','piqa',]
KILT_TASK = ['fever','aida','t-rex','eli5','hotpotqa','wow','nq','marco','tqa','musique','wiki']

def call_model(args, prompts, score_batch, user_chat_template, model, tokenizer, max_new_tokens=100):
    if user_chat_template:
        chat_prompts = []
        for prompt in prompts:
            if args.llama_style:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            else:
                prompt = "<用户>{}<AI>".format(prompt)
            chat_prompts.append(prompt)
        prompts = chat_prompts

    all_outputs = []
    all_probs = []
    for iindex,prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_length = input_ids.shape[-1]
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                return_dict_in_generate=True, 
                output_scores=True, 
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                output_logits=True,
                do_sample=False,
            )
        token_ids = output.sequences[0]
       
        log_probs = []
        for logits in output.scores:  
            log_probs.append(torch.nn.functional.log_softmax(logits, dim=-1))   
        log_probs = torch.cat(log_probs, dim=0) 
        
        generated_token_ids = token_ids[input_length:]  
        token_log_probs = []
         
        for j, token_id in enumerate(generated_token_ids):
            token_log_prob = log_probs[j, token_id].item()  
            token_log_probs.append(token_log_prob)
        # ppl的倒数
        perplexity = np.exp(np.mean(token_log_probs))
        output_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)    
        all_outputs.append(output_text)
        all_probs.append(perplexity*score_batch[iindex])
    
    ans2prob_list = defaultdict(list)
    for ans, prob in zip(all_outputs, all_probs):
        ans2prob_list[ans].append(prob)
    ans2prob = {k: sum(v) for k, v in ans2prob_list.items()}
    # bp()
    final_ans = max(ans2prob.items(), key=operator.itemgetter(1))[0]
    pred = postprocess_output(final_ans.lstrip())

    return pred
  


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default='/data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16')
    parser.add_argument('--input_file', type=str, default='/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl')
    parser.add_argument('--passage_score_file', type=str, default='/data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/trex_dev.trec')
    
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=32)
   

    parser.add_argument('--metric', type=str, default='accuracy')
    parser.add_argument('--top_n', type=int, default=5,help="number of paragraphs to be considered.")
    parser.add_argument('--task', type=str, default='t-rex')


    parser.add_argument('--user_chat_template', action='store_true')
    parser.add_argument('--llama_style', action='store_true')
    parser.add_argument('--rerank', action='store_true')

    parser.add_argument('--output_path', type=str,
                        default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--case_num', type=int, default=-1)
    args = parser.parse_args()

    print("解析后的参数为：")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    if args.output_path!=None:
        output_path = os.path.join(args.output_path, args.exp_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path=None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left",
                                            truncation_side="right", )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='cuda', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()


    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]

    input_data = process_input_data(input_data, args, args.top_n, tokenizer)

    trec_list = read_csv_to_list(args.passage_score_file)
    sub_trec=chunk_list(trec_list,100)
    if args.case_num != -1:
        sub_trec = sub_trec[:args.case_num]

    print("-----------case---------------")
    print(input_data[0])
    print("-----------case---------------")

    final_results = []
    
    for idx, example in tqdm(enumerate(input_data)):
        instruction_list = example['instruction']
        passage_id_list = example['passage_id_list']
        cur_example_trec = sub_trec[idx]
        cur_psg_score_dict = {}
        for tt in cur_example_trec:
            doc_id = str(tt[2])
            score = float(tt[4])
            cur_psg_score_dict[doc_id] = score
        score_batch = []
        for did in passage_id_list:
            docscore = cur_psg_score_dict[did]
            score_batch.append(docscore)
        score_batch = softmax(np.array(score_batch)).tolist()
        pred = call_model(
            args, instruction_list, score_batch, user_chat_template=args.user_chat_template, model=model, tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens)
        example["output"] = pred
        final_results.append(example)
        


    if output_path is not None:
        output_path = os.path.join(output_path, str(args.task)+'output.jsonl')
        with open(output_path, "w") as f:
            for item in input_data:
                json.dump(item, f)
                f.write("\n")
    print("结果文件已生成：", output_path)

    for item in input_data:
        if args.task in KILT_TASK:
            metric_result = test_kilt_em(args.task,args.metric,item["output"], item)
        else:
            if args.metric == "accuracy":
                if args.task in CHOICE_TASK:
                    metric_result = item["output"].strip().startswith(item["golds"][0])
                else:
                    metric_result = 0.0
                    for pa in item["golds"]:
                        metric_result = 1.0 if pa in item["output"] or pa.lower() in item["output"] or pa.capitalize() in item["output"] else 0.0
            elif args.metric == "match":
                metric_result = match(item["output"], item["golds"])

            else:
                raise NotImplementedError
        item["metric_result"] = metric_result

    print(args.task)
    print("overall result: {0}".format(
        np.mean([item["metric_result"] for item in input_data])))
    print('finish')

if __name__ == "__main__":
    main()