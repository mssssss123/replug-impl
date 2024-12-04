import json
import argparse
import csv
from tqdm import tqdm
import pandas as pd
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
            
def read_csv_to_list(filepath, delimiter=' '):
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        data_list = [row for row in reader]  
    return data_list
def chunk_list(original_list, chunk_size):
    return [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec", type=str, default="/home/lixz23/ragsft/data/marco_v2.1/bge_large_retriever_128_256_top100/wow_dev.trec")
    parser.add_argument("--input", type=str, default="/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl")
    parser.add_argument("--output_path", type=str, default="/data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/wow_dev.trec")
    parser.add_argument("--topk", type=int, default=100)
    
    parser.add_argument("--selectk", type=int, default=100)
    args = parser.parse_args()

    trec_list = read_csv_to_list(args.trec)
    sub_trec=chunk_list(trec_list,args.topk)
    
    ref_data = read_jsonl(args.input)
    for did,data in enumerate(ref_data):
        passages = data['passage']
        cur_trec = sub_trec[did]
        for pid,psg in enumerate(passages):
            docid = psg['docid']
            cur_cur_trec = cur_trec[pid]
            cur_cur_trec[2] = docid
            cur_cur_trec[5] = 'neu'
    combine_trec = [item for sublist in sub_trec for item in sublist]
    
    
    # 将数据写入TREC文件
    with open(args.output_path, 'w') as f:
        for entry in combine_trec:
            f.write(" ".join(entry) + "\n")

    print(f"TREC文件已保存到: {args.output_path}")

if __name__ == "__main__":
    main()