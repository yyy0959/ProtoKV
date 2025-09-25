import os
import json
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "llama-3.1": 7950,
    "mistral": 31500
}

# 标记和颜色配置
markers = ['o', 's', '^', 'D', 'v']
colors = {
    'blue': '#3A7CA5',      # 深海蓝
    'teal': '#2F7A8D',      # 海绿色
    'gray': '#7F8B8C',      # 石灰色
    'orange': '#D97C29',    # 秋叶橙
    'green': '#4A8C45',     # 森林绿
    'light_blue': '#88C1E0' # 天空蓝
}
fonttitle = 10
fontsize = 8
fontlabel = 10
tick_fontsize = 8

layer_top5_beta_results = {}

beta_values = [1.2 + i * 0.2 for i in range(10)]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

def compute_similarity_analysis(tensor, beta):
    groups = tensor.view(20, 50, 128)  

    selected_tokens = []  
    unselected_tokens = []  

    for group in groups:
        X = group  
        X_norm = X.norm(dim=1, keepdim=True) + 1e-8 
        cos_sim_matrix = torch.matmul(X, X.T) / (X_norm * X_norm.T)  
        distance_matrix = 1 - cos_sim_matrix 

        D1 = distance_matrix.sum(dim=1) / (distance_matrix.shape[1] - 1) 
        D2 = D1.mean() 
       
        mask = D1 > beta * D2 
        selected_tokens.append(group[mask])
        unselected_tokens.append(group[~mask])

    if not selected_tokens:
        return None
    C = torch.cat(selected_tokens, dim=0)
    if C.shape[0] == 0:
        return None 
    
    cluster_center = C.mean(dim=0)

    if not unselected_tokens:
        return None
    unselected_all = torch.cat(unselected_tokens, dim=0)
    if unselected_all.shape[0] == 0:
        return None 

    cc_norm = cluster_center.norm()
    ua_norms = unselected_all.norm(dim=1)
    dot_products = torch.matmul(unselected_all, cluster_center)
    cos_sim = dot_products / (ua_norms * cc_norm + 1e-8)
    distances = 1 - cos_sim
    final_result = distances.mean().item()

    return final_result

def get_top_beta(past_key_values):
    for layer in range(len(past_key_values)): 
        head_results = {}
        for head in range(past_key_values[layer][0].size(1)):
            results = []
            for beta in beta_values:
                key = past_key_values[layer][0]  
                tensor = key[0, head, :1000, :]  
                result = compute_similarity_analysis(tensor, beta)
                results.append(result)
            head_results[head] = [r for r in results if r is not None]
    
        avg_results = {}
        for head, res_list in head_results.items():
            if len(res_list) > 0:
                avg_results[head] = sum(res_list) / len(res_list)
            else:
                avg_results[head] = float('nan')

        sorted_heads = sorted(avg_results.items(), key=lambda x: x[1] if not isinstance(x[1], float) or not (x[1] != x[1]) else float('inf'))
        valid_heads = [h for h, r in sorted_heads if not isinstance(r, float) or not (r != r)] 
        top5_heads = valid_heads[:5]

        layer_data = []
        for head in top5_heads:
            beta_result = []
            for beta in beta_values:
                key = past_key_values[layer][0]
                tensor = key[0, head, :1000, :]
                result = compute_similarity_analysis(tensor, beta)
                beta_result.append(result)
            layer_data.append(beta_result)
        
        layer_top5_beta_results[layer] = (top5_heads, layer_data)

def plot_beta(filepath):

    fig, axs = plt.subplots(4, 8, figsize=(18, 9), dpi=300)  
    fig.subplots_adjust(wspace=0.3, hspace=0.5)


    axs_flat = axs.ravel()
    layers = list(range(32))

    handles_list = []

    for i, ax in zip(layers, axs_flat):
        top5_heads, beta_data = layer_top5_beta_results.get(i, ([], [])) 
        
        if top5_heads and beta_data:
            handles = []
            for head_idx, head in enumerate(top5_heads):
                data = beta_data[head_idx]
                filtered_beta = [b for b, r in zip(beta_values, data) if r is not None]
                filtered_result = [r for r in data if r is not None]
                
                color = list(colors.values())[head_idx % len(colors)]
                line = ax.plot(filtered_beta, filtered_result,
                            marker=markers[head_idx % len(markers)],
                            markersize=5,
                            linestyle='-',
                            linewidth=1.5,
                            color=color,
                            markeredgewidth=1.0)
                handles.append(line[0]) 
                
            ax.set_title(f"Layer {i}", fontsize=fonttitle, pad=6, fontweight='bold')
            ax.set_ylabel("Clustering Degree", fontsize=fontlabel-2)
            ax.set_xlabel(r"$\beta$", fontsize=fontlabel-2)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize-1)
            
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            handles_list.extend(handles)
        else:
            ax.text(0.5, 0.5, 'No\ndata', ha='center', va='center', fontsize=fontsize-1, alpha=0.6)
            ax.axis('off')

    fig.legend(
        handles=handles_list,
        loc='lower center',               
        bbox_to_anchor=(0.5, 0.96),       
        ncol=min(5, len(handles_list)),  
        fontsize=fontsize,
        frameon=True,
        edgecolor='black'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])   

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print("complete")
    print(filepath)

def main(args):
    
    test_data = []
    
    prompts = []
    
    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
                
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
    
    
    for example in test_data:
        
        prompts.append(example["prompt"])

    print("Finish loading model and tokenizer")
        
    model_name = model_path.split("/")[-1]

    save_dir = os.path.join(args.save_dir, model_name, args.dataset)
    os.makedirs(save_dir, exist_ok=True)  

    for i in tqdm(range(0, len(prompts))):
        
        batch_prompts = prompts[i:i+1]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask
        
        
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=True
        )

        past_key_values = outputs.past_key_values  # tuple: ( (k,v), (k,v), ... ) 

        get_top_beta(past_key_values)

        filename = f"beta{i}.pdf"
        filepath = os.path.join(save_dir, filename)
        plot_beta(filepath)
        
        torch.cuda.empty_cache()
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)

    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )    

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    save_dir = args.save_dir

    for idx, dataset in enumerate(datasets):
        
        print(f"Working on dataset {dataset} - {idx}/{len(datasets)}")
        
        args.dataset = dataset
        
        args.data_file = f"data/LongBench/{args.dataset}.jsonl"
        
        main(args)
