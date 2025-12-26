import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from methods.MDAgents.medagents import mdagents_test
from methods.general_model import Llama_test,Qwen_test,QwenVL_Test

def load_data(dataset):
    test_qa = []
    # examplers = []

    test_path = f'{dataset}/test.jsonl'
    with open(test_path, 'r', encoding='utf-8') as file:
        for line in file:
            test_qa.append(json.loads(line))

    # train_path = f'{dataset}/train.jsonl'
    # with open(train_path, encoding='utf-8') as file:
    #     for count, line in enumerate(file):
    #         if count == 5:  # 只读取前5行
    #             break
    #         examplers.append(json.loads(line))

    return test_qa

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        # random.shuffle(options)
        question += " ".join(options)
        return question
    return sample['question']

parser = argparse.ArgumentParser()
parser.add_argument('--general_model_path', type=str, default=r'E:\Code')
parser.add_argument('--dataset_path', type=str, default=r'E:\Code\MedData')
parser.add_argument('--dataset_name', type=str, default='medqa')
parser.add_argument('--model', type=str, default='medagents',choices=['Qwen2.5-VL-72B-Instruct','Qwen2.5-72B-Instruct','medagents','Llama-3.3-70B-Instruct'])
parser.add_argument('--root_path', type=str, default=r'E:\Code\MedToolLab')
parser.add_argument("--device", type=str, default="auto", help='Device / device_map, e.g. "cuda:3" or "auto"')
parser.add_argument('--num_samples', type=int, default=5)
args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = "sk-i5zo6MXMbPdCaPK6gP8Px1ZJQbKZysEAqVmUnWPLJkGFeFpJ"
os.environ["BASE_URL"] = 'https://yinli.one/v1' #"https://hiapi.online/v1"
os.environ["GENERAL_MODEL_PATH"] = args.general_model_path
# MODEL_API_CONFIG_PATH = os.environ["MODEL_API_CONFIG_PATH"]
model_paths = os.path.join(args.general_model_path, args.model)
dataset_path_name = os.path.join(args.dataset_path, args.dataset_name)
test_qa = load_data(dataset_path_name)

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)


if "Llama" in args.model:
    llama_model = Llama_test(model_paths)
    
if "Qwen" in args.model:
    if "VL" in args.model:
        QwenVL_model = QwenVL_Test(model_paths,args.device)
    else:
        Qwen_model = Qwen_test(model_paths,args.device)
    
    
results = []
correct_count=0
for no, sample in enumerate(tqdm(test_qa)):
    if no == args.num_samples:
        break
    print(f"\n[INFO] no: {no}")

    question = create_question(sample, args.dataset_name)
    if "Llama" in args.model:
        final_decision = llama_model.chat(question)
    elif "Qwen" in args.model and "VL" in args.model:
        final_decision = QwenVL_model.chat(question)
    elif "Qwen" in args.model:
        final_decision = Qwen_model.chat(question)
    elif args.model=="medagents":
        final_decision = mdagents_test(question, args.root_path)
        
    correct_answer_idx = sample['answer_idx']
    correct_answer_content = sample['options'][correct_answer_idx]
    print(final_decision)
    print(final_decision)
    if final_decision.strip() == correct_answer_idx or final_decision.strip() == correct_answer_content:
        correct_count += 1
    else:
        print(f"{args.model} answers wrong:{final_decision}")
        print(f"The right answer_idx:{correct_answer_idx}")
        print(f"The right answer_content:{correct_answer_content}")
        print("\n" + "="*80)
        # if args.dataset_name == 'medqa':
        results.append({
            'question': question,
            'answer': sample['answer'],
            'label': sample['answer_idx'],
            'model_response': final_decision,
        })
    

# Calculate accuracy
accuracy = (correct_count / len(test_qa)) * 100 if args.num_samples > 0 else 0

# Save results
path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(path):
    os.makedirs(path)

with open(f'output/{args.model}_{args.dataset_name}.json', 'w') as file:
    json.dump(results, file, indent=4)

# Print accuracy
print(f"\n[INFO] Accuracy: {accuracy:.2f}%")
