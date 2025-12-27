import os
import json
import random
import argparse
from pathlib import Path
import contextlib
import io
from datetime import datetime
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from methods.MDAgents.medagents import mdagents_test
from methods.autogen import autogen_infer_medqa
from methods.dylan import dylan_infer_medqa
from methods.general_model import Llama_test,Qwen_test,QwenVL_Test
from dataset_utils import load_test_split, format_question, extract_choice

PROJECT_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--general_model_path', type=str, default=str(PROJECT_ROOT / 'models'))
parser.add_argument('--dataset_path', type=str, default=str(PROJECT_ROOT / 'data'))
parser.add_argument('--dataset_name', type=str, default='medqa')
parser.add_argument(
    '--model',
    type=str,
    default='medagents',
    choices=['Qwen2.5-VL-72B-Instruct', 'Qwen2.5-72B-Instruct', 'medagents', 'autogen', 'dylan', 'Llama-3.3-70B-Instruct'],
)
parser.add_argument('--root_path', type=str, default=str(PROJECT_ROOT))
parser.add_argument("--device", type=str, default="auto", help='Device / device_map, e.g. "cuda:3" or "auto"')
parser.add_argument('--num_samples', type=int, default=2)
args = parser.parse_args()

args.root_path = str(Path(args.root_path).expanduser().resolve())
args.general_model_path = str(Path(args.general_model_path).expanduser().resolve())
args.dataset_path = str(Path(args.dataset_path).expanduser().resolve())

os.environ["GENERAL_MODEL_PATH"] = args.general_model_path
# MODEL_API_CONFIG_PATH = os.environ["MODEL_API_CONFIG_PATH"]
model_paths = str(Path(args.general_model_path) / args.model)
dataset_path_name = str(Path(args.dataset_path) / args.dataset_name)
test_qa = load_test_split(dataset_path_name, args.dataset_name)

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
correct_count = 0

num_total = len(test_qa)
num_to_run = min(args.num_samples, num_total) if args.num_samples else num_total

processed_count = 0
for sample in tqdm(test_qa[:num_to_run], total=num_to_run, desc=f"Infer {args.model}", unit="sample"):
    processed_count += 1

    question = format_question(sample, args.dataset_name)
    if "Llama" in args.model:
        final_decision = llama_model.chat(question)
    elif "Qwen" in args.model and "VL" in args.model:
        final_decision = QwenVL_model.chat(question)
    elif "Qwen" in args.model:
        final_decision = Qwen_model.chat(question)
    elif args.model=="medagents":
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            final_decision = mdagents_test(question, args.root_path)
    elif args.model == "autogen":
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            final_decision = autogen_infer_medqa(question, args.root_path)
    elif args.model == "dylan":
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            final_decision = dylan_infer_medqa(question, args.root_path)
        
    correct_answer_idx = sample['answer_idx']
    correct_answer_content = sample['options'][correct_answer_idx]

    pred = str(final_decision).strip()
    extracted = extract_choice(pred, args.dataset_name)
    if extracted:
        pred = extracted

    if pred == correct_answer_idx or pred == correct_answer_content:
        correct_count += 1
        is_correct = True
    else:
        is_correct = False

    results.append({
        'question': question,
        'answer': sample.get('answer'),
        'label': correct_answer_idx,
        'label_text': correct_answer_content,
        'model_response': pred,
        'is_correct': is_correct,
    })
    

# Calculate accuracy
accuracy = (correct_count / processed_count) * 100 if processed_count > 0 else 0

# Save results
path = Path(args.root_path) / 'output'
path.mkdir(parents=True, exist_ok=True)

# Build output filename: method + base model + dataset + timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_model_name = None
if args.model == "medagents":
    # For MDAgents, base model is declared in its own config.
    try:
        import yaml

        cfg_path = Path(args.root_path) / "methods" / "MDAgents" / "configs" / "config_main.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            base_model_name = (yaml.safe_load(f) or {}).get("model_info")
    except Exception:
        base_model_name = None
elif args.model in {"autogen", "dylan"}:
    # Current wrappers default to gpt-4o-mini unless you change the wrapper call.
    base_model_name = "gpt-4o-mini"
else:
    base_model_name = args.model

base_model_name = (base_model_name or "unknown").replace(" ", "_")

output_file = path / f"{args.model}_{base_model_name}_{args.dataset_name}_{timestamp}.json"
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(results, file, indent=4)

# Print accuracy
print(f"\n[INFO] Accuracy: {accuracy:.2f}%")
