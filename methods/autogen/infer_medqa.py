from __future__ import annotations

from dataset_utils.medqa import extract_medqa_choice
from methods.maslab_runtime_config import build_general_config

from .autogen_main import AutoGen_Main


def autogen_infer_medqa(question: str, root_path: str, model_info: str = "gpt-4o-mini") -> str:
    general_config = build_general_config(root_path, model_info)
    mas = AutoGen_Main(general_config)

    result = mas.inference({"query": question})
    response = (result or {}).get("response", "")
    print(f"autogen_response:{response}")
    token_stats = mas.get_token_stats()
    # choice = extract_medqa_choice(response)
    return str(response).strip(),token_stats
