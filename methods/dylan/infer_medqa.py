from __future__ import annotations

from dataset_utils.medqa import extract_medqa_choice
from methods.maslab_runtime_config import build_general_config
from .dylan_main import DyLAN_Main


def dylan_infer_medqa(question: str, root_path: str, model_info: str = "gpt-4o-mini") -> str:
    general_config = build_general_config(root_path, model_info)
    mas = DyLAN_Main(general_config)

    result = mas.inference({"query": question})
    # print(f"dylan_result:{result}")
    response = (result or {}).get("response", "")
    print(f"dylan_response:{response}")
    token_stats = mas.get_token_stats()
    # choice = extract_medqa_choice(response)
    return str(response).strip(),token_stats
