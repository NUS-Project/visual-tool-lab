import os
from openai import OpenAI


class JudgeModel:
    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    def chat(self, query, response, right_label,right_context):
        # 发起聊天请求
        response = self.client.chat.completions.create(
            model= self.model_id,
            messages=[
                {
                    "role": "system",
                    "content":"You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, the correct answer content, and the corresponding correct answer label. Your tasks are as follows:\n1.Determine if the output sentence is complete. If it is incomplete, respond with [incomplete].\n2.If the output sentence is complete, evaluate whether it accurately answers the question based on the provided correct answer content and correct answer label. Respond with either [Correct] or [Incorrect].\nREMEMBER: Your judgment result must be only [incomplete], [Correct], or [Incorrect]."
                },
                {"role": "user", "content": format_prompt(query, response, right_label,right_context)},
            ],
            stream=False
        )

        # 返回助手的响应
        return response.choices[0].message.content



def format_prompt(query, response, right_label,right_context):
    prompt = f'''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, the correct answer content, and the corresponding correct answer label. Your tasks are as follows:\n1.Determine if the output sentence is complete. If it is incomplete, respond with [incomplete].\n2.If the output sentence is complete, evaluate whether it accurately answers the question based on the provided correct answer content and correct answer label. Respond with either [Correct] or [Incorrect].\nREMEMBER: Your judgment result must be only [incomplete], [Correct], or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: {str(query)}

Output sentence: {str(response)}

Correct answer label: {str(right_label)}

Correct answer context: {str(right_context)}

Judgement Result:
'''
    return prompt

# def llm_eval(query, response, ground_truth):
#     try:
#         prompt = format_prompt(query, response, ground_truth)
#         response = llm.inference({"query": prompt})['response']
#         valid_label = ['correct', 'incorrect']
#
#         if isinstance(response, str):
#             item_label = response.strip().lower()
#             if item_label in valid_label:
#                 if item_label == "correct":
#                     return item_label, 1
#                 else:
#                     return item_label, 0
#             else:
#                 return f"Eval Error: {item_label}", None
#         else:
#             return "Eval Error: response is not a string", None
#     except Exception as e:
#         return f"Eval Error: {str(e)}", None