import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
from openai import OpenAI
from pptree import *
import transformers
import torch
from typing import Any, Dict, List
_MODEL_CACHE = {}


import yaml

def load_config(file_path: str) -> dict:
    """
    Load YAML configuration from a file.

    Args:
    file_path (str): Path to the YAML configuration file.

    Returns:
    dict: Dictionary of configuration parameters.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def setup_model(model_name):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    model_path=os.getenv("GENERAL_MODEL_PATH")
    if model_name == 'Llama-3.3-70B-Instruct':
        pipeline = transformers.pipeline(
            "text-generation",
            model=os.path.join(model_path, model_name),
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        _MODEL_CACHE[model_name] = pipeline
        return pipeline
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def parse_recruitment_json(recruited_text: str) -> List[Dict[str, Any]]:
    start_index = recruited_text.find('{')
    end_index = recruited_text.rfind('}')
    # 切割出有效的 JSON 字符串
    recruited_text = recruited_text[start_index:end_index + 1]
    data = json.loads(recruited_text)

    groups_out = []
    for _, g in data.items():  # 第1个组=Group1，第2个组=Group2...
        g_items = list(g.items())
        group_goal = str(g_items[0][1]).strip() if g_items else ""

        members: List[Dict[str, str]] = []
        for _, mv in g_items[1:]:  # Member1/2/3... 依次
            # 允许 mv 是 list[dict] 或 dict
            if isinstance(mv, list) and mv and isinstance(mv[0], dict):
                mv = mv[0]
            if not isinstance(mv, dict):
                continue

            mv_items = list(mv.items())
            role = str(mv_items[0][1]).strip() if len(mv_items) >= 1 else ""
            desc = str(mv_items[1][1]).strip() if len(mv_items) >= 2 else ""

            if role:
                members.append({"role": role, "expertise_description": desc})

        groups_out.append({"group_goal": group_goal, "members": members})

    return groups_out

class Agent:
    def __init__(self, instruction, role,  model_info='gemini-2.5-flash', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
         
        if self.model_info == 'gemini-2.5-flash':
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            # self.model = genai.GenerativeModel('gemini-2.5-flash')
            # self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client =  OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            # if examplers is not None:
            #     for exampler in examplers:
            #         self.messages.append({"role": "user", "content": exampler['question']})
            #         self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
                    
        elif self.model_info =='Llama-3.3-70B-Instruct':
            self.pipeline = setup_model(model_info)
            self.messages = [
                {"role": "system", "content": instruction},
            ]

    def chat(self, message, img_path=None, chat_mode=True):
        print("Message:",message)
        if self.model_info == 'gemini-2.5-flash':
            self.messages.append({"role": "user", "content": message})
            response = self.client.chat.completions.create(
                model=self.model_info,
                messages=self.messages,
                stream=False
            )
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content
            # for _ in range(10):
            #     try:
            #         response = self._chat.send_message(message, stream=True)
            #         responses = ""
            #         for chunk in response:
            #             responses += chunk.text + "\n"
            #         return responses
            #     except:
            #         continue
            # return "Error: Failed to get response from Gemini."
        
        elif self.model_info=='Llama-3.3-70B-Instruct':
            self.messages.append({"role": "user", "content": message})
            response = self.pipeline(
                self.messages,
                max_new_tokens=2560
            )
            return response[0]["generated_text"][-1]["content"]

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini','gemini-2.5-flash']:
            self.messages.append({"role": "user", "content": message})
            
            # temperatures = [0.0,0.5,1.0]
            
            # responses = {}
            # for temperature in temperatures:
            if self.model_info == 'gpt-3.5':
                model_info = 'gpt-3.5-turbo'
            else:
                model_info = 'gpt-4o-mini'
            response = self.client.chat.completions.create(
                model=model_info,
                messages=self.messages,
                temperature=0.5,
            )

            responses = response.choices[0].message.content
                
            return responses
        
        elif self.model_info == 'Llama-3.3-70B-Instruct':
            self.messages.append({"role": "user", "content": message})
            response = self.pipeline(
                self.messages,
                max_new_tokens=10240
            )
            return response[0]["generated_text"][-1]["content"]
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses

class Group:
    def __init__(self, goal, members, question,model_info):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model_info)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        # self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which names {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where names {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            # if self.examplers is not None:
            #     investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            # else:
            investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)

            return response

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

# def parse_group_info(group_info):
#     lines = group_info.split('\n')
    
#     parsed_info = {
#         'group_goal': '',
#         'members': []
#     }

#     parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
#     for line in lines[1:]:
#         if line.startswith('Member'):
#             member_info = line.split(':')
#             member_role_description = member_info[1].split('-')
            
#             member_role = member_role_description[0].strip()
#             member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
#             parsed_info['members'].append({
#                 'role': member_role,
#                 'expertise_description': member_expertise
#             })
    
#     return parsed_info


def mdagents_test(question,root_path):
    config_path = f'{root_path}/methods/MDAgents/configs/config_main.yaml'
    config = load_config(config_path)
    difficulty = config.get('difficulty', 'adaptive')
    num_teams = config.get('num_teams', 3)  # Default to 3 if not specified
    num_agents = config.get('num_agents', 3)  # Default to 3 if not specified
    intermediate_num_agents= config.get('intermediate_num_agents', 5)
    model_info = config.get('model_info', 'Llama-3.3-70B-Instruct')  # Default to 3 if not specified
    difficulty = determine_difficulty(question, difficulty,model_info)
    prompt_file=f"{root_path}/methods/MDAgents/Recruit_prompt.txt"
    print(f"difficulty: {difficulty}")

    if difficulty == 'basic':
        final_decision = process_basic_query(question,  model_info)
    elif difficulty == 'intermediate':
        final_decision = process_intermediate_query(question,  model_info,intermediate_num_agents)
    else:
        # difficulty = 'advanced':
        final_decision = process_advanced_query(question,prompt_file,model_info,num_teams,num_agents)
    return final_decision


def determine_difficulty(question, difficulty,model_info):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    # 现在，根据以下医疗问题，请您判断其难度/复杂程度：
    # {问题}。

    # 请从以下选项中选择该医疗问题的难度/复杂程度：
    # 1）基础：单个医疗人员即可给出答案。
    # 2）中级：需要不同专业背景的医疗专家共同讨论并做出最终决定。
    # 3）高级：来自不同科室的多个医疗团队需要相互协作以做出最终决定。
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model_info)
    # system:您是一位医疗专家，负责进行初步评估，您的任务是确定医疗问题的难易程度/复杂性。
    # medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'
    else:
        return 'intermediate'

def process_basic_query(question, model_info):
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model_info)
    # new_examplers = []
    # if args.dataset_name == 'medqa':
    #     random.shuffle(examplers)
    #     for ie, exampler in enumerate(examplers[:5]):
    #         tmp_exampler = {}
    #         exampler_question = exampler['question']
    #         choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
    #         random.shuffle(choices)
    #         exampler_question += " " + ' '.join(choices)
    #         exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
    #         exampler_reason = medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

    #         tmp_exampler['question'] = exampler_question
    #         tmp_exampler['reason'] = exampler_reason
    #         tmp_exampler['answer'] = exampler_answer
    #         new_examplers.append(tmp_exampler)
    
    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.Provide only the letter corresponding to your answer choice (A/B/C/D/E/F).', role='medical expert',  model_info=model_info)
    # single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    # final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=None)
    final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''')
    return final_decision

def process_intermediate_query(question, model_info,intermediate_num_agents):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    # 您是一位经验丰富的医学专家，招募了一群身份各异的专家，让他们讨论并解决给定的医学问题
    
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_info)
    tmp_agent.chat(recruit_prompt)
    
    num_agents = intermediate_num_agents # You can adjust this number as needed
    recruited = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")
    # 问题：{问题}\n您可以招募具有不同医学专长的{专家数量}名专家。
    # 考虑到医学问题以及答案的选项，您会招募哪种类型的专家来更准确地给出答案呢？
    # 此外，您需要明确专家之间的沟通结构（例如，肺科医生 == 新生儿科医生 == 医学遗传学家 == 儿科医生 > 心脏科医生），或者说明他们是否是独立的。
    # \n\n例如，如果您想招募五名专家，您的回答可以是：
    # \n1.儿科医生 - 专注于对婴儿、儿童和青少年的医疗护理。 - 层级：独立
    # \n2.心脏科医生 - 专注于心脏和血管相关疾病的诊断和治疗。 - 层级：儿科医生 > 心脏科医生
    # \n3.肺科医生 - 专注于诊断和治疗呼吸系统疾病。- 层级：独立
    # \n4.新生儿科医生 - 专注于对早产儿或出生时有医疗问题的新生儿的护理。 - 层级：独立
    # \n5.医学遗传学家 - 专注于基因和遗传的研究。- 层级：独立型
    # \n\n请按照上述格式作答，且无需说明原因。
    print(f"recruited:内容{recruited}")
    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]
    print(f"agents_info:内容{agents_info}")
    print(f"agents_data:内容{agents_data}")

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
        description = agent[0].split('-')[1].strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model_info)
        
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    # fewshot_examplers = ""

    # if args.dataset_name == 'medqa':
    # random.shuffle(examplers)
    # for ie, exampler in enumerate(examplers):
    #     exampler_question = f"[Example {ie+1}]\n" + exampler['question']
    #     options = [f"({k}) {v}" for k, v in exampler['options'].items()]
    #     random.shuffle(options)
    #     exampler_question += " " + " ".join(options)
    #     medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert',
    #                           model_info=model_info)
    #     exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
    #     exampler_reason = medical_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")
    #     # 以下是一道医学知识题及其答案示例。在查看了下面的医学问题并给出答案后，请您提供 1 - 2 句支持该答案的解释，说明您之前并不知道答案。
    #     exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
    #     fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 5
    num_turns = 5
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    initial_report = ""
    # print(f"fewshot_examplers:内容{fewshot_examplers}")
    for k, v in agent_dict.items():
        # opinion = v.chat(f'''Given the examplers, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like above format.\n\nAnswer: ''', img_path=None)
        opinion = v.chat(
            f'''Given the medical query below, please indicate the option you believe is correct and provide your reasoning.\nQuestion: {question}Your response should follow this format:\n1. Your chosen option.\n2. Your reasoning.''',
            img_path=None)
        print(f"opinion内容:{opinion}")
        initial_report += f"({k.lower()}): {opinion}\n"
        print(f"initial_report内容:{initial_report}")
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model_info)
        # 你是一名医疗助理，擅长根据来自不同领域的多位专家的观点进行总结和综合分析。
        agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        # 你是一名医疗助理，擅长根据来自不同领域的多位专家的观点进行总结和综合分析。
        
        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())
        print(f"assessment内容:{assessment}")

        report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
        # 您需要完成以下步骤：
        # 1.
        # 对以下报告进行仔细和全面的考量。
        # 2.
        # 从以下报告中提取关键知识。
        # 3.
        # 根据这些知识得出全面且总结性的分析。
        # 4.
        # 您的最终目标是基于以下报告得出一份精炼且综合的报告。
        #
        # 您应以以下格式输出：关键知识：；总分析：
        print(f"num_turns:{num_turns}")
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                # 鉴于您团队中其他医学专家的意见，请告知您是否希望与任何专家进行交流（是 / 否）
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    # 输入您想与哪位专家交谈的号码：
                    # 专家1：骨科医生 - 专注于肌肉骨骼疾病的外科治疗。
                    # 专家2：医学伦理学家 - 专注于医疗实践中的伦理问题，包括患者披露和知情同意。
                    # 专家3：风险管理专家 - 在医疗环境中致力于降低风险，并提供法律和合规意见。
                    # 专家4：患者倡导者 - 代表患者的利益和权利，确保医疗的透明度和沟通。
                    # 专家5：医疗法律专家 - 专注于影响医疗实践的法律和法规，包括医疗事故和知情同意
                    # 例如，如果您想与专家1交谈，请只返回1。如果您想与多个专家交谈，请返回1,2，并且不要返回理由。
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        # 请先展示您的专业知识，然后再将您的意见提交给您选定的专家。
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model_info)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')

    print(f"final_answer内容：{final_answer}")

    # _decision = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}", img_path=None)
    _decision = moderator.temp_responses(
        f"Question:{question}\nEach agent's final answer{final_answer}\nGiven each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. You should Provide only the letter corresponding to your answer choice (A/B/C/D/E/F).",
        img_path=None)

    # 在给出每个代理的最终答案后，请审查每个代理的意见，并通过多数票得出对该问题的最终答案。您的答案应采用以下格式：
    # 答案： C）第2个咽弓
    # {最终答案}
    # 问题： {问题}
    #Provide only the letter corresponding to your answer choice (A/B/C/D/E/F).
    final_decision = {'majority': _decision}

    print("\U0001F468\u200D\u2696\uFE0F moderator's final decision (by majority vote):", _decision)
    print()

    return _decision

def process_advanced_query(question,  prompt_file,model_info,num_teams,num_agents):
    print("[STEP 1] Recruitment")
    group_instances = []

    # recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""
    #您是一位经验丰富的医学专家。面对如此复杂的医疗问题，您需要组建多学科团队（MDT），并让团队成员共同给出准确且可靠的解答。
    prompts_file = load_prompts_from_file(prompt_file)
    prompts_recruit = prompts_file["MEDICAL_ASSISTANT"]
    print("MEDICAL_ASSISTANT",prompts_recruit)
    print("MEDICAL_ASSISTANT",prompts_recruit)
    recruit_example = prompts_file["MEDICAL_RECRUIT"]
    tmp_agent = Agent(instruction=prompts_recruit, role='system', model_info=model_info)
    # tmp_agent.chat(prompts_recruit)

    # num_teams = 3  # You can adjust this number as needed
    # num_agents = 3  # You can adjust this number as needed
    # recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.")
    recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example:{recruit_example} When you return your answer, please strictly refer to the above format.")
    #问题：{问题}\n\n您应当组建 {团队数量} 个具有不同专业特长或不同目的的医疗多学科团队（MDT），并且每个 MDT 都应配备 {医生数量} 名临床医生。考虑到医疗问题以及所提供的选项，请返回您的招募计划，以便更准确地给出答案。
    # \n\n例如，以下可以是一个示例答案：
    # \n小组 1 - 初步评估团队（IAT）
        # \n成员 1：耳鼻喉科医生（耳鼻喉外科医生）（负责人） - 专长于耳、鼻和喉手术，包括甲状腺切除术。由于其在手术干预和处理任何手术并发症（如神经损伤）方面起着关键作用，该成员担任组长。
        # \n成员 2：普通外科医生 - 提供额外的外科专业知识，并在甲状腺手术并发症的总体管理中提供支持。
        # \n成员 3：麻醉师 - 专注于围手术期护理、疼痛管理以及评估任何因麻醉可能影响声音和气道功能的并发症。
    # \n\n小组 2 - 诊断证据团队（DET）
        # \n成员 1：内分泌科医生（负责人） - 负责格雷夫斯病的长期管理，包括激素治疗和术后任何相关并发症的监测。
        # \“成员 2：言语语言病理学家——专长于嗓音和吞咽障碍的治疗，为神经受损后的患者提供康复服务，以改善其言语和嗓音质量。
        # \n成员 3：神经科医生——评估并提供有关神经损伤及潜在恢复策略的建议，为患者的治疗提供神经学方面的专业知识。
    # \n\n小组 3 - 患者病史团队（PHT）
        # \n成员1：精神科医生或心理学家（负责人）——处理慢性疾病及其治疗所带来的心理影响，包括与嗓音变化、自尊心以及应对策略相关的问题。
        # \n成员 2：物理治疗师——提供锻炼和策略，以维持身体健康，并通过整体健康状况间接支持嗓音功能的恢复。
        # \n成员 3：职业治疗师——帮助患者适应嗓音的变化，特别是如果他们的职业高度依赖于声音交流，帮助他们找到维持职业角色的策略。
    # 第 4 组 - 最终评审与决策团队（FRDT）
        # 成员 1：各专业领域的高级顾问（负责人） - 提供整体的专业知识和决策指导
        # 成员 2：临床决策专家 - 协调来自不同团队的各种建议，并制定综合治疗方案
        # 成员 3：高级诊断支持人员 - 利用先进的诊断工具和技术来确认神经损伤的确切程度和原因，从而辅助做出最终决策
        
    # 以上只是一个示例，因此您应自行组建独特的多学科团队（MDT），但您的招聘计划中应包含初始评估团队（IAT）和最终评审与决策团队（FRDT）。在提交答案时，请严格遵循上述格式。
    print("\n[DEBUG] ===== recruiter raw output =====")
    print(recruited)
    print("[DEBUG] ===== end recruiter raw output =====\n")
    # NEW: parse recruiter JSON
    try:
        groups_parsed = parse_recruitment_json(recruited)
    except json.JSONDecodeError as e:
        raise ValueError(f"Recruiter did not return valid JSON: {e}\nRaw:\n{recruited}")

    print(f"[DEBUG] parsed {len(groups_parsed)} groups from JSON")
    for g in groups_parsed:
        group_instances.append(Group(g["group_goal"], g["members"], question,model_info=model_info))
        
    # for i, g in enumerate(groups_parsed):
    #     print(f"Group {i+1} - {g['group_goal']}")
    #     for j, m in enumerate(g["members"]):
    #         print(f" Member {j+1} ({m['role']}): {m['expertise_description']}")

        # IMPORTANT: Group expects members as list of {role, expertise_description}
        # group_instances.append(Group(g["group_goal"], g["members"], question))
    # groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    # group_strings = ["Group " + group for group in groups]
     
    # print(f"[DEBUG] split into {len(group_strings)} group blocks")
    # print(f"group_strings_context: {group_strings}")
    # for i1, gs in enumerate(group_strings):
    #     print(f"\n[DEBUG] ----- Group block #{i1} -----")
    #     print(gs)
    #     print(f"[DEBUG] ----- end block #{i1} -----\n")
    #     res_gs = parse_group_info(gs)
    #     print("[DEBUG] parsed group_goal:", repr(res_gs["group_goal"]))
    #     print("[DEBUG] parsed members:", res_gs["members"])
    #     print(f"Group {i1+1} - {res_gs['group_goal']}")
    #     for i2, member in enumerate(res_gs['members']):
    #         print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
    #     print()

    #     group_instance = Group(res_gs['group_goal'], res_gs['members'], question)
    #     group_instances.append(group_instance)

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    initial_assessments = []
    # for group_instance in group_instances:
    #     if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
    #         init_assessment = group_instance.interact(comm_type='internal')
    #         initial_assessments.append([group_instance.goal, init_assessment])
    if group_instances:  # 只取第一个
        init_assessment = group_instances[0].interact(comm_type='internal')
        initial_assessments.append([group_instances[0].goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # STEP 2.2. other MDTs Process
    assessments = []
    if len(group_instances) > 2:
        for group_instance in group_instances[1:-1]:
            # if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal')
            assessments.append([group_instance.goal, assessment])
    # for group_instance in group_instances:
    #     if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
    #         assessment = group_instance.interact(comm_type='internal')
    #         assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # STEP 2.3. FRDT Process
    final_decisions = []
    # for group_instance in group_instances:
    #     if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
    #         decision = group_instance.interact(comm_type='internal')
    #         final_decisions.append([group_instance.goal, decision])
    if len(group_instances) > 1:
            group_instance = group_instances[-1]
            decision = group_instance.interact(comm_type='internal')
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # STEP 3. Final Decision
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query.Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."""
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model_info)
    tmp_agent.chat(decision_prompt)

    final_decision = tmp_agent.temp_responses(f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}""", img_path=None)

    return final_decision


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    """
    Load multiple prompts from a file.

    Args:
    file_path (str): Path to the file containing prompts.

    Returns:
    Dict[str, str]: A dictionary of prompt names and their content.

    Raises:
    FileNotFoundError: If the specified file is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts = {}
    current_prompt = None
    current_content = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_prompt:
                    prompts[current_prompt] = "\n".join(current_content).strip()
                current_prompt = line[1:-1]
                current_content = []
            elif line:
                current_content.append(line)

    if current_prompt:
        prompts[current_prompt] = "\n".join(current_content).strip()

    return prompts
