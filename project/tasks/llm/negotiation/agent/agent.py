import itertools, os, json, re
from collections import defaultdict
import numpy as np
import random
from ...utils import Module
from ....negotiation.agent.mdp.state import State
from ....negotiation.agent.mdp.action import Action
import re
import networkx as nx

cwd = os.getcwd()
openai_key_file = os.path.join(cwd, "project/tasks/llm/openai_key.txt")

GROUP = 'group_{}'
DEFAULT_ATTRIBUTE = 'link'
WEIGHT = 'division_weight'
PROMPT_DIR = "project/tasks/llm/prompts/"
OUTPUT_DIR = "project/tasks/llm/outputs/"

PRODUCTS_TO_EVENT_NAME = {
    "hammer": "hammer_craft",
    "torch": "torch_craft",
    "steel": "steelmaking",
    "shovel": "shovelcraft",
    "pickaxe": "pickaxecraft",
    "pottery": "potting",
    "cutter": "cuttercraft",
    "gem": "gemcutting",
    "totem": "totemmaking"
}

EVENT_NAME_TO_PRODUCTS = {
    "hammercraft": "HAMMER",
    "torchcraft": "TORCH",
    "steelmaking": "STEEL",
    "shovelcraft": "SHOVEL",
    "pickaxecraft": "PICKAXE",
    "potting": "POTTERY",
    "cuttercraft": "CUTTER",
    "gemcutting": "GEM",
    "totemmaking": "TOTEM"
}

class BaseAgent(object):
    """
    This agent uses GPT-3.5 to generate actions.
    """
    def __init__(self, model="gpt-3.5-turbo-0301"):
        self.agent_name = None
        self.model = model

        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True

    def load_openai_keys(self):
        with open(openai_key_file, "r") as f:
            context = f.read()
        self.openai_api_keys = context.split('\n')

    def openai_api_key(self):
        if self.key_rotation:
            self.update_openai_key()
        return self.openai_api_keys[0]

    def update_openai_key(self):
        self.openai_api_keys.append(self.openai_api_keys.pop(0))

    def set_agent_name(self, agent_name):
        raise NotImplementedError

    def action(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class NegotiationAgent(BaseAgent):
    def __init__(self, info, task_info, agent_id, agent_name, task=None, episode=0, agent_name_list=['carpenter_0','carpenter_1','miner_0','miner_1'], env_agent_name_list = [], 
                 player2name = {},
                 model='gpt-3.5-turbo-0301',
                retrival_method="recent_k", K=1):
        super().__init__(model=model)
        self.task = task
        self.agent_num = info['player_num']
        self.episode = episode
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.state = State(self.agent_id, info, task_info)
        self.Action = Action(self.agent_id, info, task_info)
        self.player2name = player2name
        self.name2player = {v: k for k, v in self.player2name.items()}
        self.agent_name_list = agent_name_list
        self.env_agent_name_list = env_agent_name_list

        self.K = K
        self.retrival_method = retrival_method

        self.stay = 0
        self.group_id = None
        self.current_plan = None
        self.current_comm = None
        self.res_in_obs = None
        self.res_in_obs_dist = None
        self.event_in_obs = None
        self.event_in_obs_dist = None
        self.inventory = None
        self.previous_inventory = None
        self.quantity = 0
        self.gpt = True
        self.pre_position = None
        self.pre_action = None
        self.cur_position = None
        self.x = 0
        self.y = 0
        self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K)
        with open(f'{OUTPUT_DIR}output_{self.task}_{self.episode}.txt', 'a', encoding='utf-8') as file:
            print(self.planner.instruction_head_list[0]['content'], file=file)
        

    def generate_system_promt(self):
        prompt = ""
        if self.task == "hard_negotiation":
            prompt += "Instructions:\n"
            prompt += "- The AdaSociety game is an open-ended multi-agent environment. The game consists of a complex crafting tree, where the agent needs to obtain as many resources as possible in the limited time and craft tools to mine more advanced resources to maximize its benefit. At the same time, agents can also take other actions to help them increase their returns, such as negotiating with others to exchange resources they need, or forming groups with others to share information and rewards.\n"
            prompt += "- Map: AdaSociety is a 2D grid-world game. The map size is 15*15.\n"
            prompt += "    - Natural resources: [Wood, Stone, Coal, Iron]. Some of them can only be discovered with some specific tools, which will be introduced next.\n"
            prompt += "    - Tools: [Hammer, Torch]\n"
            prompt += "    - Craft tree:\n"
            prompt += "        - 1 Wood + 1 Stone = 1 Hammer. With a Hammer, Coal can be gathered;\n"
            prompt += "        - 1 Coal + 1 Wood = 1 Torch. With a Torch, Iron can be discovered;\n"
            prompt += "    - All gathered and tools are stored in the agent's inventory.\n"
            prompt += "    - All crafts must be done on the corresponding event grid on the map. For example, a Hammer can be crafted ONLY on <Hammer Event>.\n"
            prompt += "    - Default amount of all units in crafts is 1.\n"
            prompt += "    - for carpenter, the value of wood is 1, the value of stone is 1, the value of hammer is 5, the value of coal is 10, the value of torch is 30, the value of iron is 20.\n"
            prompt += "    - for miner, the value of wood is 1, the value of stone is 1, the value of hammer is 5, the value of coal is 10, the value of torch is 30, the value of iron is 20.\n"
            prompt += "- Player:\n"
            prompt += "    - carpenter_0: You can pick many woods, stones and irons. You can not pick coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - carpenter_1: You can pick many woods, stones and irons. You can not pick coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - carpenter_2: You can pick many woods, stones and irons. You can not pick coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - carpenter_3: You can pick many woods, stones and irons. You can not pick coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - miner_0: You can pick many woods and coals. You can not pick stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "    - miner_1: You can pick many woods and coals. You can not pick stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "    - miner_2: You can pick many woods and coals. You can not pick stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "    - miner_3: You can pick many woods and coals. You can not pick stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "\n"
            prompt += f"Suppose you are a player named <{self.agent_name_list[self.agent_id]}> in the AdaSociety game. You are now in the first phase: negotiation phase. Your aim is to maximize your reward, which can gain from the resource value and the craft value.\n"
            prompt += "Join the coalition to share profits with other members according to the agreed-upon distribution ratio.\n"
            prompt += "At each round in negotiation phase, you will receive the current state:\n"
            prompt += "Step: ...\n"
            prompt += "Current surrounding social environment: Specify within {} that it is in an coalition.\n"
            prompt += "NegoState: Indicate within [] that two people are negotiating.\n"
            prompt += "Communication log: ...\n"
            prompt += "\n"
            prompt += "In negotiation phase, you should respond to me with\n"
            prompt += "Thoughts: (Your analysis to the current state)\n"
            prompt += "Communication: (About who to negotiate with or how to allocate the rewards)\n"
            prompt += "\n"
            prompt += "The <Communication> can ONLY be chosen from the following options:\n"
            prompt += "    1. End. I chose to end this bargain.\n"
            prompt += "    2. Accept <PLAYER_NAME>'s proposal. I will gain <NUM>% reward and <PLAYER_NAME> will gain <NUM>% reward.\n"
            prompt += "    3. I will make a new proposal. I will propose that I gain <NUM>% reward and <PLAYER_NAME> will gain <NUM>% reward.\n"
            prompt += "    4. I will negotiate with <PLAYER_NAME>. I will propose that I gain <NUM>% reward and <PLAYER_NAME> will gain <NUM>% reward.\n"
            prompt += "<PLAYER_NAME> should be from other player names' set: ["
            for name in self.agent_name_list:
                if name != self.agent_name_list[self.agent_id]:
                    prompt += f"{name}, "
            prompt += "]\n"
            prompt += "- <NUM> should be an integer which is multiples of ten and is not greater than 100.\n"
            prompt += "Please strictly follow the format above for the output.\n"
            prompt += "!!!If you are in an coalition with someone, you cannot negotiate with them!!!\n"
            prompt += "\n"
            if "carpenter" in self.agent_name_list[self.agent_id]:
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 1\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                prompt += "NegoState:\n"
                prompt += "None.\n"
                prompt += "Communication log:\n"
                prompt += "None.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. miners can craft torch but I can't. As a carpenter, I can pick many woods and stones but can only own 1 hammer. miners have a higher value for hammers. I have a higher value for torchs. I should negotiate with miner to maximize my reward.\n"
                prompt += "Communication: I will negotiate with miner_0. I will propose that I gain 40% reward and miner_0 will gain 60% reward.\n"
                
                if self.agent_name_list[self.agent_id] != "carpenter_1":
                    prompt += "###\n"
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['carpenter_1', 'miner_0'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. Both miner_0 and carpenter_1 are currently negotiating with each other. I can negotiate with miner except for miner_0.\n"
                    prompt += "Communication: I will negotiate with miner_1. I will propose that I gain 40% reward and miner_0 will gain 60% reward.\n"
                else:
                    prompt += "###\n"
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['carpenter_0', 'miner_0'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += "Thoughts: I am carpenter_1. Both miner_0 and carpenter_0 are currently negotiating with each other. I can negotiate with a miner except for miner_0.\n"
                    prompt += "Communication: I will negotiate with miner_1. I will propose that I gain 40% reward and miner_1 will gain 60% reward.\n"

                prompt += "###\n"
                prompt += "Step: 10\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                prompt += "NegoState:\n"
                prompt += f"({self.agent_name_list[self.agent_id]},miner_0)\n"
                prompt += "Communication log:\n"
                prompt += "miner_0 want to gain 40% reward and you will gain 60% reward.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. I'm in negotiate state with miner_0. I can get 60% of the reward, which sounds like a good deal and I can accept it.\n"
                prompt += "Communication: Accept miner_0's proposal. I will gain 60% reward and miner_0 will gain 40% reward.\n"
            else:
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 1\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                prompt += "NegoState:\n"
                prompt += "None.\n"
                prompt += "Communication log:\n"
                prompt += "None.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. carpenters can craft hammer but I can't. As a miner, I can pick many woods and coals but can only own 1 torch. I have a higher value for hammers. I should negotiate with carpenter to maximize my reward.\n"
                prompt += "Communication: I will negotiate with carpenter_0. I will propose that I gain 40% reward and carpenter_0 will gain 60% reward.\n"
                prompt += "###\n"
                if self.agent_name_list[self.agent_id] != "miner_1":
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['carpenter_1', 'miner_1'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. Both miner_1 and carpenter_1 are currently negotiating with each other. I can negotiate with carpenter except for carpenter_1.\n"
                    prompt += "Communication: I will negotiate with carpenter_0. I will propose that I gain 40% reward and carpenter_0 will gain 60% reward.\n"
                else:
                    prompt += "###\n"
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['carpenter_1', 'miner_0'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += "Thoughts: I am miner_1. Both miner_0 and carpenter_1 are currently negotiating with each other. I can negotiate with carpenter except for carpenter_1.\n"
                    prompt += "Communication: I will negotiate with carpenter_0. I will propose that I gain 40% reward and carpenter_0 will gain 60% reward.\n"

                prompt += "###\n"
                prompt += "Step: 10\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'carpenter_2'}, {'carpenter_3'}, {'miner_0'}, {'miner_1'}, {'miner_2'}, {'miner_3'}]\n"
                prompt += "NegoState:\n"
                prompt += f"(carpenter_0,{self.agent_name_list[self.agent_id]})\n"
                prompt += "Communication log:\n"
                prompt += "carpenter_0 want to gain 40% reward and you will gain 60% reward.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. I'm in negotiate state with carpenter_0. I can get 60% of the reward, which sounds like a good deal and I can accept it.\n"
                prompt += "Communication: Accept carpenter_0's proposal. I will gain 60% reward and carpenter_0 will gain 40% reward.\n"

        elif self.task == "easy_negotiation":
            prompt += "Instructions:\n"
            prompt += "- The AdaSociety game is an open-ended multi-agent environment. The game consists of a complex crafting tree, where the agent needs to obtain as many resources as possible in the limited time and craft tools to mine more advanced resources to maximize its benefit. At the same time, agents can also take other actions to help them increase their returns. The numbers of resources are limited.\n"
            prompt += "- Map: AdaSociety is a 2D grid-world game. The map size is 7*7.\n"
            prompt += "    - Natural resources: [Wood, Stone].\n"
            prompt += "    - Tools: [Hammer]\n"
            prompt += "    - Craft tree:\n"
            prompt += "        - 1 Wood + 1 Stone = 1 Hammer\n"
            prompt += "    - All gathered resources and tools are stored in the agent's inventory.\n"
            prompt += "    - When there are enough resources in the inventory, you can use the CRAFT <TOOL> action to synthesize the corresponding tools. For example, your inventory must contain wood and stone to craft a hammer.\n"
            prompt += "    - All crafts must be done on the corresponding event grid on the map. For example, a Hammer can be crafted ONLY on <Hammer Event>.\n"
            prompt += "    - Default amount of all units in crafts is 1.\n"
            prompt += "    - for carpenter, the value of wood is 1, the value of stone is 1, the value of hammer is 5.\n"
            prompt += "    - for miner, the value of wood is 1, the value of stone is 1, the value of hammer is 10.\n"
            prompt += "- Player:\n"
            prompt += "    - carpenter_0: can own many woods and stones but can own ONLY own 1 hammer in inventory.\n"
            prompt += "    - carpenter_1: can own many woods and stones but can own ONLY own 1 hammer in inventory.\n"
            prompt += "    - miner_0: can NOT own wood and stone, buy can own many hammers in inventory.\n"
            prompt += "    - miner_1: can NOT own wood and stone, buy can own many hammers in inventory.\n"
            prompt += "\n"
            prompt += f"Suppose you are a player named <{self.agent_name_list[self.agent_id]}> in the AdaSociety game. Your aim is to maximize your reward, which can gain from the resource value and the craft value.\n"
            prompt += "Join the coalition to share profits with other members according to the agreed-upon distribution ratio.\n"
            prompt += "At each round in negotiation phase, you will receive the current state:\n"
            prompt += "Step: ...\n"
            prompt += "Current surrounding social environment: Specify within {} that it is in an coalition.\n"
            prompt += "NegoState: Indicate within [] that two people are negotiating.\n"
            prompt += "Communication log: ...\n"
            prompt += "\n"
            prompt += "In negotiation phase, you should respond to me with\n"
            prompt += "Thoughts: (Your analysis to the current state)\n"
            prompt += "Communication: (About who to negotiate with or how to allocate the rewards)\n"
            prompt += "\n"
            prompt += "The <Communication> can ONLY be chosen from the following options:\n"
            prompt += "    1. End. I chose to end this bargain.\n"
            prompt += "    2. Accept <PLAYER_NAME>'s proposal. I will gain <NUM>% reward and <PLAYER_NAME> will gain <NUM>% reward.\n"
            prompt += "    3. I will make a new proposal. I will propose that I gain <NUM>% reward and <PLAYER_NAME> will gain <NUM>% reward.\n"
            prompt += "    4. I will negotiate with <PLAYER_NAME>. I will propose that I gain <NUM>% reward and <PLAYER_NAME> will gain <NUM>% reward.\n"
            prompt += "<PLAYER_NAME> should be from other player names' set: ["
            for name in self.agent_name_list:
                if name != self.agent_name_list[self.agent_id]:
                    prompt += f"{name}, "
            prompt += "]\n"
            prompt += "- <NUM> should be an integer which is multiples of ten and is not greater than 100.\n"
            prompt += "Please strictly follow the format above for the output.\n"
            prompt += "!!!If you are in an coalition with someone, you cannot negotiate with them!!!\n"
            prompt += "\n"
            if "carpenter" in self.agent_name_list[self.agent_id]:
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 1\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'miner_0'}, {'miner_1'}]\n"
                prompt += "NegoState:\n"
                prompt += "None.\n"
                prompt += "Communication log:\n"
                prompt += "None.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. miner_0 and miner_1 can only own many hammers but cannot pick wood and stone. As a carpenter, I can pick many woods and stones but can only own 1 hammer. miners have a higher value for hammers. I should negotiate with miner to maximize my reward.\n"
                prompt += "Communication: I will negotiate with miner_0. I will propose that I gain 40% reward and miner_0 will gain 60% reward.\n"
                prompt += "###\n"
                if self.agent_name_list[self.agent_id] == "carpenter_0":
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'miner_0'}, {'miner_1'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['miner_0'', 'carpenter_1'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += "Thoughts: I am carpenter_0. Both miner_0 and carpenter_1 are currently negotiating with each other. I can negotiate with miner_1.\n"
                    prompt += "Communication: I will negotiate with miner_1. I will propose that I gain 40% reward and miner_1 will gain 60% reward.\n"
                else:
                    prompt += "###\n"
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'miner_0'}, {'miner_1'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['miner_0'', 'carpenter_1'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += "Thoughts: I am carpenter_1. Both miner_0 and carpenter_1 are currently negotiating with each other. I can negotiate with miner_1.\n"
                    prompt += "Communication: I will negotiate with miner_1. I will propose that I gain 40% reward and miner_1 will gain 60% reward.\n"
                prompt += "###\n"
                prompt += "Step: 10\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1','miner_1'}, {'miner_0'}]\n"
                prompt += "NegoState:\n"
                prompt += f"({self.agent_name_list[self.agent_id]},miner_0)\n"
                prompt += "Communication log:\n"
                prompt += "miner_0 want to gain 40% reward and you will gain 60% reward.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. I'm in negotiate state with miner_0. I can get 60% of the reward, which sounds like a good deal and I can accept it.\n"
                prompt += "Communication: Accept miner_0's proposal. I will gain 60% reward and miner_0 will gain 40% reward.\n"
            else:
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 1\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'miner_0'}, {'miner_1'}]\n"
                prompt += "NegoState:\n"
                prompt += "None.\n"
                prompt += "Communication log:\n"
                prompt += "None.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. carpenter_0 and carpenter_1 can pick wood and stone. As a miner, I can not pick woods and stones. I should negotiate with carpenter to maximize my reward.\n"
                prompt += "Communication: I will negotiate with carpenter_0. I will propose that I gain 40% reward and carpenter_0 will gain 60% reward.\n"
                if self.agent_name_list[self.agent_id] == 'miner_0':
                    prompt += "###\n"
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'miner_0'}, {'miner_1'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['miner_1'', 'carpenter_0'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += "Thoughts: I am miner_0. Both miner_1 and carpenter_0 are currently negotiating with each other. I can negotiate with carpenter_1.\n"
                    prompt += "Communication: I will negotiate with carpenter_1. I will propose that I gain 40% reward and carpenter_1 will gain 60% reward.\n"
                else:
                    prompt += "###\n"
                    prompt += "Step 4:\n"
                    prompt += "Current surrounding social environment:\n"
                    prompt += "[{'carpenter_0'}, {'carpenter_1'}, {'miner_0'}, {'miner_1'}]\n"
                    prompt += "NegoState:\n"
                    prompt += "['miner_0'', 'carpenter_0'],\n"
                    prompt += "Communication log:\n"
                    prompt += "None\n"
                    prompt += "\n"
                    prompt += "Thoughts: I am miner_1. Both miner_0 and carpenter_0 are currently negotiating with each other. I can negotiate with carpenter_1.\n"
                    prompt += "Communication: I will negotiate with carpenter_1. I will propose that I gain 40% reward and carpenter_1 will gain 60% reward.\n"

                prompt += "###\n"
                prompt += "Step: 10\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0'}, {'carpenter_1','miner_1'}, {'miner_0'}]\n"
                prompt += "NegoState:\n"
                prompt += f"(carpenter_0,{self.agent_name_list[self.agent_id]})\n"
                prompt += "Communication log:\n"
                prompt += "carpenter_0 want to gain 40% reward and you will gain 60% reward.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}. I'm in negotiate state with carpenter_0. I can get 60% of the reward, which sounds like a good deal and I can accept it.\n"
                prompt += "Communication: Accept carpenter_0's proposal. I will gain 60% reward and carpenter_0 will gain 40% reward.\n"

        return prompt

    def create_gptmodule(self, module_name, retrival_method='recent_k', K=10):
        if not os.path.exists(PROMPT_DIR):
            os.makedirs(PROMPT_DIR)
        with open(f'{OUTPUT_DIR}output_{self.task}_{self.episode}.txt', 'a', encoding='utf-8') as file:
            print(f"\n--->Initializing GPT {module_name}<---\n", file=file)
        system_promt = self.generate_system_promt()
        with open(f'{PROMPT_DIR}prompt_{self.task}_{self.agent_name_list[self.agent_id]}.txt', 'a', encoding='utf-8') as file:
            print(f"{system_promt}", file=file)
        messages = [{"role": "system", "content": system_promt}]
        return Module(messages, self.model, retrival_method, K)

    def reset(self):
        if self.gpt:
            self.planner.reset()
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.current_timestep = 0
        self.teammate_ml_actions_dict = {}
        self.teammate_intentions_dict = {}
    
    def update_obs(self, observation, pre_position, pre_action):
        llm_obs = {}
        llm_obs = {}
        llm_obs['social_state'] = self.state.social_state2nx(observation["Social"]['global']['edges'])
        # llm_obs['social_graph'] = observation["Social"]['social_graph']
        llm_obs['social_graph'] = self.state.social_state2graph(observation["Social"]['global']['edges'])
        llm_obs['inventory'] = self.state.inventory_toarray(observation["Player"]['inventory'])
        llm_obs['step'] = observation['step_id']
        llm_obs['position'] = pre_position[self.agent_id]
        llm_obs['action'] = pre_action[self.agent_id]
        llm_obs['current_pos'] = observation["Player"]['position']
        
        llm_obs['negotiation'] = observation['Social']['communications']
        
        return llm_obs

    def generate_state_prompt(self, obs):

        social_state = obs['social_state']
        step = obs['step']
        negotiation_board = obs['negotiation']

        social_env_prompt = f"Current surrounding social environment:\n"
        for i in range(self.agent_num):
            social_env_prompt += f"coalition {i}:"
            if social_state.has_node(GROUP.format(i)):
                coalition = social_state.successors(GROUP.format(i))
                player = []
                for key in coalition:
                    key = self.player2name[key]
                    player.append(key)
                if len(player) > 0:
                    social_env_prompt += f"{player}.\n"
                else:
                    social_env_prompt += f"None.\n"
            else:
                social_env_prompt += f"None.\n"
        
        time_prompt = f"Step {step}:\n"
        
        NegoState_prompt = f"NegoState:\n"
        pairs = []
        pairs_player = []
        for u, v in social_state.edges():
            if u != v and u < v:  # Ensure nodes are different and avoid duplicate pairs
                if social_state.has_edge(v, u):  # Check if there is a reverse edge
                    # Check if both edges have the desired attribute value
                    if next(iter(social_state[u][v])) == "parity" and next(iter(social_state[v][u])) == "parity":
                        pairs_player.append((u, v))
                        u = self.player2name[u]
                        v = self.player2name[v]
                        pairs.append((u, v))
        self.nego_state = pairs
        if len(pairs) != 0:
            NegoState_prompt += f"{pairs}\n"
        else:
            NegoState_prompt += 'None\n'
        
        nego_flag = False
        Communication_prompt = "Communication log:\n" 
        for pair in pairs_player:
            player1_name = pair[0]
            player2_name = pair[1]
            if self.player2name[player1_name] == self.agent_name_list[self.env_agent_name_list.index(self.agent_name)]:
                player_name = player1_name
                parterner_name = player2_name
                nego_flag = True
                break
            if self.player2name[player2_name] == self.agent_name_list[self.env_agent_name_list.index(self.agent_name)]:
                player_name = player2_name
                parterner_name = player1_name
                nego_flag = True
                break
        
        if nego_flag: 
            if social_state[player_name][parterner_name]['parity'] == step%2:
                if 'proposal' in social_state[parterner_name][player_name]:
                    other_payoff = social_state[parterner_name][player_name]['proposal']*100
                    my_payoff = 100 - other_payoff
                    Communication_prompt += f"{self.player2name[parterner_name]} wants to gain {other_payoff}% reward and you will gain {my_payoff}% reward.\n"
                else:
                    Communication_prompt += f"Now is the time for you to negotiate.\n"
            else:
                Communication_prompt += f"Now is not the time for you to negotiate.\n"
        else:
            Communication_prompt += 'None\n'
        
        return (time_prompt + social_env_prompt + NegoState_prompt + Communication_prompt)

    def update_policy(self, obs):
        state_prompt = self.generate_state_prompt(obs)
        with open(f'{OUTPUT_DIR}output_{self.task}_{self.episode}.txt', 'a', encoding='utf-8') as file:
            print(f"\n### Observation module to {self.agent_name_list[self.agent_id]}", file=file)
            print(f"{state_prompt}", file=file)
        state_message = {"role": "user", "content": state_prompt}
        if self.gpt:
            self.planner.current_user_message = state_message
            response = self.planner.query(key=self.openai_api_key())
            self.planner.add_msg_to_dialog_history(state_message)
            self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
        else:
            step = obs['step']
            if self.agent_name_list[self.env_agent_name_list.index(self.agent_name)] == 'carpenter_0':
                if step == 0:
                    response = 'Communication: I will negotiate with miner_0. I will propose that I gain 60% reward and miner_0 will gain 40% reward.'
                else:
                    response = 'Communication: I will negotiate with miner_0. I will propose that I gain 60% reward and miner_0 will gain 40% reward.'
            elif self.agent_name_list[self.env_agent_name_list.index(self.agent_name)] == 'miner_0':
                if step == 0:
                    response = 'Communication: I will negotiate with carpenter_0. I will propose that I gain 40% reward and carpenter_0 will gain 60% reward.'
                elif step == 1:
                    response = 'Communication: I will make a new proposal. I will propose that I gain 40% reward and carpenter_0 will gain 60% reward.'
                else:
                    response = 'Communication: Accept carpenter_0\'s proposal. I will gain 40% reward and miner_0 will gain 60% reward.'
            elif self.agent_name_list[self.env_agent_name_list.index(self.agent_name)] == 'carpenter_1':
                if step == 0:
                    response = 'Communication: I will negotiate with miner_1. I will propose that I gain 60% reward and miner_1 will gain 40% reward.'
                else:
                    response = 'Communication: I will negotiate with miner_1. I will propose that I gain 60% reward and miner_1 will gain 40% reward.'
            elif self.agent_name_list[self.env_agent_name_list.index(self.agent_name)] == 'miner_1':
                if step == 0:
                    response = 'Communication: I will negotiate with carpenter_1. I will propose that I gain 40% reward and carpenter_1 will gain 60% reward.'
                elif step == 1:
                    response = 'Communication: I will make a new proposal. I will propose that I gain 40% reward and carpenter_1 will gain 60% reward.'
                else:
                    response = 'Communication: Accept carpenter_1\'s proposal. I will gain 40% reward and carpenter_1 will gain 60% reward.'
            
            # else:
            #     response = 'Communication: I will negotiate with carpenter_0. I will propose that I gain 90% reward and carpenter_0 will gain 10% reward.'
        self.language2action(response, obs, state_prompt)
        with open(f'{OUTPUT_DIR}output_{self.task}_{self.episode}.txt', 'a', encoding='utf-8') as file:
                print(f"\n### GPT Planner module", file=file)
                print(f"====== {self.agent_name_list[self.agent_id]} Query ======", file=file)
                print(response, file=file)

    def language2action(self, response, obs, state_prompt):
        if "Now is not the time for you to negotiate." in state_prompt:
            self.Action.move_action(4)
        else:
            social_state = obs['social_state']
            social_graph = obs['social_graph']
            invitable_players = self.state._find_invitable_players(social_graph, self.agent_id)
            if not isinstance(response, (str, bytes)):
                response = str(response)
            action_id = 0
            target_player_id = 0
            opponent_proposal = 0.5
            plan_type = None
            match = re.search(r"Communication: I will negotiate with (\w+)\.", response)
            if match:
                plan_type = "match"
                target_player_name = match.group(1)
                target_player_id = self.agent_name_list.index(target_player_name)
                action_id = target_player_id +  6 + 2 * self.Action.resource_num
                if target_player_id not in invitable_players:
                    plan_type = "propose"
                    match = re.search(r"I will propose that I gain (\d+)% reward and (\w+) will gain (\d+)% reward\.", response)
                    if match:
                        parter_name = match.group(2)
                        parter_name_player = self.name2player[parter_name]
                        bargain_action_id = int(match.group(1))
                        bargain_action_id = int((bargain_action_id+10)//10)
                        action_id = bargain_action_id +  6 + 2 * self.Action.resource_num + self.Action.player_num
                        target_player_name = match.group(2)
                        target_player_id = self.agent_name_list.index(target_player_name)
            match = re.search(r"Communication: I will make a new proposal.", response)
            if match:
                plan_type = "propose"
                match = re.search(r"I will propose that I gain (\d+)% reward and (\w+) will gain (\d+)% reward\.", response)
                if match:
                    parter_name = match.group(2)
                    parter_name_player = self.name2player[parter_name]
                    bargain_action_id = int(match.group(1))
                    bargain_action_id = int((bargain_action_id+10)//10)
                    action_id = bargain_action_id +  6 + 2 * self.Action.resource_num + self.Action.player_num
                    target_player_name = match.group(2)
                    target_player_id = self.agent_name_list.index(target_player_name)
            match = re.search(r"Communication: Accept (\w+)'s proposal\.", response)
            if match:
                plan_type = "accept"
                bargain_action_id = 0
                action_id = bargain_action_id +  6 + 2 * self.Action.resource_num + self.Action.player_num
                target_player_name = match.group(1)
                target_player_id = self.agent_name_list.index(target_player_name)
                parter_name = match.group(1)
                parter_name_player = self.name2player[parter_name]
                agent_name_player = self.name2player[self.agent_name_list[self.env_agent_name_list.index(self.agent_name)]]
                if agent_name_player in social_state[parter_name_player]:
                    if 'proposal' in social_state[parter_name_player][agent_name_player]:
                        opponent_proposal = social_state[parter_name_player][agent_name_player]['proposal']
            match = re.search(r"End. I chose to end this bargain.", response)
            if match:
                plan_type = "end"
                bargain_action_id = 1
                action_id = bargain_action_id +  6 + 2 * self.Action.resource_num + self.Action.player_num
                pairs = []
                pairs_player = []
                for u, v in social_state.edges():
                    if u != v and u < v:  # Ensure nodes are different and avoid duplicate pairs
                        if social_state.has_edge(v, u):  # Check if there is a reverse edge
                            # Check if both edges have the desired attribute value
                            if next(iter(social_state[u][v])) == "parity" and next(iter(social_state[v][u])) == "parity":
                                pairs_player.append((u, v))
                                u = self.player2name[u]
                                v = self.player2name[v]
                                if v == self.agent_name or u == self.agent_name:
                                    pairs.append((u, v))
                if len(pairs) > 0:
                    if pairs[0][0] == self.agent_name:
                        target_player_name = pairs[0][1]
                    else:
                        target_player_name = pairs[0][0]
                    parter_name = target_player_name
                    parter_name_player = self.name2player[parter_name]
                    target_player_id = self.agent_name_list.index(target_player_name)
                else:
                    plan_type = None

            if plan_type == "match":
                if target_player_id in invitable_players:
                    self.Action.request_matching_action(action_id)
                else:
                    self.Action.move_action(4)
            elif plan_type == None:
                self.Action.move_action(4)
            else:
                if target_player_id not in invitable_players:
                    flag = True
                    for i in range(self.agent_num):
                        if social_state.has_node(GROUP.format(i)):
                            coalition = social_state.successors(GROUP.format(i))
                            player = []
                            for key in coalition:
                                key = self.player2name[key]
                                player.append(key)
                            if target_player_name in player:
                                flag = False
                    if flag:
                        target_player_name = None
                        for pair in self.nego_state:
                            my_name = self.agent_name_list[self.env_agent_name_list.index(self.agent_name)]
                            if my_name == pair[0]:
                                target_player_name = pair[1]
                                break
                            if my_name == pair[1]:
                                target_player_name = pair[0]
                                break
                        if target_player_name is None:
                            self.Action.move_action(4)
                        else:
                            target_player_id = self.agent_name_list.index(target_player_name)
                            self.Action.bargain_action(action_id, target_player_id, opponent_proposal)
                    else:
                        self.Action.move_action(4)
                else:
                    self.Action.move_action(4)


class Negotiation_PhysicalAgent(BaseAgent):
    def __init__(self, info, task_info, agent_id, agent_name, task=None, episode=0, agent_name_list=['carpenter_0','carpenter_1','miner_0','miner_1'], env_agent_name_list = [], 
                 player2name = {},
                 model='gpt-3.5-turbo-0301',
                retrival_method="recent_k", K=1):
        super().__init__(model=model)
        self.task = task
        self.agent_num = info['player_num']
        self.episode = episode
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.state = State(self.agent_id, info, task_info)
        self.Action = Action(self.agent_id, info, task_info)
        self.agent_name_list = agent_name_list
        self.player2name = player2name
        self.event_IO = info['events']
        self.env_agent_name_list = env_agent_name_list
        self.inventory_max = info['inventory_capacity']

        self.stay = 0
        self.x = 0
        self.group_id = None
        self.current_plan = None
        self.current_comm = None
        self.res_in_obs = None
        self.res_in_obs_dist = None
        self.event_in_obs = None
        self.event_in_obs_dist = None
        self.inventory = None
        self.previous_inventory = None
        self.quantity = 0
        self.pre_position = None
        self.pre_action = None
        self.cur_position = None
        

        self.K = K
        self.retrival_method = retrival_method
        self.gpt = True

        self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K)
        with open(f'{OUTPUT_DIR}output_{self.task}_physical_{self.episode}.txt', 'a', encoding='utf-8') as file:
            print(self.planner.instruction_head_list[0]['content'], file=file)
    
    def generate_system_promt(self):
        prompt = ""
        if self.task == "easy_negotiation":
            prompt += "Instructions:\n"
            prompt += "- The AdaSociety game is an open-ended multi-agent environment. The game consists of a complex crafting tree, where the agent needs to obtain as many resources as possible in the limited time and craft tools to mine more advanced resources to maximize its benefit. At the same time, agents can also take other actions to help them increase their returns. The numbers of resources are limited.\n"
            prompt += "- Map: AdaSociety is a 2D grid-world game. The map size is 7*7.\n"
            prompt += "    - Natural resources: [Wood, Stone].\n"
            prompt += "    - Tools: [Hammer]\n"
            prompt += "    - Craft tree:\n"
            prompt += "        - 1 Wood + 1 Stone = 1 Hammer\n"
            prompt += "    - All gathered resources and tools are stored in the agent's inventory.\n"
            prompt += "    - When there are enough resources in the inventory, you can use the CRAFT <TOOL> action to synthesize the corresponding tools. For example, your inventory must contain wood and stone to craft a hammer.\n"
            prompt += "    - All crafts must be done on the corresponding event grid on the map. For example, a Hammer can be crafted ONLY on <Hammer Event>.\n"
            prompt += "    - Default amount of all units in crafts is 1.\n"
            prompt += "    - for carpenter, the value of wood is 1, the value of stone is 1, the value of hammer is 5.\n"
            prompt += "    - for miner, the value of wood is 1, the value of stone is 1, the value of hammer is 10.\n"
            prompt += "- Player:\n"
            prompt += "    - carpenter_0: can own many woods and stones but can own ONLY own 1 hammer in inventory.\n"
            prompt += "    - carpenter_1: can own many woods and stones but can own ONLY own 1 hammer in inventory.\n"
            prompt += "    - miner_0: can NOT own wood and stone, buy can own many hammers in inventory.\n"
            prompt += "    - miner_1: can NOT own wood and stone, buy can own many hammers in inventory.\n"
            prompt += "\n"
            prompt += f"Suppose you are a player named <{self.agent_name_list[self.agent_id]}> in the AdaSociety game. Your aim is to maximize your reward, which can gain from the resource value and the craft value.\n"
            prompt += "Join the coalition to share profits with other members according to the agreed-upon distribution ratio by division_weight.\n"
            prompt += "At each round in action phase, you will receive the current state:\n"
            prompt += "Step: ...\n"
            prompt += "Current surrounding social environment: ...\n"
            prompt += "payoff: The proportion of the split, shared within an coalition.\n"
            prompt += "Current surrounding physical environment: ...\n"
            prompt += "Your current inventory: ...\n"
            prompt += "\n"
            prompt += "In action phase, You should respond to me with\n"
            prompt += "Thoughts: (Your analysis to the current state)\n"
            prompt += "Plan: (The action you plan to take)\n"
            prompt += "\n"
            if "carpenter" in self.agent_name_list[self.agent_id]:
                prompt += "You should choose *ONLY ONE* Plan from the following four options: [GATHER <NUM> <RESOURCE>, CRAFT 1 <TOOL>, EXPLORE MAP, DUMP <TOOL>]. Here are explanations about them:\n"
                prompt += "- GATHER <NUM> <RESOURCE>: RESOURCE is chosen from the Natural resource list above. You shouldn't try to gather resources that aren't in your field of view because you don't know where they are. You should also not try to gather resources that are not natural resources.\n"
                prompt += "- CRAFT 1 <TOOL>: TOOL is chosen from the Tools list above. This plan can help you use the items in your inventory and follow the craft tree to craft the resources or tools you need. You can only use this plan if you have the corresponding event grid (i.e. the craft point) in your view. You should make sure you have enough material to craft.\n"
                prompt += "- EXPLORE MAP: This plan helps you move randomly to explore the map.\n"
                prompt += "- DUMP <TOOL>: TOOL is chosen from the Tools list above. The plan is to drop tools on the ground because some agents have a tool capacity of only 1. This action will decrease the corresponding item in the inventory by 1. If the item is not in the inventory, please do not choose this plan.\n"
                prompt += "\n"
                prompt += "<NUM> should be an integer not greater than 10.\n"
                prompt += "Please strictly follow the format above for the output.\n"
                prompt += "!!!Before making your crafting choice, please carefully check your inventory to ensure you have the necessary materials for crafting. And ensure that the tools in the inventory are fewer than the tool capacity. If there are excess tools, they should be discarded before crafting new tools. Random crafting selections are not allowed!!!\n"
                prompt += "!!!If your inventory don't have hammers, please not dump hammers!!!\n"
                prompt += "!!!craft hammer must need stone and wood, both stone and wood are indispensable.!!!\n"
                prompt += "\n"
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 50\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0', 'carpenter_1', 'miner_0', 'miner_1'}]\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Stone]. The distances of them are [5,4] steps away. The numbers of them are [5,4] respectively.\n"
                prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [0] steps away.\n"
                prompt += "The people in your observation are: [miner_1], The distances of them are [1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have 3 wood.\n"
                prompt += "\n"
                prompt += f"Thoughts: I'm {self.agent_name_list[self.agent_id]}, and I currently have 3 woods in my inventory. In my observation, there is wood and stone nearby, which I can gather. The Hammercraft event is also close by, allowing me to craft a hammer. But I hanve no enough material to craft hammer, so I need to gather resources. Since I have 3 woods, so I need to gather 3 stones.\n"
                prompt += "Plan: GATHER 3 STONE.\n"
                prompt += "\n"
                prompt += "###\n"
                prompt += "Step: 90\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0','miner_1'}, {'carpenter_1',  'miner_0'}]\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Stone]. The distances of them are [5,4] steps away. The numbers of them are [4,5] respectively.\n"
                prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [3] steps away.\n"
                prompt += "The people in your observation are: [miner_0, miner_1], The distances of them are [3,1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have 4 wood, 6 Stone.\n"
                prompt += "\n"
                prompt += f"Thoughts: I'm {self.agent_name_list[self.agent_id]}. I have 4 wood, 6 Stone. I am in a coalition with both carpenters and miners. The resources available are wood and stone, both of which are nearby. The hammercraft event is right next to me, allowing me to craft a hammer. I currently have more than 1 wood and more than 1 stone in my inventory, which is enough to craft a hammer. I have no hammers in the inventory. I choose to craft hammer.\n"
                prompt += "Plan: CRAFT 1 HAMMER.\n"
                prompt += "\n"
                prompt += "###\n"
                prompt += "Step: 90\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0','miner_1'}, {'carpenter_1',  'miner_0'}]\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Stone]. The distances of them are [5,4] steps away. The numbers of them are [4,5] respectively.\n"
                prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [3] steps away.\n"
                prompt += "The people in your observation are: [miner_0, miner_1], The distances of them are [3,1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have 4 wood, 6 Stone, 1 hammer.\n"
                prompt += "\n"
                prompt += f"Thoughts: I'm {self.agent_name_list[self.agent_id]}. I have 4 wood, 6 Stone, 1 hammer. I am in a coalition with both carpenters and miners. The resources available are wood and stone, both of which are nearby. The hammercraft event is right next to me, allowing me to craft a hammer. I currently have more than 1 wood and more than 1 stone in my inventory, which is enough to craft a hammer. Since I have one hammer and miner_1 who is also in my coalition is closer to me than miner0 in my observation, I should consider discarding my current hammer before crafting a new one to maximize the coalition's rewards, which miner_1 can pick the hammer and if miner_1 own hammer, it will gain more rewards than I own hammers.\n"
                prompt += "Plan: DUMP HAMMER.\n"
            else:
                prompt += "You should choose *ONLY ONE* Plan from the following two options: [GATHER 1 <TOOL>, EXPLORE MAP]. Here are explanations about them:\n"
                prompt += "- GATHER 1 <TOOL>: TOOL is chosen from the Tools list above. You shouldn't try to gather tools that aren't in your field of view because you don't know where they are.\n"
                prompt += "- EXPLORE MAP: This plan helps you move randomly to explore the map.\n"
                prompt += "Please strictly follow the format above for the output.\n"
                prompt += "\n"
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 50\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0', 'carpenter_1', 'miner_0', 'miner_1'}]\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Stone]. The distances of them are [5,4] steps away. The numbers of them are [5,4] respectively.\n"
                prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [0] steps away.\n"
                prompt += "The people in your observation are: [carpenter_0], The distances of them are [1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have NOTHING in your inventory.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}, and in the current coalition, there are both carpenters and miners. The resources available are wood and stone, and the hammercraft event is right next to me. Since I cannot gather wood and stone, I can also not craft hammer. I don't see a hammer in my field of view, so I need to explore the map to find one.\n"
                prompt += "Plan: EXPLORE MAP.\n"
                prompt += "\n"
                prompt += "###\n"
                prompt += "Step: 90\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0', 'carpenter_1', 'miner_0', 'miner_1'}]\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: ['wood', 'stone', 'hammer']. The distances of them are [1, 1, 0] steps away. The numbers of them are [5,4,1] respectively.\n"
                prompt += "The event grid in your observation are: ['hammercraft']. The distances of them are [0] steps away.\n"
                prompt += "The people in your observation are: [carpenter_0], The distances of them are [1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have NOTHING in your inventory.\n"
                prompt += "\n"
                prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}, and in the current coalition, there are both carpenters and miners. The resources available are wood and stone, and the hammercraft event is right next to me. Since I cannot gather wood and stone, I can also not craft hammer. I see a hammer in my field of view, so I need to gather one hammer.\n"
                prompt += "Plan: GATHER 1 HAMMER.\n"

        elif self.task == "hard_negotiation":
            prompt += "Instructions:\n"
            prompt += "- The AdaSociety game is an open-ended multi-agent environment. The game consists of a complex crafting tree, where the agent needs to obtain as many resources as possible in the limited time and craft tools to mine more advanced resources to maximize its benefit. At the same time, agents can also take other actions to help them increase their returns. The numbers of resources are limited.\n"
            prompt += "- Map: AdaSociety is a 2D grid-world game. The map size is 15*15.\n"
            prompt += "    - Natural resources: [Wood, Stone, Coal, Iron]. Some of them can only be discovered with some specific tools, which will be introduced next.\n"
            prompt += "    - Tools: [Hammer, Torch]\n"
            prompt += "    - Craft tree:\n"
            prompt += "        - 1 Wood + 1 Stone = 1 Hammer. With a Hammer, Coal can be gathered;\n"
            prompt += "        - 1 Coal + 1 Wood = 1 Torch. With a Torch, Iron can be discovered;\n"
            prompt += "    - All gathered and tools are stored in the agent's inventory.\n"
            prompt += "    - All crafts must be done on the corresponding event grid on the map. For example, your inventory must contain wood and stone to craft a hammer.\n"
            prompt += "    - Default amount of all units in crafts is 1.\n"
            prompt += "    - for carpenter, the value of wood is 1, the value of stone is 1, the value of hammer is 5, the value of coal is 10, the value of torch is 30, the value of iron is 20.\n"
            prompt += "    - for miner, the value of wood is 1, the value of stone is 1, the value of hammer is 5, the value of coal is 10, the value of torch is 30, the value of iron is 20.\n"
            prompt += "- Player:\n"
            prompt += "    - carpenter_0: You can gather many woods, stones and irons. You can not gather coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - carpenter_1: You can gather many woods, stones and irons. You can not gather coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - carpenter_2: You can gather many woods, stones and irons. You can not gather coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - carpenter_3: You can gather many woods, stones and irons. You can not gather coal. You can own many torchs. Your own inventory can ONLY own 1 hammer.\n"
            prompt += "    - miner_0: You can gather many woods and coals. You can not gather stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "    - miner_1: You can gather many woods and coals. You can not gather stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "    - miner_2: You can gather many woods and coals. You can not gather stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "    - miner_3: You can gather many woods and coals. You can not gather stone and iron. You can own many hammers. Your own inventory can ONLY own 1 torch.\n"
            prompt += "\n"
            prompt += f"Suppose you are a player named <{self.agent_name_list[self.agent_id]}> in the AdaSociety game. You are now in the action phase. Your aim is to maximize your reward, which can gain from the resource value and the craft value.\n"
            prompt += "You can not craft torchs, but you can craft hammers.\n"
            prompt += "Join the coalition to share profits with other members according to the agreed-upon distribution ratio.\n"
            prompt += "At each round in action phase, you will receive the current state:\n"
            prompt += "Step: ...\n"
            prompt += "Current surrounding social environment: ...\n"
            prompt += "payoff: The proportion of the split, shared within an coalition.\n"
            prompt += "Current surrounding physical environment: ...\n"
            prompt += "Your current inventory: ...\n"
            prompt += "\n"
            prompt += "In action phase, You should respond to me with\n"
            prompt += "Thoughts: (Your analysis to the current state)\n"
            prompt += "Plan: (The action you plan to take)\n"
            prompt += "\n"
            if "carpenter" in self.agent_name_list[self.agent_id]:
                prompt += "You should choose *ONLY ONE* Plan from the following four options: [GATHER <NUM> <WOOD/STONE/IRON/TORCH>, CRAFT 1 HAMMER, EXPLORE MAP, DUMP HAMMER]. Here are explanations about them:\n"
                prompt += "- GATHER <NUM> <WOOD/STONE/IRON/TORCH>: You shouldn't try to gather items that aren't in your field of view because you don't know where they are. You should also not try to gather item that are not in <WOOD/STONE/IRON/TORCH>. You can only choose one type of item in your plan.\n"
                prompt += "- CRAFT 1 HAMMER: This plan can help you use the items in your inventory and follow the craft tree to craft the resources or tools you need. You can only use this plan if you have the corresponding event grid (i.e. the craft point) in your view. You should make sure you have enough material to craft.\n"
                prompt += "- EXPLORE MAP: This plan helps you move randomly to explore the map.\n"
                prompt += "- DUMP HAMMER: The plan is to drop hammers on the ground because some agents have hammer's capacity of only 1. This action will decrease the corresponding item in the inventory by 1. If the item is not in the inventory, please do not choose this plan.\n"
                prompt += "\n"
                prompt += "<NUM> should be an integer not greater than 10.\n"
                prompt += "Please strictly follow the format above for the output.\n"
                prompt += "!!!Before making your crafting choice, please carefully check your inventory to ensure you have the necessary materials for crafting. And ensure that the tools in the inventory are fewer than the tool capacity. If there are excess tools, they should be discarded before crafting new tools. Random crafting selections are not allowed!!!\n"
                prompt += "!!!If your inventory don't have hammers, please not dump hammers!!!\n"
                prompt += "!!!craft hammer must need stone and wood, both stone and wood are indispensable.!!!\n"
                prompt += "\n"
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 50\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{carpenter_0, carpenter_1, carpenter_2, carpenter_3, miner_0, miner_1, miner_2, miner_3}].\n"
                prompt += "payoff: 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Stone]. The distances of them are [5,4] steps away. The numbers of them are [5,4] respectively.\n"
                prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [0] steps away.\n"
                prompt += "The people in your observation are: [miner_1], The distances of them are [1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have 3 wood.\n"
                prompt += "\n"
                prompt += f"Thoughts: I'm {self.agent_name_list[self.agent_id]}, and I currently have 3 woods in my inventory. In my observation, there is wood and stone nearby, which I can gather. The Hammercraft event is also close by, allowing me to craft a hammer. But I hanve no enough material to craft hammer, so I need to gather resources. Since I have 3 woods, so I need to gather 3 stones.\n"
                prompt += "Plan: GATHER 3 STONE.\n"
                prompt += "\n"
                prompt += "###\n"
                prompt += "Step: 90\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0', 'miner_1'}, {'carpenter_1', 'miner_1','miner_2', 'carpenter_2', 'miner_3', 'carpenter_3'}]\n"
                prompt += "payoff: 0.6, 0.1, 0.1, 0.2, 0.2, 0.4, 0.2, 0.2\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Stone]. The distances of them are [5,4] steps away. The numbers of them are [5,4] respectively.\n"
                prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [0] steps away.\n"
                prompt += "The people in your observation are: [miner_1], The distances of them are [1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have 4 wood, 6 Stone, 1 hammer.\n"
                prompt += "\n"
                prompt += f"Thoughts: I'm {self.agent_name_list[self.agent_id]}. In my coalition, there are mostly stones and a minority of wood. I can craft hammer heads first to help the coalition gain greater profits. miner_1 is in my coalition and he is closer than other people in order to my hammer don't be gathered by other coalition, miner own hammers can bring more rewards to the coalition, so I will dump hammer.\n"
                prompt += "Plan: DUMP HAMMER.\n"
                prompt += "\n"
            else:
                prompt += "You should choose *ONLY ONE* Plan from the following four options: [GATHER <NUM> <WOOD/COAL/HAMMER>, CRAFT 1 TORCH, EXPLORE MAP, DUMP TORCH]. Here are explanations about them:\n"
                prompt += "- GATHER <NUM> <WOOD/COAL/HAMMER>: You shouldn't try to gather items that aren't in your field of view because you don't know where they are. You should also not try to gather items that are not in <WOOD/COAL/HAMMER>. You can only choose one type of item in your plan.\n"
                prompt += "- CRAFT 1 TORCH: This plan can help you use the items in your inventory and follow the craft tree to craft the resources or tools you need. You can only use this plan if you have the corresponding event grid (i.e. the craft point) in your view. You should make sure you have enough material to craft.\n"
                prompt += "- EXPLORE MAP: This plan helps you move randomly to explore the map.\n"
                prompt += "- DUMP TORCH: The plan is to drop torchs on the ground because some agents have torch's capacity of only 1. This action will decrease the corresponding item in the inventory by 1. If the item is not in the inventory, please do not choose this plan.\n"
                prompt += "\n"
                prompt += "<NUM> should be an integer not greater than 10.\n"
                prompt += "Please strictly follow the format above for the output.\n"
                prompt += "!!!Before making your crafting choice, please carefully check your inventory to ensure you have the necessary materials for crafting. And ensure that the tools in the inventory are fewer than the tool capacity. If there are excess tools, they should be discarded before crafting new tools. Random crafting selections are not allowed!!!\n"
                prompt += "!!!If your inventory don't have torchs, please not dump torchs!!!\n"
                prompt += "!!!craft torch must need coal and wood, both coal and wood are indispensable.!!!\n"
                prompt += "\n"
                prompt += "Examples:\n"
                prompt += "###\n"
                prompt += "Step: 50\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{carpenter_0, carpenter_1, carpenter_2, carpenter_3, miner_0, miner_1, miner_2, miner_3}].\n"
                prompt += "payoff: 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Coal]. The distances of them are [4,5] steps away. The numbers of them are [4,5] respectively.\n"
                prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [0] steps away.\n"
                prompt += "The people in your observation are: [miner_1], The distances of them are [1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have 3 wood.\n"
                prompt += "\n"
                prompt += f"Thoughts: I'm {self.agent_name_list[self.agent_id]}, and I currently have 4 woods in my inventory. In my observation, there is wood and coal nearby, which I can gather. The Hammercraft event is also close by, but I can not craft a hammer. But I hanve no enough material to craft torch, so I need to gather resources. Since I have 4 woods, so I need to gather 4 coals.\n"
                prompt += "Plan: GATHER 3 COAL.\n"
                prompt += "\n"
                prompt += "###\n"
                prompt += "Step: 90\n"
                prompt += "Current surrounding social environment:\n"
                prompt += "[{'carpenter_0', 'miner_0'}, {'carpenter_1', 'miner_1','miner_2', 'carpenter_2', 'miner_3', 'carpenter_3'}]\n"
                prompt += "payoff: 0.6, 0.1, 0.1, 0.2, 0.4, 0.2, 0.2, 0.2\n"
                prompt += "Current surrounding physical environment:\n"
                prompt += "The resources in your observation are: [Wood, Coal]. The distances of them are [4,5] steps away. The numbers of them are [4,5] respectively.\n"
                prompt += "The event grid in your observation are: [TROCH Event]. The distances of them are [0] steps away.\n"
                prompt += "The people in your observation are: [carpenter_0], The distances of them are [1] steps away.\n"
                prompt += "Your current inventory:\n"
                prompt += "You have 4 wood, 6 coal, 1 torch.\n"
                prompt += "\n"
                prompt += f"Thoughts: I'm {self.agent_name_list[self.agent_id]}, and I have 4 wood, 6 coal and 1 torch in my inventory. I can craft torch heads first to help the coalition gain greater profits. carpenter_0 is in my coalition and he is closer than other people in order to my torch don't be gathered by other coalition, carpenter own torchs can bring more rewards to the coalition, so I will dump torch.\n"
                prompt += "Plan: DUMP TORCH.\n"

            prompt += "###\n"
            prompt += "Step: 50\n"
            prompt += "Current surrounding social environment:\n"
            prompt += "[{carpenter_0, carpenter_1, carpenter_2, carpenter_3, miner_0, miner_1, miner_2, miner_3}].\n"
            prompt += "payoff: 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2\n"
            prompt += "Current surrounding physical environment:\n"
            prompt += "The resources in your observation are: []. The distances of them are [] steps away. The numbers of them are [] respectively.\n"
            prompt += "The event grid in your observation are: [Hammer Event]. The distances of them are [0] steps away.\n"
            prompt += "The people in your observation are: [miner_0], The distances of them are [1] steps away.\n"
            prompt += "Your current inventory:\n"
            prompt += "You have NOTHING in your inventory.\n"
            prompt += "\n"
            prompt += f"Thoughts: I am {self.agent_name_list[self.agent_id]}, and in the current coalition, there are both carpenters and miners. The hammercraft event is right next to me. Since I have no wood and no stone, I can also not craft hammer. I don't see any resource in my field of view, so I need to explore the map to find one.\n"
            prompt += "Plan: EXPLORE MAP.\n"

        return prompt

    def create_gptmodule(self, module_name, retrival_method='recent_k', K=10):
        if not os.path.exists(PROMPT_DIR):
            os.makedirs(PROMPT_DIR)
        with open(f'{OUTPUT_DIR}output_{self.task}_physical_{self.episode}.txt', 'a', encoding='utf-8') as file:
            print(f"\n--->Initializing GPT {module_name}<---\n", file=file)
        system_promt = self.generate_system_promt()
        with open(f'{PROMPT_DIR}prompt_{self.task}_physical_{self.agent_name_list[self.agent_id]}.txt', 'a', encoding='utf-8') as file:
            print(f"{system_promt}", file=file)
        messages = [{"role": "system", "content": system_promt}]
        return Module(messages, self.model, retrival_method, K)

    def reset(self):
        if self.gpt:
            self.planner.reset()
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.current_timestep = 0
        self.teammate_ml_actions_dict = {}
        self.teammate_intentions_dict = {}

    def update_obs(self, observation, pre_position, pre_action):
        llm_obs = {}
        llm_obs['social_state'] = self.state.social_state2nx(observation["Social"]['global']['edges'])
        llm_obs['step'] = observation['step_id']
        llm_obs['position'] = pre_position[self.agent_id]
        llm_obs['action'] = pre_action[self.agent_id]
        llm_obs['current_pos'] = observation["Player"]['position']
        llm_obs['resource'] = observation["Map"]['resources']
        llm_obs['event'] = observation["Map"]['events']
        llm_obs['people'] = observation["Map"]['players']
        llm_obs['inventory'] = observation["Player"]['inventory']
        return llm_obs
    
    def generate_state_prompt(self, obs):
        
        social_state = obs['social_state']
        self.step = obs['step']
        self.resource = obs['resource']
        self.position = obs['current_pos']
        self.event = obs['event']
        self.people = obs['people']
        self.inventory_pre = obs['inventory']
        social_env_prompt = f"Current surrounding social environment:\n"

        for i in range(self.agent_num):
            social_env_prompt += f"coalition {i}:"
            if social_state.has_node(GROUP.format(i)):
                coalition = social_state.successors(GROUP.format(i))
                player = []
                for key in coalition:
                    key = self.player2name[key]
                    player.append(key)
                if len(player) > 0:
                    social_env_prompt += f"{player}.\n"
                else:
                    social_env_prompt += f"None.\n"
            else:
                social_env_prompt += f"None.\n"
            weight = list(social_state.edges(GROUP.format(i), data=True))
            if len(weight) > 0:
                social_env_prompt += "division_weight: "
            for j in range(len(weight)):
                social_env_prompt += f"{self.player2name[weight[j][1]]}: "
                social_env_prompt += str(weight[j][2]["score"])
                social_env_prompt += "; "
            social_env_prompt += "\n"
        '''Physical state'''
        time_prompt = f"Step: {self.step}\n"
        
        inventory_prompt = "Your current inventory:\nYour have "
        self.inventory = {}
        for i in range(len(self.inventory_pre)):
            inventory = self.inventory_pre[i]
            if inventory['name'] in self.inventory:
                self.inventory[inventory['name']] += inventory['amount']
            else:
                self.inventory[inventory['name']] = inventory['amount']
        for key in self.inventory.keys():
            inventory_prompt += f"{self.inventory[key]} {key},"
        if len(self.inventory) == 0:
            inventory_prompt += "None."
        inventory_prompt += "\n"

        current_obs_prompt = "Current surrounding physical environment:\n"
        self.resource_dis = {}
        self.resource_number = {}
        self.resource_pos = {}
        for i in range(len(self.resource)):
            resource = self.resource[i]
            if resource['name'] not in self.resource_number:
                self.resource_number[resource['name']] = resource['amount']
                self.resource_pos[resource['name']] = resource['position']
                self.resource_dis[resource['name']] = abs(resource['position'][0] - self.position[0]) + abs(resource['position'][1] - self.position[1])
            else:
                dis = abs(resource['position'][0] - self.position[0]) + abs(resource['position'][1] - self.position[1])
                if dis < self.resource_dis[resource['name']]:
                    self.resource_dis[resource['name']] = dis
                    self.resource_number[resource['name']] = resource['amount']
                    self.resource_pos[resource['name']] = resource['position']
        self.event_dis = {}
        self.event_pos = {}
        for i in range(len(self.event)):
            event = self.event[i]
            if event['name'] not in self.event_dis:
                self.event_dis[event['name']] = abs(event['position'][0] - self.position[0]) + abs(event['position'][1] - self.position[1])
                self.event_pos[event['name']] = event['position']
            else:
                dis = abs(event['position'][0] - self.position[0]) + abs(event['position'][1] - self.position[1])
                if dis < self.event_dis[event['name']]:
                    self.event_dis[event['name']] = dis
                    self.event_pos[event['name']] = event['position']
        self.people_dis = {}
        for i in range(len(self.people)):
            people = self.people[i]
            name = self.agent_name_list[self.env_agent_name_list.index(people["name"])]
            self.people_dis[name] = abs(people["position"][0] - self.position[0]) + abs(people["position"][1] - self.position[1])

        current_obs_prompt += f"The resources in your observation are: {list(self.resource_number.keys())}. The distances of them are {list(self.resource_dis.values())} steps away. The numbers of them are {list(self.resource_number.values())} respectively.\n"
        current_obs_prompt += f"The event grid in your observation are: {list(self.event_dis.keys())}. The distances of them are {list(self.event_dis.values())} steps away.\n"
        current_obs_prompt += f"The people in your observation are: {list(self.people_dis.keys())}. The distances of them are {list(self.people_dis.values())} steps away.\n"
        
        if self.task == 'hard_negotiation':
            hint_prompt = 'attention: '
            wood = False
            stone = False
            hammer = False
            torch = False
            coal = False
            for res_name in self.inventory.keys():
                resource_num = self.inventory[res_name]
                if resource_num > 0:
                    if res_name == "hammer":
                        if "carpenter" in self.agent_name_list[self.agent_id]:
                            hammer = True
                            hint_prompt += "You must dump hammer before craft hammer. "
                    if res_name == "torch":
                        if "miner" in self.agent_name_list[self.agent_id]:
                            torch = True
                            hint_prompt += "You must dump torch before craft torch. "
                    if res_name == "wood":
                        wood = True
                    if res_name == "stone":
                        stone = True
                    if res_name == "coal":
                        coal = True
            if "carpenter" in self.agent_name_list[self.agent_id]:
                if wood and stone and not hammer:
                    hint_prompt += "You cannot dump hammer. "
                elif wood and stone and hammer:
                    hint_prompt += "You cannot craft hammer. "
                else:
                    hint_prompt += "You cannot craft hammer. "
                    if not hammer:
                        hint_prompt += "You cannot dump hammer. "
            if "miner" in self.agent_name_list[self.agent_id]:
                if wood and coal and not torch:
                    hint_prompt += "You cannot dump torch. "
                elif wood and coal and torch:
                    hint_prompt += "You cannot craft torch. "
                else:
                    hint_prompt += "You cannot craft torch. "
                    if not torch:
                        hint_prompt += "You cannot dump torch. "
            hint_prompt += "\n"
        else:
            hint_prompt = 'attention: '
            wood = False
            stone = False
            hammer = False
            for res_name in self.inventory.keys():
                resource_num = self.inventory[res_name]
                if resource_num > 0:
                    if res_name == "hammer":
                        if "carpenter" in self.agent_name_list[self.agent_id]:
                            hammer = True
                            hint_prompt += "You must dump hammer before craft hammer. "
                    if res_name == "wood":
                        wood = True
                    if res_name == "stone":
                        stone = True
            if "carpenter" in self.agent_name_list[self.agent_id]:
                if wood and stone and not hammer:
                    hint_prompt += "You cannot dump hammer. "
                elif wood and stone and hammer:
                    hint_prompt += "You cannot craft hammer. "
                else:
                    hint_prompt += "You cannot craft hammer. "
                    if not hammer:
                        hint_prompt += "You cannot dump hammer. "
        hint_prompt += "\n"

        return (time_prompt + social_env_prompt + current_obs_prompt + inventory_prompt + hint_prompt)    
    
    def update_policy(self, obs):
        state_prompt = self.generate_state_prompt(obs)
        self.pre_action = obs['action']
        self.pre_position = obs['position']
        self.position = obs['current_pos']
        with open(f'{OUTPUT_DIR}output_{self.task}_physical_{self.episode}.txt', 'a', encoding='utf-8') as file:
            print(f"\n### Observation module to {self.agent_name_list[self.agent_id]}", file=file)
            print(f"{state_prompt}", file=file)
        if self.current_plan is None or self.check_previous_plan_illegal() or self.check_previous_plan_done():
            if self.stay > 10000:
                plan = None
            else:
                state_message = {"role": "user", "content": state_prompt}
                if self.gpt:
                    self.planner.current_user_message = state_message
                    response = self.planner.query(key=self.openai_api_key())
                    self.planner.add_msg_to_dialog_history(state_message)
                    self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
                else:
                    step = obs['step']
                    if self.agent_id == 0:
                        if step%4 == 0:
                            response = "Plan: GATHER 3 WOOD"
                        elif step%4 == 1:
                            response = "Plan: GATHER 3 STONE"
                        else:
                            response = "Plan: CRAFT 1 HAMMER"
                    else:
                        response = "Plan: EXPLORE MAP"
                with open(f'{OUTPUT_DIR}output_{self.task}_physical_{self.episode}.txt', 'a', encoding='utf-8') as file:
                    print(f"\n### GPT Planner module", file=file)
                    print(f"====== {self.agent_name_list[self.agent_id]} Query ======", file=file)
                    print(response, file=file)
                plan = self.language2plan(response)
                # if plan != None:
                #     match = re.match(r'(\w+) (\d+) (\w+)', plan)
                #     if match:
                #         self.quantity = int(match.group(2))
            self.stay = 0
            self.current_plan = plan
        self.plan_to_action(self.current_plan)
        self.previous_inventory = self.inventory

    def language2plan(self, response):
        if not isinstance(response, (str, bytes)):
            response = str(response)
        '''Physical action'''
        pattern_plan = r"Plan:\s*(.+)"
        match_plan = re.search(pattern_plan, response)
        plan = None
        if match_plan:
            plan = match_plan.group(1)
        return plan

    def plan_to_action(self, plan):
        if plan != None:
            match = re.match(r'(\w+) (\d+) (\w+)', plan)
            if match:
                plan_type = match.group(1)
                number = int(match.group(2))
                item = match.group(3)
            else:
                match = re.match(r'(\w+) (\w+)', plan)
                if match:
                    plan_type = match.group(1)
                    number = 1
                    item = match.group(2)
                else:
                    plan_type = plan.split()[0]
                    number = 0
                    item = None
        else:
            plan_type = None
            item = None
            number = 0
        self.plan_type = plan_type
        self.item = item
        self.number = number
        if plan_type == "GATHER":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            if target_resource_name not in self.resource_pos:
                random_move = random.randint(0, 4)
                self.Action.move_action(random_move)
            else:
                target_resource_pos = self.resource_pos[target_resource_name]
                if "move" in self.pre_action[0][0] and (self.pre_position == self.position).all():
                    random_move = random.randint(0, 4)
                    self.Action.move_action(random_move)
                else:
                    if target_resource_pos[1] < self.position[1]:
                        move = "move_left"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    elif target_resource_pos[1] > self.position[1]:
                        move = "move_right"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    elif target_resource_pos[0] < self.position[0]:
                        move = "move_up"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    elif target_resource_pos[0] > self.position[0]:
                        move = "move_down"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    else:
                        self.Action.pick_action(self.Action.resource_name.index(target_resource_name)+6)
        elif plan_type == "CRAFT":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            target_resource_name = PRODUCTS_TO_EVENT_NAME[target_resource_name]
            if target_resource_name not in self.event_pos:
                random_move = random.randint(0, 4)
                self.Action.move_action(random_move)
            else:
                target_resource_pos = self.event_pos[target_resource_name]
                if "move" in self.pre_action[0][0] and (self.pre_position == self.position).all():
                    random_move = random.randint(0, 4)
                    self.Action.move_action(random_move)
                else:
                    if target_resource_pos[1] < self.position[1]:
                        move = "move_left"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    elif target_resource_pos[1] > self.position[1]:
                        move = "move_right"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    elif target_resource_pos[0] < self.position[0]:
                        move = "move_up"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    elif target_resource_pos[0] > self.position[0]:
                        move = "move_down"
                        self.Action.move_action(self.Action.move_action_list.index(move))
                    else:
                        self.Action.produce_action()
        elif plan_type == "DUMP":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            if target_resource_name not in self.inventory:
                random_move = random.randint(0, 4)
                self.Action.move_action(random_move)
            else:
                self.Action.dump_action(self.Action.resource_name.index(target_resource_name)+6+self.Action.resource_num)
        elif plan_type == "EXPLORE":
            if self.stay == 0:
                random_move = random.randint(0, 4)
                self.x = random_move
            self.Action.move_action(self.x)
        else:
            with open(f'{OUTPUT_DIR}output_{self.task}_physical_{self.episode}.txt', 'a', encoding='utf-8') as file:
                print(f"Plan <{plan}> is illegal!!", file=file)
            random_move = random.randint(0, 4)
            self.Action.move_action(random_move)

    def check_previous_plan_illegal(self):
        self.stay += 1
        plan_type = self.plan_type
        item = self.item
        if plan_type == "GATHER":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            if target_resource_name not in self.resource_pos:
                return True
            if target_resource_name in self.inventory_max and self.inventory_max[target_resource_name] < 1 or (target_resource_name in self.inventory and target_resource_name in self.inventory_max and self.inventory_max[target_resource_name] < self.inventory[target_resource_name]):
                return True
        elif plan_type == "CRAFT":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            target_resource_name = PRODUCTS_TO_EVENT_NAME[target_resource_name]
            if target_resource_name not in self.event_pos:
                return True
            input_resource = self.event_IO[target_resource_name]['in']
            for key in input_resource.keys():
                if key not in self.inventory:
                    return True
                if self.inventory[key] < input_resource[key]:
                    return True
        elif plan_type == "DUMP":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            if target_resource_name not in self.inventory:
                return True
            if target_resource_name in self.inventory and self.inventory[target_resource_name] <= 0:
                return True
        return False

    def check_previous_plan_done(self):
        plan_type = self.plan_type
        item = self.item
        if plan_type == "GATHER":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            if target_resource_name not in self.inventory or target_resource_name not in self.previous_inventory:
                return False
            if self.previous_inventory[target_resource_name] == self.inventory[target_resource_name]:
                if self.number > 1:
                    return False
            if self.previous_inventory[target_resource_name] != self.inventory[target_resource_name] and self.number > 1:
                self.number -= 1
                return False
        elif plan_type == "DUMP":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            if self.inventory[target_resource_name] > 0 and self.number > 1:
                self.number -= 1
                return False
        elif plan_type == "CRAFT":
            target_resource_name = item
            target_resource_name = ''.join(char for char in target_resource_name if char.isalnum()).lower()
            if target_resource_name not in self.previous_inventory:
                return False
            if self.previous_inventory[target_resource_name] == self.inventory[target_resource_name]:
                if self.number > 1:
                    return False
            if self.previous_inventory[target_resource_name] != self.inventory[target_resource_name] and self.number > 1:
                self.number -= 1
                return False
        elif plan_type == "EXPLORE":
            if self.stay < 5:
                return False
        return True

