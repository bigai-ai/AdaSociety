from project.tasks.llm.contract.agent.agent import ContractAgent, Contract_PhysicalAgent
from project.tasks.llm.negotiation.agent.agent import NegotiationAgent, Negotiation_PhysicalAgent
from project.tasks.llm.social_structure.agent.agent import PhysicalAgent
import copy
import argparse
import numpy as np
import os

from project.tasks.llm.llm_env_wrapper import LLMEnvWrapper
OUTPUT_DIR = "./project/tasks/llm/outputs/"

def parse_args():
    parser = argparse.ArgumentParser()

    #===common env config===
    parser.add_argument('--task_name', type=str, choices = ["easy_contract",
                                                            "hard_contract","easy_negotiation","hard_negotiation",
                                                            "social_structure_unconnected","social_structure_connected","social_structure_ind_group",
                                                            "social_structure_ovlp_group","social_structure_hierarchical","social_structure_dynamic",], default='easy_contract')

    #===experiment===
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0301')
    parser.add_argument('--max_episodes', type=int, default=1)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    
    env = LLMEnvWrapper()
    task_info = env.env_handler.config_loader.config.task
    for episode in range(0, args.max_episodes):
        obs, info = env.reset()
        agent_num = len(info)
        env_agent_name_list = []
        for key in info.keys():
            env_agent_name_list.append(key)
        if agent_num == 4:
            agent_name_list = ['carpenter_0','carpenter_1','miner_0','miner_1']
            player2name = {'player_0':'carpenter_0','player_1':'carpenter_1','player_2':'miner_0','player_3':'miner_1'}
        if agent_num == 8:
            agent_name_list = ['carpenter_0','carpenter_1','carpenter_2','carpenter_3','miner_0','miner_1','miner_2','miner_3']
            player2name = {'player_0':'carpenter_0','player_1':'carpenter_1','player_2':'carpenter_2','player_3':'carpenter_3','player_4':'miner_0','player_5':'miner_1','player_6':'miner_2','player_7':'miner_3'}
        terminated_point = info[env_agent_name_list[0]]["max_length"] 
        agents = []
        physical_agents = []
        pre_action = [['None'] for _ in range(agent_num)]
        pre_position = [[-1, -1] for _ in range(agent_num)]
        reward_total = [0 for _ in range(agent_num)]
        if "contract" in args.task_name:
            phase1_length = 5 * info[env_agent_name_list[0]]["group_num"]
            for agent_id in range(agent_num):
                agent_name = env_agent_name_list[agent_id]
                agent = ContractAgent(info=info[agent_name], task_info = task_info, agent_id=agent_id, agent_name=agent_name, agent_name_list=agent_name_list, env_agent_name_list = env_agent_name_list, player2name=player2name, task=args.task_name, model=args.model)
                agent.reset()
                agents.append(agent)
            for agent_id in range(agent_num):
                agent_name = env_agent_name_list[agent_id]
                agent = Contract_PhysicalAgent(info=info[agent_name], task_info = task_info, agent_id=agent_id, agent_name=agent_name, agent_name_list=agent_name_list, env_agent_name_list = env_agent_name_list, player2name=player2name, task=args.task_name, model=args.model)
                agent.reset()
                physical_agents.append(agent)
        elif "negotiation" in args.task_name:
            phase1_length = info[env_agent_name_list[0]]["negotiation_steps"]
            for agent_id in range(agent_num):
                agent_name = env_agent_name_list[agent_id]
                agent = NegotiationAgent(info=info[agent_name], task_info = task_info, agent_id=agent_id, agent_name=agent_name, agent_name_list=agent_name_list, env_agent_name_list = env_agent_name_list, player2name=player2name, task=args.task_name, model=args.model)
                agent.reset()
                agents.append(agent)
            for agent_id in range(agent_num):
                agent_name = env_agent_name_list[agent_id]
                agent = Negotiation_PhysicalAgent(info=info[agent_name], task_info = task_info, agent_id=agent_id, agent_name=agent_name, agent_name_list=agent_name_list, env_agent_name_list = env_agent_name_list, player2name=player2name, task=args.task_name, model=args.model)
                agent.reset()
                physical_agents.append(agent)
        elif "social_structure" in args.task_name:
            phase1_length = 0
            for agent_id in range(agent_num):
                agent_name = env_agent_name_list[agent_id]
                agent = PhysicalAgent(info=info[agent_name], task_info = task_info, agent_id=agent_id, agent_name=agent_name, agent_name_list=agent_name_list, env_agent_name_list = env_agent_name_list, player2name=player2name, task=args.task_name, model=args.model)
                agent.reset()
                physical_agents.append(agent)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        for step in range(0, terminated_point):
            if step == phase1_length:
                agents = physical_agents
            if step >= phase1_length:
                with open(f'{OUTPUT_DIR}output_{args.task_name}_physical_{episode}_{args.model}.txt', 'a', encoding='utf-8') as file:
                    print(step, file=file)
            else:
                with open(f'{OUTPUT_DIR}output_{args.task_name}_{episode}_{args.model}.txt', 'a', encoding='utf-8') as file:
                    print(step, file=file)
            actions = {}
            for agent_id in range(agent_num):
                agent = agents[agent_id]
                agent_name = agent.agent_name
                llm_obs = agent.update_obs(obs[agent_name], pre_position, pre_action)
                agent.update_policy(llm_obs)
                action = agent.Action.action
                agent.Action.new()
                pre_action[agent_id] = copy.deepcopy(action)
                pre_position[agent_id] = copy.deepcopy(llm_obs['current_pos'])
                actions[agent_name] = action
                if step >= phase1_length-1:
                    with open(f'{OUTPUT_DIR}output_{args.task_name}_physical_{episode}_{args.model}.txt', 'a', encoding='utf-8') as file:
                        print(f"agent {agent.agent_name_list[agent.env_agent_name_list.index(agent_name)]}'s current plan is {agent.current_plan}",file=file)
                        print(f"agent {agent.agent_name_list[agent.env_agent_name_list.index(agent_name)]}'s action is {action}",file=file)
                else:
                    with open(f'{OUTPUT_DIR}output_{args.task_name}_{episode}_{args.model}.txt', 'a', encoding='utf-8') as file:
                        print(f"agent {agent.agent_name_list[agent.env_agent_name_list.index(agent_name)]}'s current plan is {agent.current_plan}",file=file)
                        print(f"agent {agent.agent_name_list[agent.env_agent_name_list.index(agent_name)]}'s action is {action}",file=file)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            obs = next_obs
            for agent_id in range(agent_num):
                reward_total[agent_id] += reward[env_agent_name_list[agent_id]]
            if step >= phase1_length:
                with open(f'{OUTPUT_DIR}output_{args.task_name}_physical_{episode}_{args.model}.txt', 'a', encoding='utf-8') as file:
                    print("########## RESULT ###########",file=file)
                    print("Step: ", step,file=file)
                    print("Reward: ", reward,file=file)
            else:
                with open(f'{OUTPUT_DIR}output_{args.task_name}_{episode}_{args.model}.txt', 'a', encoding='utf-8') as file:
                    print("########## RESULT ###########",file=file)
                    print("Step: ", step,file=file)
                    print("Reward: ", reward,file=file)
        # env.env.save_video()

        with open(f'{OUTPUT_DIR}output_{args.task_name}_physical_{episode}_{args.model}.txt', 'a', encoding='utf-8') as file:
            print(f"eps {episode} final payoff {reward_total}",file=file)