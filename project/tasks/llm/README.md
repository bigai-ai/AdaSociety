## LLM

### Deployment

Please refer to the [llm_main.py](../../../llm_main.py) file for details.

```python
# Retrieve the environment and LLM-related parameters，args.task_name is in ["easy_contract","hard_contract","easy_negotiation","hard_negotiation","social_structure"]
args = parse_args()
env = LLMEnvWrapper()
# Initialize the environment and obtain the agents’ observations.
obs, info = env.reset()
# Instantiate the agents, passing in environment-related information, agent features, agent name conversion table, task name, and the model used by the LLM. 
# LLMAgent can choose from [ContractAgent, Contract_PhysicalAgent, NegotiationAgent,Negotiation_PhysicalAgent,PhysicalAgent]. ContractAgent is used in the contract phase of contract minigame; Contract_PhysicalAgent is used in the physical phase of contract minigame;NegotiationAgent is used in the negotiation phase of negotiation minigame;Negotiation_PhysicalAgent is used in the physical phase of negotiation minigame;PhysicalAgent is used in the physical phase of social_structure minigame. The physical phase involves activities like gathering and crafting resources, without any social interaction.
# agent_name_list is the name of the agent in the prompt of the large model; env_agent_name_list is the name of the agent in the environment.
agent = LLMAgent(info，agent_id, agent_name, agent_name_list, env_agent_name_list, args.task_name, args.model)
# The agents convert the environment observations into the observations required by the LLM.
llm_obs = agent.update_obs(obs)
# The agents make decisions and take actions based on their observations, and the set of actions taken by all agents is denoted as actions.
agent.update_policy(llm_obs)
# store agent's action
action = agent.Action.action
# clean up agent's action
agent.Action.new()
# The set of actions is input into the environment, which then returns relevant information such as the agents’ observations, rewards, whether the game has ended, etc.
obs, reward, terminated, truncated, info = env.step(actions)
```

### How to use LLM

- The LLMAgent class is located in the three minigames of   [project/tasks/llm](.)
- `system_prompt = generate_system_prompt(env)` , retrieves information from the environment, converts it into the system part of the prompt for the LLM-based agent, which serves as the fixed prompt for each round of action. This prompt mainly includes the game rules, agent features, and some examples.
- `user_prompt = generate_state_prompt(obs)`, Receives the observations returned by the environment and converts them into the user part of the prompt for the LLM-based agent. This prompt changes with each round of action, primarily including the observations of each round and the set of actions available to the agent.
- `action = update_policy(system_prompt + user_prompt)`, The LLM outputs the corresponding action based on the prompt.
- You can change the LLM model being called by modifying args.model during the instantiation of the agents, `agent = LLMAgent(info，agent_id, agent_name, agent_name_list, env_agent_name_list, args.task_name, args.model)`
- The API key is located in [openai_key.txt](./openai_key.txt) file.

### Quickstart

Before starting the task, remember to change the loading path of the task from [main.json](../../../config/main.json) to the path of the corresponding task.

```bash
# easy_contract
python llm_main.py --task_name "easy_contract"
# easy_contract
python llm_main.py --task_name "hard_contract"
# easy_negotiation
python llm_main.py --task_name "easy_negotiation"
# hard_negotiation
python llm_main.py --task_name "hard_negotiation"
# social structure
  # "unconnected"
  python llm_main.py --task_name "social_structure_unconnected"
  # "connected"
  python llm_main.py --task_name "social_structure_connected"
  # "ind_group"
  python llm_main.py --task_name "social_structure_ind_group"
  # "ovlp_group"
  python llm_main.py --task_name "social_structure_ovlp_group"
  # "hierarchical"
  python llm_main.py --task_name "social_structure_hierarchical"
  # "dynamic"
  python llm_main.py --task_name "social_structure_dynamic"
```
