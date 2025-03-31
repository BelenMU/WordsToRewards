# From Words to Rewards: Leveraging Natural Language for Reinforcement Learning
We explore the use of natural language for specifying rewards in Reinforcement Learning with Human Feedback (RLHF). Human language provides rich and nuanced information, yet most existing approaches rely on simplistic preference data or constrain the text structure. In contrast, we harness the power of Large Language Models (LLMs) to fully leverage natural text to efficiently train a reward model. Our empirical studies with human participants highlight the remarkable benefits of this strategy. Even with minimal human interaction, our method of integrating text feedback with LLMs accurately approximates the reward function and leads to significant 



## Gridworld

To create the environment

`conda env create -f environment.yml`
`conda activate env_gridworld`

## Rubik

There are two different notebooks:

1. `main.ipny` to interactively show the greedy policy and to update the reward model and policy. To run this notebook activate the `env_Rubik` environment.
2. `ask_GPT.ipny` to interpret human feedback to ChatGPT and translate into state-level rewards. To run this notebook activate the `env_GPT` environment.

These notebooks should be run iteratively. 

## Reacher
Activate the environment with `conda activate env_mujoco`
Run iteratively 
1. `policy_learning.ipynb` to update the agent given the rewards and to record a video to be shown to the oracle for feedback.
2. `HF_to_rewards.ipynb` to leverage an LLM to interpret the human feedback and transalte it into state-reward pairs.
