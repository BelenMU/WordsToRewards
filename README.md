# From Words to Rewards: Leveraging Natural Language for Reinforcement Learning
Belén Martín-Urcelay, Andreas Krause, Giorgia Ramponi. [TMLR 2026](https://openreview.net/forum?id=Gbx0pLANdf)


We explore the use of natural language to specify rewards in Reinforcement Learning with Human Feedback (RLHF). Unlike traditional approaches that rely on simplistic preference feedback, we harness Large Language Models (LLMs) to translate rich text feedback into state-level labels for training a reward model. Our empirical studies with human participants demonstrate that our method accurately approximates the reward function and achieves significant performance gains with fewer interactions than baseline methods.



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


## Citation

If you use this code, please cite:

```bibtex
@article{words2rewards,
  title   = {From Words to Rewards: Leveraging Natural Language for Reinforcement Learning},
  author  = {Martín-Urcelay, Belén and Krause, Andreas and Ramponi, Giorgia},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
