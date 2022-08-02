# Awesome Exploration Methods in Reinforcement Learning 

The balance of **exploration and exploitation** is one of the most central problems in reinforcement learning.

Here is a collection of research papers for **Exploration methods in Reinforcement Learning (ERL)**.
The repository will be continuously updated to track the frontier of ERL. 

Welcome to follow and star!

## Table of Contents

- [A Taxonomy of Exploration RL Algorithms](#a-taxonomy-of-exploration-rl-algorithms)
- [Papers](#papers)
  - [Classic Exploration RL Papers](#classic-exploration-rl-papers)
  <!-- - [NeurIPS 2022](#nips-2022) (**<font color="red">New!!!</font>**)  -->
  - [ICML 2022](#icml-2022)
  - [ICLR 2022](#iclr-2022)
- [Contributing](#contributing)


## A Taxonomy of Exploration RL Algorithms

We simply divide `Exploration methods in RL` into four categories: `classic`, `intrinsic reward based`, `memory based`, `others`.
Note that there may be overlap between these categories, and an algorithm may belong to several of them.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./assets/erl-taxonomy.png" width=50% height=50%>
    <br>
    <center>A non-exhaustive, but useful taxonomy of algorithms in Exploration methods in RL.</center>
</center>

We give some examples algorithms for the different categories as shown in the figure above. 
There are links to algorithms in taxonomy.

>[1] [DQN-PixelCNN](https://arxiv.org/abs/1606.01868): Marc G. Bellemare et al, 2016  
[2] [#Exploration](http://papers.neurips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning.pdf) Haoran Tang et al, 2017  
[3] [EX2](https://papers.nips.cc/paper/2017/file/1baff70e2669e8376347efd3a874a341-Paper.pdf): Justin Fu et al, 2017  
[4] [ICM](https://arxiv.org/abs/1705.05363): Deepak Pathak et al, 2018  
[5] [RND](https://arxiv.org/abs/1810.12894): Yuri Burda et al, 2018  
[6] [VIME](https://arxiv.org/abs/1605.09674): Rein Houthooft et al, 2016  
[7] [EMI](https://openreview.net/forum?id=H1exf64KwH) (Exploration Policy Planning): Wang et al, 2019  
[8] [DIYAN](https://arxiv.org/abs/1802.06070): Benjamin Eysenbach et al, 2019  
[9] [NGU](https://arxiv.org/abs/2002.06038): Adrià Puigdomènech Badia et al, 2020  
[10] [Agent57](https://arxiv.org/abs/2003.13350): Adrià Puigdomènech Badia et al, 2020  
[11] [Go-Explora](https://www.nature.com/articles/s41586-020-03157-9): Adrien Ecoffet et al, 2021  
[12] [BootstrappedDQN](https://arxiv.org/abs/1602.04621): Ian Osband et al, 2016


## Papers

```
format:
- [title](paper link) (publisher, openreview score [if the score is public])
  - author1, author2, author3, ...
  - Key: key problems and insights
  - ExpEnv: experiment environments
```

### Classic Exploration RL Papers

- [Using Confidence Bounds for Exploitation-Exploration Trade-offs.](https://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf) *Journal of Machine Learning Research, 2002*
  - Peter Auer
  - Key:  linear contextual bandits
  - ExpEnv: None

- [How can we define intrinsic motivation?](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.567.6524&rep=rep1&type=pdf) *Conf. on Epigenetic Robotics, 2008.*
  - Pierre-Yves Oudeyer, Frederic Kaplan. 
  - Key: intrinsic motivation
  - ExpEnv: None

- [A Tutorial on Thompson Sampling](https://arxiv.org/pdf/1707.02038.pdf).
  - Daniel J. Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen
  - Key: Thompson sampling
  - ExpEnv: None

- [An empirical evaluation of thompson sampling.](http://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf) *NeurIPS 2011*
  - Olivier Chapelle and Lihong Li.
  - Key: Thompson sampling, empirical results
  - ExpEnv: None

- [Unifying Count-Based Exploration and Intrinsic Motivation.](https://arxiv.org/abs/1606.01868) *NeurIPS 2016*
  - Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Remi Munos
  - Key: intrinsic motivation, density models, pseudo-count
  - ExpEnv: [Atari](https://github.com/openai/gym)

- [Deep Exploration via Bootstrapped DQN.](https://arxiv.org/abs/1602.04621)  *NeurIPS 2016*
  - Ian Osband, Charles Blundell, Alexander Pritzel, Benjamin Van Roy
  - Key:  temporally-extended (or deep) exploration, randomized value functions, bootstrapped DQN
  - ExpEnv: [Atari](https://github.com/openai/gym)

- [VIME: Variational information maximizing exploration.](https://arxiv.org/abs/1605.09674)  *NeurIPS 2016*
  - Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel
  - Key:  maximization of information gain, belief of environment dynamics, variational inference in Bayesian neural networks
  - ExpEnv: [rllab](https://github.com/rll/rllab)

- [\#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning.](http://papers.neurips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning.pdf) *NeurIPS 2017*
  - Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel
  - Key: hash conut,  intrinsic motivation
  - ExpEnv: [rllab](https://github.com/rll/rllab), [Atari](https://github.com/openai/gym)

- [EX2: Exploration with Exemplar Models for Deep Reinforcement Learning.](https://papers.nips.cc/paper/2017/file/1baff70e2669e8376347efd3a874a341-Paper.pdf) *NeurIPS 2017*
   - Justin Fu, John D. Co-Reyes, Sergey Levine
   - Key: novelty detection, discriminatively trained exemplar models, implicit density estimation
   - ExpEnv: [VizDoom](https://github.com/mwydmuch/ViZDoom), 

- [Curiosity-driven exploration by self-supervised prediction.](https://arxiv.org/abs/1705.05363) *ICML 2017*
  - Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell
  - Key: curiosity, self-supervised inverse dynamics model
  - ExpEnv: [VizDoom](https://github.com/mwydmuch/ViZDoom), Super Mario Bros

- [Exploration by random network distillation.](https://arxiv.org/abs/1810.12894) *ICLR 2018*
  - Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov
  - Key: random network distillation
  - ExpEnv: [Atari](https://github.com/openai/gym)
  
- [Large-Scale Study of Curiosity-Driven Learning.](https://arxiv.org/abs/1808.04355)  *ICLR 2019*
  - Yuri Burda, Harri Edwards & Deepak Pathak, Amos Storkey, Trevor Darrell, Alexei A. Efros
  - Key:  curiosity, prediction error, purely curiosity-driven learning, feature spaces
  - ExpEnv: [Atari](https://github.com/openai/gym), Super Mario Bros

- [Diversity is all you need: Learning skills without a reward function.](https://arxiv.org/abs/1802.06070) *ICLR 2019*
  - Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine
  - Key:  maximizing an information theoretic objective, unsupervised emergence of diverse skills
  - ExpEnv: [MuJoCo](https://github.com/openai/mujoco-py)
  
- [Episodic Curiosity through Reachability.](https://arxiv.org/abs/1810.02274) *ICLR 2019*
  - Nikolay Savinov, Anton Raichuk, Rapha¨el Marinier, Damien Vincent, Marc Pollefeys, Timothy Lillicrap, Sylvain Gelly
  - Key:  curiosity,  episodic memory, how many environment steps it takes to reach the current observation
  - ExpEnv: [VizDoom](https://github.com/mwydmuch/ViZDoom), [DMLab](https://github.com/deepmind/lab), [MuJoCo](https://github.com/openai/mujoco-py)

- [Self-Supervised Exploration via Disagreement.](https://arxiv.org/abs/1906.04161) *ICML 2019*
  - Deepak Pathak, Dhiraj Gandhi, Abhinav Gupta
  - Key:  ensemble of dynamics models, maximize the disagreement of those ensembles, differentiable manner
  - ExpEnv: [Atari](https://github.com/openai/gym), MuJoCo, Unity, real robot

- [EMI: Exploration with Mutual Information.](https://arxiv.org/abs/1810.01176) *ICML 2019*
  - Hyoungseok Kim, Jaekyeom Kim, Yeonwoo Jeong, Sergey Levine, Hyun Oh Song
  - Key: embedding representation of states and actions, forward prediction, mutual Information
  - ExpEnv: [Atari](https://github.com/openai/gym), [MuJoCo](https://github.com/openai/mujoco-py)

- [Never give up: Learning directed exploration strategies.](https://arxiv.org/abs/2002.06038)  *ICLR 2020*
  - Adrià Puigdomènech Badia, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Bilal Piot, Steven Kapturowski, Olivier Tieleman, Martín Arjovsky, Alexander Pritzel, Andew Bolt, Charles Blundell
  - Key:  ICM+RND, different degrees of exploration/exploitation
  - ExpEnv: [Atari](https://github.com/openai/gym)
  
- [Agent57: Outperforming the atari human benchmark.](https://arxiv.org/abs/2003.13350) *ICML 2020* 
  - Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Charles Blundell
  - Key:  parameterizes a family of policies, adaptive mechanism, state-action value function parameterization
  - ExpEnv: [Atari](https://github.com/openai/gym), [roboschool](https://github.com/openai/roboschool)

- [Neural Contextual Bandits with UCB-based Exploration.](https://arxiv.org/pdf/1911.04462.pdf) *ICML 2020*
  - Dongruo Zhou, Lihong Li, Quanquan Gu
  - Key:  stochastic contextual bandit,  neural network-based random feature, near-optimal regret guarantee.
  - ExpEnv: contextual bandits, UCI Machine Learning Repository, mnist

- [First return then explore.](https://www.nature.com/articles/s41586-020-03157-9) *Nature 2021*
  - Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune
  - Key:  detachment and derailment, remembering states, returning to them, and exploring from them
  - ExpEnv: [Atari](https://github.com/openai/gym), pick-and-place robotics task
  

### ICML 2022

- [From Dirichlet to Rubin: Optimistic Exploration in RL without Bonuses](https://arxiv.org/pdf/2205.07704) (Oral)
  - Daniil Tiapkin, Denis Belomestny, Eric Moulines, Alexey Naumov, Sergey Samsonov, Yunhao Tang, Michal Valko, Pierre Menard.
  - Key: Bayes-UCBVI, regret bound, quantile of a Q-value function posterior, anticoncentration inequality for a Dirichlet weighted sum
  - ExpEnv: simple tabular grid-world env, [Atari](https://github.com/openai/gym)

- [The Importance of Non-Markovianity in Maximum State Entropy Exploration](https://arxiv.org/pdf/2202.03060) (Oral)
  - Mirco Mutti, Riccardo De Santi, Marcello Restelli
  - Key: maximum state entropy exploration, non-Markovianity,  finite-sample regime
  - ExpEnv: 3State and River Swim

- [Thompson Sampling for (Combinatorial) Pure Exploration](https://arxiv.org/abs/2206.09150) (Spotlight)
  - Siwei Wang, Jun Zhu
  - Key: combinatorial pure exploration, Thompson Sampling, lower complexity
  - ExpEnv: combinatorial multi-armed bandit

- [Near-Optimal Algorithms for Autonomous Exploration and Multi-Goal Stochastic Shortest Path](https://arxiv.org/pdf/2205.10729.pdf) (Spotlight)
  - Haoyuan Cai, Tengyu Ma, Simon Du
  - Key: incremental autonomous exploration, stronger sample complexity bounds, multi-goal stochastic shortest path
  - ExpEnv: hard MDP

- [Safe Exploration for Efficient Policy Evaluation and Comparison](https://arxiv.org/pdf/2202.13234.pdf) (Spotlight)
  - Runzhe Wan, Branislav Kveton, Rui Song
  - Key:  efficient and safe data collection for bandit policy evaluation.
  - ExpEnv: multi-armed bandit, contextual multi-armed bandit, linear bandits

- [Phasic Self-Imitative Reduction for Sparse-Reward Goal-Conditioned Reinforcement Learning](https://arxiv.org/abs/2206.12030) (Spotlight) 
  - Yunfei Li, Tian Gao, Jiaqi Yang, Huazhe Xu, Yi Wu
  - Key:  sparse-reward goal-conditioned, RL/SL phasic, task reduction
  - ExpEnv: Sawyer Push, Ant Maze, Stacking


### ICLR 2022

- [When should agents explore?](https://arxiv.org/abs/2108.11811) (Spotligt: 8, 8, 6, 6)
  - Miruna Pislar, David Szepesvari, Georg Ostrovski, Diana Borsa, Tom Schaul
  - Key: mode-switching, non-monolithic exploration, intra-episodic exploration
  - ExpEnv: [Atari](https://github.com/openai/gym)

- [Learning more skills through optimistic exploration](https://openreview.net/pdf?id=cU8rknuhxcDJ) (Spotlight: 8, 8, 8, 6)
  - DJ Strouse, Kate Baumli, David Warde-Farley, Vlad Mnih, Steven Hansen
  - Key: discriminator disagreement intrinsic reward, information gain auxiliary objective
  - ExpEnv: tabular grid world, [Atari](https://github.com/openai/gym)

- [Learning Long-Term Reward Redistribution via Randomized Return Decomposition](https://arxiv.org/abs/2111.13485) (Spotlight: 8, 8, 8, 5)
  - Zhizhou Ren, Ruihan Guo, Yuan Zhou, Jian Peng
  - Key: sparse and delayed rewards, randomized return decomposition, 
  - ExpEnv: [MuJoCo](https://github.com/openai/mujoco-py)

- [Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration](https://openreview.net/pdf?id=YJ1WzgMVsMt) (Spotlight: 8, 8, 8, 6, 6)
  - Desik Rengarajan,Gargi Vaidya,Akshay Sarvesh,Dileep Kalathil,Srinivas Shakkottai
  - Key: learning online with guidance offline
  - ExpEnv: [MuJoCo](https://github.com/openai/mujoco-py), TurtleBot(Waypoint tracking, Obstacle avoidance)

- [Generative Planning for Temporally Coordinated Exploration in Reinforcement Learning](https://openreview.net/pdf?id=YZHES8wIdE) (Spotlight: 8, 8, 8, 6)
  - Haichao Zhang, Wei Xu, Haonan Yu
  - Key: Generative Planning method, temporally coordinated exploration, crude initial plan
  - ExpEnv: classic continuous control env, CARLA

- [Multi-Stage Episodic Control for Strategic Exploration in Text Games](https://openreview.net/forum?id=Ek7PSN7Y77z) (Spotlight: 8, 8, 6, 6)
  - Jens Tuyls, Shunyu Yao, Sham M. Kakade, Karthik R Narasimhan
  - Key: multi-stage approach, policy decomposition, 
  - ExpEnv: Jericho

- [Learning Altruistic Behaviours in Reinforcement Learning without External Rewards](https://arxiv.org/abs/2107.09598) (Spotlight: 8, 8, 6, 6)
  - Tim Franzmeyer, Mateusz Malinowski, João F. Henriques
  - Key: altruistic behaviour, task-agnostic
  - ExpEnv: grid env, level-based foraging, multi-agent tag

- [Anti-Concentrated Confidence Bonuses for Scalable Exploration](https://arxiv.org/abs/2110.11202) (Poster: 8, 6, 5)
  - Jordan T. Ash, Cyril Zhang,Surbhi Goel,Akshay Krishnamurthy,Sham Kakade
  - Key: anti-concentrated confidence bounds, elliptical bonus
  - ExpEnv: multi-armed bandit, [Atari](https://github.com/openai/gym)

- [Lipschitz-constrained Unsupervised Skill Discovery](https://arxiv.org/abs/2202.00914) (Poster: 8, 6, 6, 6)
  - Seohong Park, Jongwook Choi, Jaekyeom Kim, Honglak Lee, Gunhee Kim
  - Key: unsupervised skill discovery, Lipschitz-constrained
  - ExpEnv: [MuJoCo](https://github.com/openai/mujoco-py)

- [On the Convergence of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning](https://openreview.net/forum?id=JzNB0eA2-M4)  (Poster: 8, 8, 5, 5)
  - Che Wang, Shuhan Yuan, Kai Shao, Keith Ross
  - Key: Monte Carlo Exploring Starts, Optimal Policy Feed-Forward MDPs
  - ExpEnv: blackjack, cliff Walking

- [LIGS: Learnable Intrinsic-Reward Generation Selection for Multi-Agent Learning](https://arxiv.org/pdf/2112.02618.pdf) (Poster: 8, 6, 5, 5)
  - David Henry Mguni,Taher Jafferjee,Jianhong Wang,Nicolas Perez-Nieves,Oliver Slumbers,Feifei Tong,Yang Li,Jiangcheng Zhu,Yaodong Yang,Jun Wang
  - Key: coordinated exploration and behaviour, learnable intrinsic-reward generation selection, switching controls
  - ExpEnv: foraging, StarCraft II
  

## Contributing
Our purpose is to provide a starting paper guide to who are interested in exploration methods in RL.
If you are interested in contributing, please refer to [HERE](CONTRIBUTING.md) for instructions in contribution.


## License
Awesome Exploration RL is released under the Apache 2.0 license.
