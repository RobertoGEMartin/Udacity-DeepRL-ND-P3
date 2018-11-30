[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Report
# "Project  - Collaboration and Competition"

We will train a system of DeepRL agents to demonstrate collaboration or cooperation on a complex task.

## Architecture

+ This image shows the flow of processes in a reinforcement learning training cycle.

![arch-rl](./img/arch-rl-intel.png "arch-rl")


+ In this project, we use Unity like environment simulator engine and the PyTorch framework to build the deep RL agent.

![arch-deeprl-unity](./img/arch-deeprl-unity-2.png "arch-deeprl-unity")


+ The next image defines the block diagram of ML-Agents toolkit for our sample environment. 
+ In our project, we use 2 agent.

![arch-unity-1.png](./img/arch-unity-1.png "arch-unity-1.png")

+ The next image overviews the high level flow of a MADDPG architecture
  
![arch-unity-1.png](./img/MADDPG-Architecture-1.png "arch-MADDPG-1.png")

+ The next image overviews the low level flow of a MADDPG architecture 
  
![arch-unity-1.png](./img/MADDPG-Architecture-2.png "arch-MADDPG-2.png")

## Unity Environment

+ Set-up: Two-player game where agents control rackets to bounce ball over a net.
+ Goal: The agents must bounce ball between one another while not dropping or sending ball out of bounds.
+ Agents: The environment contains two agent linked to a single Brain named TennisBrain. After training you can attach another Brain named MyBrain to one of the agent to play against your trained model.
+ Agent Reward Function (independent):
  + +0.1 To agent when hitting ball over net.
  + -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
+ Brains: One Brain with the following observation/action space.
  + Vector Observation space: 8 variables corresponding to position and velocity of ball and racket.
  + Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
  + Visual Observations: None.
+ Reset Parameters: One, corresponding to size of ball.
+ Benchmark Mean Reward: 2.5
+ Optional Imitation Learning scene: TennisIL

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


~~~~
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
~~~~

~~~~
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.
  1.          0.          0.          0.          0.          0.
  2.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
~~~~

## Code

The code is written in PyTorch 0.4 and Python 3.6.2.

Main Folders and files:  

+ ./apps/* : This folder should contain the unity apps (headless ubuntu and mac). These apps will simulate the Unity environment.
+ ddpq_agent.py: This file defines the DDPG agent.
+ model.py: This file defines the NN architecture [Actor and Critic].
+ Tennis_Rober.ipynb: This notebook will train the agent.
+ ./cp folder: This folder contains the checkpoints of trained agents: actor & critic.

## Learning Algorithm

We implement an artificial agent, termed [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)(DDPG)

DDPG is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

+ DDPG is an off-policy algorithm.
+ DDPG can only be used for environments with continuous action spaces.
+ DDPG can be thought of as being deep Q-learning for continuous action spaces.
+ DDPG can be implemented with parallelization

DDPG is a similarly foundational algorithm to VPG. DDPG is closely connected to Q-learning algorithms, and it concurrently learns a Q-function and a policy which are updated to improve each other.

Algorithms like DDPG and Q-Learning are off-policy, so they are able to reuse old data very efficiently. They gain this benefit by exploiting Bellman’s equations for optimality, which a Q-function can be trained to satisfy using any environment interaction data (as long as there’s enough experience from the high-reward areas in the environment).

### Implemtation

The Tennis Unity environment give us two agents to train.
In our implementation, these agents will share the actor network and the critic network.
These networks will be fully connected networks with the same inputs and outputs units.
You can see the details of conections between actor and critic in the next flowchart.

#### Details of implementation

+ We apply tanh activation to the output layer of actor network to bound the output between -1 and 1. Output layer has two units.
+ We apply relu activation to the output layer of critic network. Output layer has one unit.
+ Each agent uses the same actor network to take an action, sampled from a shared replay buffer.
+ Each agent uses the same critic network.
+ In the critic network, the action vector is added between the hidden layers.

#### DDPG Flowchart
![ddpg-flow-graph](./img/ddpg-flow-graph.png "ddpg-flow-graph")
+ Ref: https://nervanasystems.github.io/coach/algorithms/policy_optimization/ddpg/

### DDPG Pseudocode
![dpg-pseudocode](./img/ddpg-pseudocode.png "dpg-pseudocode")
+ Ref: https://spinningup.openai.com/en/latest/algorithms/ddpg.html#the-policy-learning-side-of-ddpg
  

### Hyper Parameters
#### DDPG Parameters

~~~~
INPUTS_UNITS = 512
OUTPUTS_UNITS = 384
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 3e-1              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
~~~~~

#### Neural Network. Model Architecture & Parameters
For this project we use these models:

~~~~
Actor NN: 
-----------
 Actor(
  (fc1): Linear(in_features=24, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=384, bias=True)
  (fc3): Linear(in_features=384, out_features=2, bias=True)
)
Critic NN: 
-----------
 Critic(
  (fcs1): Linear(in_features=24, out_features=512, bias=True)
  (fc2): Linear(in_features=514, out_features=384, bias=True)
  (fc3): Linear(in_features=384, out_features=1, bias=True)
)
~~~~

### Training

We trained the agents with several values of BATCH_SIZE {64,128,256}

+ With batchs of 64, we stoped the training after 1050 episodes
+ With batchs of 128, we solved the env in 932 episodes.
+ With batchs of 256, we solver the env in 672 episodes.

### Plot of Rewards
Environment solved in 932 episodes!	Average Score: 0.50
A plot of rewards per episode is included to illustrate that:
![report-ddpg-agent.png - gif](./img/report-ddpg-agent-932.png "report-ddpg-agent.png")

Environment solved in 672 episodes!	Average Score: 0.51
A plot of rewards per episode is included to illustrate that:
![report-ddpg-agent.png - gif](./img/report-ddpg-agent-672.png "report-ddpg-agent.png")

### Watch The DDPG Agent in Action

Video of trained DDPG Agent with checkpoints:

+ cp_actor_cc_agent_932.pth
+ cp_critic_cc_agent_932.pth

![Video of Training](./videos/trained-tennis-agents-932.gif "Video of Training")


[youtube video](https://youtu.be/BdRdK2KzHQM)

Video of trained DDPG Agent with checkpoints:

+ cp_actor_cc_agent_672.pth
+ cp_critic_cc_agent_672.pth

![Video of Training](./videos/trained-tennis-agents-672.gif "Video of Training")

[youtube video](https://youtu.be/tT1T2DkNKBM)

### Ideas for Future Work
Future ideas for improving the agent's performance.
+ Implement a real MADDPG where the actors and critics don't share NN.

+ Try new algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience

![drl-algorithms.png - gif](./img/drl-algorithms.png "drl-algorithms.png")

+ Try new algorithm [Hierarchical Actor Critic (HAC)](https://arxiv.org/abs/1712.00948.pdf) [code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ddpg_hac_agent.py). HAC enables agents to learn to break down problems involving continuous action spaces into simpler subproblems belonging to differenttime scales. The ability to learn at different resolutions in time may help overcome one of the main challenges in deep reinforcement learning — sample efficiency.
+ Try new algorithm [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495.pdf) [code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/memories/episodic/episodic_hindsight_experience_replay.py) 
  + [Video HER: Vanilla DDPG vs DDPG video](https://www.youtube.com/watch?time_continue=130&v=Dz_HuzgMxzo )


#### References
+ [Udacity Gihub Repo](https://github.com/udacity/deep-reinforcement-learning)
+ [Unity Docs](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md)
+ [Unity Paper](https://arxiv.org/abs/1809.02627)
+ [OpenAI master RL](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
+ [DDPG paper](https://arxiv.org/abs/1509.02971)
+ [OpenAI Baselines](https://blog.openai.com/better-exploration-with-parameter-noise/)
+ [Book: Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)
+ [Reinforcement Learning Coach by Intel® AI Lab](https://nervanasystems.github.io/coach/)
+ [RL Coach - DDPG - Docs](https://nervanasystems.github.io/coach/algorithms/policy_optimization/ddpg/)
+ [Modularized Implementation of Deep RL Algorithms in PyTorch](https://github.com/ShangtongZhang/DeepRL)
+ [Measuring collaborative emergent behavior in multi-agent reinforcement learning](https://www.researchgate.net/publication/326570321_Measuring_collaborative_emergent_behavior_in_multi-agent_reinforcement_learning)