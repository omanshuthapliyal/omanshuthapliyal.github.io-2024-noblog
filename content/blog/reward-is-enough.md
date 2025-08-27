---
title: "Technology Readiness Levels of AI methods"
date: 2025-08-13T11:07:46-07:00
draft: false
katex: true
tags: [rl, ml, research]
# links:
#     website: "https://omanshuthapliyal.github.io/"
#     alias : "blog/reward-is-enough/"


---

# Reward is enough -- *when can we "reinforce" the learning?*

*This blog post is summarizes the papers **Settling the Reward Hypothesis**[^1] and **Utility Theory for Sequential Decision Making** [^2].*

The **reward hypothesis** is at the core of Reinforcement Learning (RL) in that whether sequential action problems can be posed as some sort of a reward maximization.  One of the founders of modern RL, Richard S. Sutton, puts it as[^3]: 
> "That all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward)."

In some ways, it conveys sufficiency of RL's capabilities, especially in modeling decision problems. It is often even called *the reinforcement learning hypothesis*. The conceptual basis of the hypothesis can be traced to von Neumann-Morgenstern (vNM) utility axioms from the foundational work for modern game theory: Theory of Games and Economic Behavior[^4]. While vNM deals with when can an agent's rational behavior can be a result of maximizing the expectation of some utility function (the utility function must exhibit Completeness, Transitivity, Continuity, and Independence), vNM does not impose any structure to the said utility function. 

So to understand the reward hypothesis, it doesn't hurt to first revisit the necessary and sufficient conditions that vNM provides for some policy preference to be expressible as some value function:

 - **Completeness**: some preference can be made when presented with 2 choices, i.e., either $apple$ is preferred over $banana$, vice versa, or both preferred equally.
 - **Transitivity**: if $apple$ is preferred over $banana$, and $banana$ preferred over $coconut$, then $apple$ is preferred over $coconut$.
 - **Continuity**: if $apple$ and $banana$ are two options, then one can form more options as some mixture as $x\% apple$ and $(1-x)\% banana$. 
 - **Independence**: suppose you are indifferent among $apple$, $banana$ and $coconut$. Further suppose that there are two coins with equal weights that when flipped give $\{H: apple, T:banana\}$ and $\{H: coconut, T:banana\}$. Then answer to "Do you prefer $apple$ over $banana$?", and "Which coin do you prefer to flip?" must be the same. 

Completeness and Transitivity simply allow you to compare between presented options. A consequence of continuity being that if you prefer $apple$ over $banana$, then there would be some $\alpha\in[0,1]$ for which your preference would flip. This is important. Independence implies, if both your options are truly equally preferred, then how the procurement between the two choices is carried out should not matter to your preference. In this way, if your preference relations satisfy the 4 properties above, you can make rational choices after maximizing some (scalar valued) utility function. RL simply imposes a specific structure of the utility function in this regard: the well known (discounted-)cumulative sum. 

However, this extension is non-trivial. vNM does not offer structures on policies that depend on past sequences and state transitions. That is, without additional structure to the problem, each trajectory would end up being assigned a different utility function, and no state transition (Controlled Markov Process) information would end up being incorporated. Shakerinava & Ravanbakhsh[^2] propose an additional property to be satisfied for mVN to extend to sequential decision making in Markovian settings: **memorylessness**. Memorylessness encodes exactly what it would seem to mean in Markovian settings. Suppose the decision making process has some underlying transition structure, e.g., choosing $apple$ moves state $s_0$ to $s_a$, choosing $banana$ moves state $s_0$ to $s_b$, choosing $coconut$ moves state $s_0$ to $s_c$, etc. Then if you are presented with a choice to choose among the 3 when you started from state $s_0$, and are now in state $s_*$, then you need not consider your past trajectory to make the optimal choice. 

RL problems are often more complicated than simple controlled Markov processes. One way in which RL differs is that, parts of your trajectory might need learning, and other parts might be know, so no optimization takes place of parts of the trajectory. Here, a final tweak to memorylessness is presented as **additivity** property in policies. It is difficult to present additivity as a sequential choice among $\{apple, banana, coconut\}$, but I will make an attempt. Suppose a sequential policy that is a mixture of $apple, banana$ strategy is preferred over a sequential policy that is a mixture of $coconut, durian$ in a trajectory $T_1$ where policy choices $apple, coconut$ start at state $s$ over $T_1$. Then, the same mixture preference holds over other trajectories $T_2$ leading to $s$ where policy choices $apple, coconut$ start at state $s$ (and vice versa -- it is an if and only if relation). More importantly for us, a consequence of additivity is that the known parts of a trajectory can be ignored, and the unknown parts optimized independently. If this reminds you of the Bellman optimality principle, then additivity (in conjunction with all other properties above), allow us to apply dynamic programming to RL, and allows agents to act optimally according to some expected cumulative sum of reward function in Markov decision processes. 

[^1]: Bowling, Michael, et al. "[Settling the Reward Hypothesis](https://arxiv.org/abs/2212.10420)", *International Conference on Machine Learning*. PMLR, 2023. 
[^2]: Shakerinava, Mehran, and Siamak Ravanbakhsh. "[Utility theory for sequential decision making](https://arxiv.org/abs/2206.13637)." *International Conference on Machine Learning*. PMLR, 2022.
[^3]: Richard Sutton, [The Reward Hypothesis](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/rewardhypothesis.html)
[^4]: [Theory of Games and Economic Behavior](https://press.princeton.edu/books/paperback/9780691130613/theory-of-games-and-economic-behavior)

> Written with [StackEdit](https://stackedit.io/).
