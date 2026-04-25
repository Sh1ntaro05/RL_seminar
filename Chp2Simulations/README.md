# Simulation of the Multi-Armed Bandit (Sutton & Barto, Ch. 2)

> **Summary:**
> Constructed 3 classes of bandits ($\epsilon$-greedy, UCB, and gradient) and conducted multiple experiments to visualize their behavior.

## Visual Output
![Comparison of the performance of 3 types of bandits in a stationary environment](assets/combo_plot.png)

---

## 1. Objective & Architecture

**The Problem:** An ML agent is faced with $k$ different actions, and each action returns a numerical reward based on an unknown stationary probability distribution (in this case, a Normal distribution with a randomly produced mean and a variance of 1). The objective is to maximize the total reward through a set number of repetitions. 

**The Approach:** Implemented 3 classes of bandits that run on different algorithms.

**Tech Stack:** Python 3, Matplotlib

## 2. Mathematical Formulation

The core engine relies on the following dynamics:

Let $A_i$ and $R_i$ be the action taken and the reward received on iteration $i$, respectively. Also, let $q_{*}(a)$ be the true mean reward of action $a$, and $Q_t(a)$ be the estimation of $q_{*}(a)$ at iteration $t$ defined as:

$$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \mathbf{1}_{\{A_i=a\}}}{\sum_{i=1}^{t-1} \mathbf{1}_{\{A_i=a\}}}$$

where $\mathbf{1}_{\{ \text{predicate} \}}$ denotes the indicator function that is 1 if the predicate is true, and 0 if it is not. If the denominator is 0, assign the value 0 to $Q_t(a)$. 

For each iteration, it is necessary to update the value of $Q_t(a)$. Letting $n$ be the number of times the action $a$ has been taken prior to iteration $t$, using the exact definition above takes $O(n)$ time to update. To achieve a more efficient $O(1)$ update time, letting $Q_n$ denote the estimated reward of action $a$ after it has been selected $n-1$ times, we use the following incremental computation:

$$Q_{n+1} = \frac{1}{n}\sum_{i=1}^n R_i = \frac{1}{n}((n-1)Q_n + R_n) = Q_n + \frac{1}{n}(R_n-Q_n)$$

We must also take into account the conflict between exploration and exploitation. Exploitation is when the agent takes its current knowledge and chooses the greedy action (the action that corresponds to the maximum prediction of reward). Exploring is when the agent selects a non-greedy action, which enables the improvement of current predictions. The following bandits balance this trade-off differently.

### a. $\epsilon$-Greedy Bandit

**Parameters:** * $\epsilon$: a small number between 0 and 1 that defines the probability of choosing an exploration action. 

**Algorithm:** For each iteration, with a probability of $\epsilon$, choose a random action. With a probability of $1 - \epsilon$, choose the action that gives the maximum value of $Q_t(a)$ (specifically, for iteration $t$, $A_t = \arg\max_a Q_t(a)$), and update $Q_t(a)$.

### b. UCB (Upper Confidence Bound) Bandit

**Parameters:** * $c$: a positive number that controls the degree of exploration.
* $N_t(a)$: the number of times that action $a$ has been selected prior to iteration $t$.
* $\alpha$: the step-size parameter, which controls how much past information is reflected in the update of $Q_t(a)$ (in the stationary case, $\alpha = 1/n$).

**Algorithm:** For each iteration, if there are actions that have not been selected yet, choose a random action from that group. If all actions have been selected at least once, the action for iteration $t$ is chosen as follows:

$$A_t = \arg\max_a \left( Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right)$$

**Handling Non-Stationarity:** While the stationary case uses $\alpha = 1/n$, this causes the agent to become "stubborn" over time. For non-stationary environments where reward means drift, I implemented a constant step-size $\alpha \in (0, 1]$. This effectively weights recent rewards exponentially higher than older ones, allowing the agent to "track" shifting optimal actions. 

### c. Gradient Bandit

**Parameters:** * $H_t(a)$: the preference of action $a$ at iteration $t$, initialized to 0 for all $a$.
* $\pi_t(a)$: the probability of choosing action $a$ at iteration $t$.
* $\alpha$: the step-size parameter.

**Algorithm:** For each iteration, the action is chosen with respect to the probability distribution $\pi_t(a)$. The values of $\pi_t(a)$ and $H_t(a)$ are updated as follows, where $R_t$ is the reward received at iteration $t$ and $\bar{R}_t$ is the average reward obtained prior to iteration $t$:

$$\pi_t(a) = \text{Pr}\{A_t=a\} = \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}}$$

$$H_{t+1}(A_t) = H_t(A_t) + \alpha (R_t - \bar{R}_t)(1 - \pi_t(A_t))$$

$$H_{t+1}(a) = H_t(a) - \alpha (R_t - \bar{R}_t)\pi_t(a) \quad \text{for all } a \neq A_t$$

## 3. Engineering Trade-offs & Edge Cases

**Avoiding Overflow:** Because $e^x$ grows explosively as $x$ increases, floating-point overflow is a risk in the gradient bandit. For each iteration, I calculated $h_t = \max_a H_t(a)$, subtracted it from both the numerator and denominator exponents, and calculated $\pi_t(a)$ as follows:

$$\pi_t(a) = \frac{e^{H_t(a)-h_t}}{\sum_{b=1}^k e^{H_t(b)-h_t}}$$

**Using a Baseline to Reduce Variance:** In the gradient bandit, using $\bar{R}_t$ (the average reward) acts as a baseline. Mathematically, it doesn't change the expected update, but it significantly reduces variance. In simulations, removing this baseline led to much slower convergence, especially when the mean rewards of all arms were shifted far from zero.

## 4. Quick Start

```bash
git clone [https://github.com/Sh1ntaro05/RL_seminar.git](https://github.com/Sh1ntaro05/RL_seminar.git)
cd RL_seminar
pip install -r requirements.txt
python experiment_combo.py