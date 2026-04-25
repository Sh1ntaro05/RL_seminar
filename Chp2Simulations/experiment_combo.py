#Import libraries
import random
from epsilon_greedy_bandit import EpsilonGreedyBandit
from ucb_bandit import UCBBandit
from gradient_bandit import GradientBandit
from simulation import simulateBandit
import matplotlib.pyplot as plt

def main():
    #Parameters
    M = 1000 #Number of trials
    N = 1000 #Number of steps
    n = 10 #Number of arms
    true_mu = [random.gauss() for i in range(n)] #Expected value for each arm
    true_SD = [1 for i in range(n)] #Standard deviation of each arm
    epsilon = 0.1
    c = 1
    alpha = 0.1

    plt.figure(figsize=(10,6))

    average_reward_t = [simulateBandit(EpsilonGreedyBandit,M,N,n,true_mu,true_SD,epsilon=epsilon),
                        simulateBandit(UCBBandit,M,N,n,true_mu,true_SD,c=c),
                        simulateBandit(GradientBandit,M,N,n,true_mu,true_SD,alpha=alpha)]
    cumulative_average = [[],[],[]]
    running_sum = [0,0,0]
    labels = [f"epsilon={epsilon}",f"c={c}",f"alpha={alpha}"]

    for i in range(3):
        for t in range(1,N+1):
            running_sum[i] += average_reward_t[i][t-1]
            cumulative_average[i].append(running_sum[i] / t)
        plt.plot(cumulative_average[i],label=labels[i])

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Epsilon greedy vs. UCB vs. Gradient Performance")
    plt.legend()
    plt.grid(True, linestyle='--',alpha=0.7)
    plt.savefig("combo_plot.png")


if __name__ == "__main__":
    main()