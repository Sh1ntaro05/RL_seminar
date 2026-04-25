#Import libraries
from epsilon_greedy_bandit import EpsilonGreedyBandit
from simulation import simulateBandit
import random
import matplotlib.pyplot as plt

def main():
    #Parameters
    M = 1000 #Number of trials
    N = 1000 #Number of steps
    n = 10 #Number of arms
    true_mu = [random.gauss() for i in range(n)] #Expected value for each arm
    true_SD = [1 for i in range(n)] #Standard deviation of each arm
    epsilon = 0.1
    drift = 0.1
    alpha = 0.1

    plt.figure(figsize=(10,6))

    average_reward_t1 = simulateBandit(EpsilonGreedyBandit,M,N,n,true_mu,true_SD,drift=drift,epsilon=epsilon)
    average_reward_t2 = simulateBandit(EpsilonGreedyBandit,M,N,n,true_mu,true_SD,drift=drift,epsilon=epsilon,alpha=alpha)

    cumulative_average1 = []
    cumulative_average2 = []
    running_sum1 = 0
    running_sum2 = 0
    for t in range(1,N+1):
        running_sum1 += average_reward_t1[t-1]
        running_sum2 += average_reward_t2[t-1]
        cumulative_average1.append(running_sum1 / t)
        cumulative_average2.append(running_sum2 / t)
    
    plt.plot(cumulative_average1, label="alpha=1/n")
    plt.plot(cumulative_average2, label="alpha=0.1")

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Constant vs. 1/n alpha Bandit Performance")
    plt.legend()
    plt.grid(True, linestyle='--',alpha=0.7)
    plt.savefig("const_v_var_alpha.png")

if __name__ == "__main__":
    main()   