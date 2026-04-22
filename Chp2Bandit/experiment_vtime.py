#Import libraries
from epsilon_greedy_bandit import EpsilonGreedyBandit
from simulation import simulateBandit
import random
import matplotlib.pyplot as plt

def main():
    #Parameters
    M = 100 #Number of trials
    N = 100 #Number of steps
    n = 10 #Number of arms
    true_mu = [random.gauss() for i in range(n)] #Expected value for each arm
    true_SD = [1 for i in range(n)] #Standard deviation of each arm
    epsilons = [0, 0.1, 0.5, 1] #Each epsilon to test
    
    plt.figure(figsize=(10,6))
    
    for i in range(len(epsilons)):
        average_reward_t = simulateBandit(EpsilonGreedyBandit,M,N,n,true_mu,true_SD,drift=0.0,epsilon=epsilons[i])

        cumulative_average = []
        running_sum = 0
        for t in range(1,N+1):
            running_sum += average_reward_t[t-1]
            cumulative_average.append(running_sum / t)

        plt.plot(cumulative_average, label=f"epsilon={epsilons[i]}")

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Epsilon-Greedy Multi-Armed Bandit Performance")
    plt.legend()
    plt.grid(True, linestyle='--',alpha=0.7)
    plt.savefig("bandit_v_time.png")

if __name__ == "__main__":
    main()   




    