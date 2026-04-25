#Import libraries
from gradient_bandit import GradientBandit
from simulation import simulateBandit
import random
import matplotlib.pyplot as plt

def main():
    #Parameters
    M = 1000 #Number of trials
    N = 1000 #Number of steps
    n = 10 #Number of arms
    true_mu = [random.gauss(10,1) for i in range(n)] #Expected value for each arm
    true_SD = [1 for i in range(n)] #Standard deviation of each arm
    alpha = 0.1
    baseline = 0.0
    drift = 0.1


    plt.figure(figsize=(10,6))

    average_reward_t1 = simulateBandit(GradientBandit,M,N,n,true_mu,true_SD,drift=drift,alpha=alpha)
    average_reward_t2 = simulateBandit(GradientBandit,M,N,n,true_mu,true_SD,drift=drift,alpha=alpha,baseline=5.0)

    cumulative_average1 = []
    cumulative_average2 = []
    running_sum1 = 0
    running_sum2 = 0
    for t in range(1,N+1):
        running_sum1 += average_reward_t1[t-1]
        running_sum2 += average_reward_t2[t-1]
        cumulative_average1.append(running_sum1 / t)
        cumulative_average2.append(running_sum2 / t)
    
    plt.plot(cumulative_average1, label="baseline=R_bar")
    plt.plot(cumulative_average2, label=f"baseline={baseline}")

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Reward")
    plt.title(f"R_bar vs {baseline} Baseline Bandit Performance")
    plt.legend()
    plt.grid(True, linestyle='--',alpha=0.7)
    plt.savefig("gradient_baseline.png")

if __name__ == "__main__":
    main()   

