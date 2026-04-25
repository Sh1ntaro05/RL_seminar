#Import libraries
from epsilon_greedy_bandit import EpsilonGreedyBandit
import random
import matplotlib.pyplot as plt

def main():
    #Parameters
    M = 10 #Number of trials
    N = 1000 #Number of steps
    n = 10 #Number of arms
    true_mu = [random.gauss() for i in range(n)] #Expected value for each arm
    true_SD = [1 for i in range(n)] #Standard deviation of each arm
    epsilons = [i*0.01 for i in range(101)] #Each epsilon to test

    optimal_mu = max(true_mu)
    ideal_reward = N * optimal_mu
    total_regrets = []

    for i in range(len(epsilons)):
        total_reward = 0
        for j in range(M):
            bandit = EpsilonGreedyBandit(arms=n,epsilon=epsilons[i])
            for k in range(N):
                action = bandit.select_action()
                reward = random.gauss(true_mu[action], true_SD[action])
                bandit.update(action, reward)
                total_reward += reward / M
        total_regrets.append(ideal_reward-total_reward)

    plt.figure(figsize=(10,6))
    plt.plot(epsilons, total_regrets, marker='o', linestyle='-', markersize=4)

    plt.xlabel("Epsilon (ε)")
    plt.ylabel(f"Total Regret")
    plt.title("Total Regret vs. Epsilon (ε)")
    plt.grid(True, alpha=0.3)
    plt.savefig("bandit_v_epsilon.png")

if __name__ == "__main__":
    main()   




