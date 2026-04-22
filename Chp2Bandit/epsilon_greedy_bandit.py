#Import libraries
import random

class EpsilonGreedyBandit:
    def __init__(self, arms, epsilon, alpha=None):
        self.n = arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = [0 for i in range(self.n)]
        self.count = [0 for i in range(self.n)]

    def select_action(self):
        if random.random() < self.epsilon: #Exploration
            return random.randint(0,self.n-1)
        else: #Exploitation
            max_Q = max(self.Q)
            best_arms = [i for i, Q in enumerate(self.Q) if Q == max_Q]
            return random.choice(best_arms)
    
    def update(self, action, reward):
        self.count[action] += 1
        
        if self.alpha is None:
            step_size = 1.0 / self.count[action]
        else:
            step_size = self.alpha

        self.Q[action] += step_size * (reward - self.Q[action]) 

def main():
    #Parameters
    N = 10 #Number of steps
    n = 10 #Number of arms
    true_mu = [random.gauss() for i in range(n)] #Expected value for each arm
    true_SD = [1 for i in range(n)] #Standard deviation of each arm

    bandit = EpsilonGreedyBandit(arms = n, epsilon = 0.1)
    results = []
    reward_sum = 0
    ideal_sum = N*max(true_mu)

    #Conduct RL
    for t in range(N):
        action = bandit.select_action()
        reward = random.gauss(true_mu[action], true_SD[action])
        reward_sum += reward
        bandit.update(action, reward)
        results.append([action, reward])

    #Display results
    print(f"Final Estimates: {bandit.Q}")
    print(f"True Expected Values: {true_mu}")
    print(f"Ideal Sum: {ideal_sum}")
    print(f"Actual Sum: {reward_sum}")
    print(f"Regret: {ideal_sum-reward_sum}")

if __name__ == "__main__":
    main()
