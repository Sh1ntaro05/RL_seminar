#Import libraries
import random
import math

class UCBBandit:
    def __init__(self, arms, c, alpha=None):
        self.n = arms
        self.c = c
        self.alpha = alpha
        self.Q = [0 for i in range(self.n)]
        self.count = [0 for i in range(self.n)]
        self.t = 1

    def select_action(self):
        self.t += 1

        zeros = [i for i in range(self.n) if self.count[i] == 0]
        if len(zeros) == 0:
            ucb_values = []
            for i in range(self.n):
                ucb_values.append(self.Q[i] + self.c * math.sqrt(math.log(self.t) / self.count[i]))
            max_ucb = max(ucb_values)
            best_arms = [i for i, val in enumerate(ucb_values) if val == max_ucb]       
            return random.choice(best_arms)
        else:
            return random.choice(zeros)
        
    def update(self, action, reward):
        self.count[action] += 1

        if self.alpha is None:
            step_size = 1.0 / self.count[action]
        else:
            step_size = self.alpha

        self.Q[action] += step_size * (reward - self.Q[action])

   

