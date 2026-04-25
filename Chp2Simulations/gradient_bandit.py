#import libraries
import random
import math

class GradientBandit:
    def __init__(self, arms, alpha, baseline:float=None):
        self.n = arms
        self.alpha = alpha
        self.R_bar = 0.0
        self.t = 0
        self.baseline_val = baseline
        self.H = [0.0] * self.n
        self.pi = [1.0 / self.n] * self.n

    def select_action(self):
        return random.choices(range(self.n), self.pi)[0]
    
    def update(self, action, reward):
        self.t += 1

        curr_baseline = self.baseline_val if self.baseline_val is not None else self.R_bar
    
        for i in range(self.n):
            if i != action:
                self.H[i] -= self.alpha * (reward - curr_baseline) * self.pi[i]
            else: 
                self.H[action] += self.alpha * (reward - curr_baseline) * (1 - self.pi[action])

        if self.baseline_val is None:
            self.R_bar += (reward - self.R_bar) / self.t

        max_h = max(self.H)
        exps = [math.exp(self.H[i] - max_h) for i in range(self.n)]
        sum_exps = sum(exps)
        self.pi = [exps[i] / sum_exps for i in range(self.n)]
                
       
        

