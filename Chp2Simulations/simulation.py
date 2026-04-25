#Import
import random

#Does M simulations of N steps with given environment factors(such as distributions of arms)
#and bandit attributes(such as epsilon and c) and returns the average reward at step t in list form
def simulateBandit(BanditClass, M:int, N:int, n:int, true_mu:list[float], true_SD:list[float],drift=0.0,**kwargs):
    average_reward_t = [0.0] * N
    
    for i in range(M):
        mu_copy = list(true_mu)
        SD_copy = list(true_SD)

        bandit = BanditClass(arms=n, **kwargs)
        for t in range(N):
            action = bandit.select_action()
            reward = random.gauss(mu_copy[action], SD_copy[action])
            bandit.update(action, reward)
            average_reward_t[t] += reward / M
            
            if drift > 0:
                for j in range(n):
                    mu_copy[j] += random.gauss(0,drift)
    
    return average_reward_t 
