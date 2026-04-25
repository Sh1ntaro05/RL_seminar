#Import libraries
import matplotlib.pyplot as plt

#Returns the probaility distribution of next state
#when given a potential state and current action
#For this case, p = 0.4(const.) of winning
def prob(state, action):
    return {state+action: 0.4,
            state-action: 0.6}
    
def main():
    states = [i for i in range(101)]
    actions = [i for i in range(len(states))]
    V = [0 for i in range(len(states))]
    V[100] = 1.0
    theta = 0.00001

    #Value iteration
    while True:
        delta = 0
        for state in range(1,len(states)-1):
            v = V[state]
            max_stake = min(state,100-state)
            v_max = 0
            for action in range(1, max_stake+1):
                v_max = max(v_max,
                prob(state,action)[state+action] * V[state+action] 
                + prob(state,action)[state-action] * V[state-action])
            V[state] = v_max
            delta = max(delta, abs(v - V[state]))
        if delta < theta: 
            break

    #Evaluating the best action at each state
    pi = [0 for i in range(len(states))]
    for state in range(1,len(states)-1):
        max_stake = min(state,100-state)
        maxarg = 1
        v_max = 0
        for action in range(1,max_stake+1):
            v = prob(state,action)[state+action] * V[state+action] + prob(state,action)[state-action] * V[state-action]
            v = round(v,5)
            if v > v_max:
                v_max = v
                maxarg = action
        pi[state] = maxarg
    
    #Display value function
    capital = range(1,100)
    values = V[1:100]
    policy = pi[1:100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    #Plot 1: Value Function
    ax1.plot(capital, values, color='#1f77b4', linewidth=1.5)
    ax1.set_title('Value Function ($p_h = 0.4$)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Capital')
    ax1.set_ylabel('Probability of Winning')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)

    #Plot 2: Optimal Policy
    ax2.bar(capital, policy, color='#ff7f0e', width=1.0, edgecolor='black', linewidth=0.5)
    ax2.set_title('Final Policy $\pi(s)$', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Capital')
    ax2.set_ylabel('Stake (Action)')
    ax2.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 55)

    plt.tight_layout()
    plt.savefig("value_optimal_policy_constant_prob.png")
            


if __name__ == "__main__":
    main()


            




