#Import libraries
import random
import matplotlib.pyplot as plt

#Does a simulation until terminal state is reached
#b is a dict that represents a prob dist with keys "L","R"
#Returns list S,A,R with length of T,T,T+1, respectively, also returns T
#R is indexed 1-bas, so ignore R[0]
def generate_epsiode(b):
    S = [0.0]
    A = []
    R = [0.0]
    T = 0
    state = 0
    while state == 0:
        if random.random() <= b["R"]:
            state = 1
            A.append("R")
            R.append(0.0)
        else:
            if random.random() <= 0.1:
                state = 1
                A.append("L")
                R.append(1.0)
            else:
                state = 0
                S.append(0)
                A.append("L")
                R.append(0.0)
        T += 1
    return S,A,R,T

#Evaluates target policy pi with behavior policy b
#Returns a list of length episodes that shows how v(s) changed over time
def policy_evaluation(pi,b,episodes):
    V = [0.0]
    i = 0
    C = 0.0
    gamma = 1.0
    for i in range(episodes):
        S,A,R,T = generate_epsiode(b)
        G = 0
        W = 1
        for t in range(T-1,-1,-1):
            G = gamma * G + R[t+1]
            W = W * pi[A[t]] / b[A[t]]
            
            C += W
            if C != 0:
                V.append(V[-1] + (W * (G - V[-1])) / C)
            else:
                V.append(V[-1])
    return V

def main():
    pi = {"L":1.0,"R":0.0}
    b = {"L":0.5,"R":0.5}
    episodes = pow(10,5)
    V_estimates = [policy_evaluation(pi,b,episodes) for i in range(10)]

    plt.figure(figsize=(12,6))
    for i, v_run in enumerate(V_estimates):
        plt.plot(v_run,linewidth=1,alpha=0.7,label=f'Run{i+1}')

    plt.axhline(y=1.0,color='r',linestyle='--',label='Theoretical V=1.0')

    plt.xscale('log')
    plt.xlabel("Episodes(log scale)")
    plt.ylabel("Value Estimate V(s)")
    plt.ylim(-0.1, 10.0)  
    plt.title("Off-Policy MC Prediction: Infinite Variance Case")
    plt.legend()
    plt.grid(True, which="both",ls="-",alpha=0.5)

    print("Saving")
    plt.savefig("infinte_var_test.png")
    #plt.show()
    

if __name__ == "__main__":
    main()