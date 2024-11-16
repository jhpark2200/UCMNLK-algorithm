import numpy as np
import gurobipy as gp
from gurobipy import GRB
# import cvxpy as cp
from scipy.optimize import fmin_tnc
from tqdm import tqdm

class UCMNLK:
    def __init__(self, env, c, gamma, N, eta, lam, T):
        self.env = env
        # self.K = K  # number of episodes
        self.c = c  # confidence radius parameter
        self.gamma = gamma  # discount factor
        self.N = N  # number of DEVI rounds
        self.eta = eta
        self.T = T  # number of rounds

        # Feature dimensions
        self.d1 = self.env.nState * self.env.nAction
        self.d2 = self.env.nState
        self.d = self.d1 * self.d2

        # # reachable states
        self.reachable_states = {(s,a):{} for s in self.env.states.keys() for a in range(self.env.nAction)}


    def _initialize_phi(self):
        phi = {(s,a): np.zeros(self.d1) for s in self.env.states.keys() for a in range(self.env.nAction)}
        i = 0
        for key in phi.keys():
            phi[key][i] = 1
            i += 1
        return phi

    def _initialize_psi(self):
        psi = {s: np.zeros(self.d2) for s in self.env.states.keys()}
        j = 0
        for key in psi.keys():
            psi[key][j] = 1
            j += 1
        return psi

    def _initialize_varphi(self):
        varphi = {(s,a,s_): np.zeros(self.d) for s in self.psi.keys() for a in range(self.env.nAction) for s_ in self.reachable_states[s,a]}
        for s in self.psi.keys():
            for a in range(self.env.nAction):
                for s_ in self.reachable_states[(s,a)]:
                    varphi[(s,a,s_)] = np.outer(self.phi[(s,a)],
                                                 (self.psi[s_] - self.psi[self.support_states[s,a]])).flatten()
        return varphi

    def proj(self, x, lo, hi):
        return max(min(x, hi), lo)

    def DEVI(self, t_k):
        """
        Discounted Extended Value Iteration Algorithm
        
        Inputs:
        - gamma: discount factor
        - P: confidence polytope
        - N: number of rounds
        
        Returns:
        - Q: Action-value function
        """
        # Initialize
        Q = {(s, a): 1/(1 - self.gamma) for s in self.env.states.keys() for a in range(self.env.nAction)}
        V = {s: 1/(1 - self.gamma) for s in self.env.states.keys()}
        # Round
        for n in range(self.N):
            V_prev = V.copy()

            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                   
                    comp = [np.exp(np.dot(self.varphi[(s,a,s_)], self.theta)) for s_ in self.reachable_states[(s,a)]]
                    p_hat = comp / np.sum(comp)
                
                    sigma_norm = [np.sqrt(np.dot(np.dot(self.varphi[(s,a,s_)], self.Ainv), self.varphi[(s,a,s_)])) for s_ in self.reachable_states[(s,a)]] 
                    tmp_norm = np.max(sigma_norm)
                    R_t = min(2*self.Beta(t_k) * tmp_norm, len(self.reachable_states[(s,a)]))

                    # Create a Gurobi model
                    model = gp.Model("DEVI_optimization")
                    model.Params.LogToConsole = 0
                    # model.setParam('OutputFlag', 0)  # Suppress Gurobi output
                    
                    # Create variables
                    p = model.addVars(self.env.nState, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="p")
                    s_aux = model.addVars(len(self.reachable_states[(s,a)]), vtype=GRB.CONTINUOUS, name='auxiliary var') # Auxiliary variable
                    # Set objective
                    obj = gp.quicksum(p[s_next] * V_prev[s_next] for s_next in self.reachable_states[(s,a)])
                    model.setObjective(obj, GRB.MAXIMIZE)
                    
                    # Add constraints from the confidence polytope P
                    model.addConstr(gp.quicksum(p[s_next] for s_next in self.reachable_states[(s,a)]) == 1, "prob_sum")
                    # model.addConstr(gp.quicksum(gp.abs_(p[s_next] - p_hat[s_next]) for s_next in self.reachable_states[(s,a)]) <= R_t)
                    # Auxiliary constatnt
                    for s_next in range(len(self.reachable_states[(s,a)])):
                        model.addConstr(p[s_next] - p_hat[s_next] <= s_aux[s_next])
                        model.addConstr(p[s_next] - p_hat[s_next] >= -s_aux[s_next])
                    model.addConstr(gp.quicksum(s_aux) <= R_t)
                    
                    # Optimize the model
                    model.optimize()
                    
                    if model.status == GRB.OPTIMAL:
                        # Calculate Q-value
                        Q[s, a] = self.env.R[s,a][0] + self.gamma * model.objVal
                    else:
                        print(f"Optimization failed for state {s}, action {a}")

            if n == self.N-1:
                continue

            for s in self.env.states.keys():
                V[s] = max(np.array([Q[s, a] for a in range(self.env.nAction)]))
        self.Q = Q.copy()

    def Beta(self, t_k):
        return self.c * np.sqrt(self.d) * np.log((self.env.nState * t_k / (1 -self.gamma)))**2    # delta = 1 - gamma 

    def act(self, s):
        return self.env.argmax(np.array([self.Q[(s,a)] for a in range(self.env.nAction)]))
    
    def update_theta(self, X, Y):
        self.mnl.fit(X, Y, self.theta, self.eta, self.Ainv, self.A) # x, y, theta, eta, Ainv, A
        self.theta = self.mnl.w
    
    def compute_prob(self, theta, x):  # l_t (theta)
        probs = []
        for i in range(len(x)):
            means = np.dot(x[i], theta)
            u = np.exp(means)
            logSumExp = u.sum()
            prob = u/logSumExp
            probs.append(prob)
        return probs
    
    def update_gram_matrix(self, X):
        probs = self.compute_prob(self.theta, X)

        for i in range(len(X)):
            # print(probs[i], np.outer(X[i], X[i]))
            self.A += probs[i] * np.outer(X[i], X[i])
            self.Ainv -= probs[i] * np.dot((np.outer(np.dot(self.Ainv, X[i]), X[i])),self.Ainv)/ (1 + probs[i] * np.dot(np.dot(X[i], self.Ainv), X[i]))
            for j in range(len(X)):
                self.A -= probs[i] * probs[j] * np.outer(X[i], X[j])
                self.Ainv += probs[i] * probs[j] * np.dot((np.outer(np.dot(self.Ainv, X[i]), X[i])),self.Ainv)/ (1 + probs[i] * probs[j] * np.dot(np.dot(X[i], self.Ainv), X[i]))
                # print(probs[i], probs[j], )
        # self.Ainv = np.linalg.inv(self.A)
       
    def run(self):
        print("UCMNLK")
        episode_return = []  # round_return

        A_k = self.A.copy()
        # A_k = self.A
        t_k = 1
        R = 0  # cumulative reward
        for t in tqdm(range(1, self.T+1)):
            # self.env.reset()
            # print(t)
            if np.linalg.det(self.A) > 2*np.linalg.det(A_k):
                t_k = t
                # print(t_k)    
                # A_k = self.A
                A_k = self.A.copy()
                self.DEVI(t_k)
            
            s = self.env.state
            a = self.act(s)

            r, s_= self.env.advance(a)
            R += r
            # print(f't: {t}, R: {R}, s: {s}, a: {a}')
            X = []
            Y = []
            y = np.zeros(len(self.reachable_states[s,a]))
            for i in range(len(self.reachable_states[s,a])):
                    if list(self.reachable_states[s,a])[i] == s_:
                        y[i] = 1
                    X.append(self.varphi[(s,a,list(self.reachable_states[s,a])[i])])
                               
            self.Y.append(y)
            self.X.append(np.array(X))
            
            # update
            self.update_theta(self.X, self.Y)
            self.update_gram_matrix(X)

            episode_return.append(r)

        return episode_return

class OMD:
    # def __init__(self, theta):
    #     self.initial_theta = theta

    def compute_prob(self, theta, x):  # l_t (theta)

        return probs

    def cost_function(self, theta, x, y, eta, Ainv, A):

        return np.dot( np.dot((theta - theta_prime), A), (theta - theta_prime))

    def gradient(self, theta, x, y, eta, Ainv, A):  # 비용함수의 그래디언트
        return

    def fit(self, x, y, theta, eta, Ainv, A):
        return self
