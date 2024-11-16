import numpy as np
from itertools import product

class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        '''
        Moves one step in the environment.
        Args:
            action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0
    

def make_hardToLearnMDP(T=1000):
    '''
    Makes a hard-to-learn MDP as described in Figure 1 of the paper.
    Args:
        T - int: total length (default 1000)
        d - int: dimension of the action space (default 10)
        delta - float: small probability for transitions (default 0.01)
    Returns:
        hardMDP - TabularMDP environment 
    '''
    d = 4
    nState = 2  
    nAction = 2**(d-1)

    R_true = {}
    P_true = {}
    states = {}
    for s in range(nState):
        states[(s)] = 0.0
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    for a in range(nAction):
        if s == 1:  # High reward 
            R_true[s, a] = (1, 0)
            # hardMDP.R[s, a] = (1, 0)
        else:
            R_true[s, a] = (0, 0)
            # hardMDP.R[s, a] = (0, 0)
    
    # Transitions    
    delta = 0.25
    # A = [[] for _ in range(nAction)]
    # for i in range(nAction):
    #     A[i] = [1, 1, 1, 1, 1, 1, 1]
    A = np.array(list(product([-1, 1], repeat=d-1)))
    # <a, \tehta>
    # a = [1,1,1,1,1,1,1,1]
    theta = [0.01] * (d-1)

    for a in range(nAction):
        P_true[0, a][0] = 1 - delta - np.dot(A[a], theta)
        P_true[0, a][1] = delta + np.dot(A[a], theta)


    for a in range(nAction):
        P_true[1, a][0] = delta
        P_true[1, a][1] = 1 - delta
    

    # Initialize states
    hardMDP = TabularMDP(nState, nAction, T)
    hardMDP.R = R_true
    hardMDP.P = P_true
    hardMDP.states = {s: 0.0 for s in range(nState)}
    # Always start in state 0
    hardMDP.reset = lambda: setattr(hardMDP, 'state', 0)
    
    return hardMDP


class TabularMDP(Environment):
    '''
    Tabular MDP
    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    '''

    def __init__(self, nState, nAction, T):
        '''
        Initialize a tabular [non-]episodic MDP
        Args:
            nState  - int - number of states
            nAction - int - number of actions
            T   - int - total length
        Returns:
            Environment object
        '''

        self.nState = nState
        self.nAction = nAction
        self.T = T
        # self.K = K

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        self.states = {}
        for state in range(nState):
            for action in range(nAction):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(nState) / nState
                
    def reset(self):
        "Resets the Environment"
        # self.timestep = 0
        self.state = 0
        
    def advance(self, action):
        '''
        Move one step in the environment
        Args:
        action - int - chosen action
        Returns:
        reward - double - reward
        newState - int - new state
        episodeEnd - 0/1 - flag for end of the episode
        '''
        if self.R[self.state, action][1] < 1e-9:
            # Hack for no noise
            reward = self.R[self.state, action][0]
        else:
            reward = np.random.normal(loc=self.R[self.state, action][0],
                                      scale=self.R[self.state, action][1])
        #print(self.state, action, self.P[self.state, action])
                

        newState = np.random.choice(self.nState, p=self.P[self.state, action])
        # Update the environment
        self.state = newState
        # self.timestep += 1

        # episodeEnd = 0
        # # if self.timestep == self.K:
        # if det_t > 2 * det_t_k
        #     episodeEnd = 1
        #     #newState = None
        #     self.reset()

        return reward, newState#, episodeEnd
    
    def argmax(self,b):
        #print(b)
        return np.random.choice(np.where(b == b.max())[0])
    
# make_riverSwim(100, 6)
# make_randomMDP(1000, 3, 3)
# make_hardToLearnMDP(1000) 