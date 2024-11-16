import numpy as np

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
    
    
def make_riverSwim(T = 1000, nState=6):  
    '''
    Makes the benchmark RiverSwim MDP.
    Args:
        NULL - works for default implementation
    Returns:
        riverSwim - Tabular MDP environment '''
    nAction = 2
    R_true = {}
    P_true = {}
    states = {}
    for s in range(nState):
        states[(s)] = 0.0
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0.05, 0)
    R_true[nState - 1, 1] = (1.0, 0)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.0

    for s in range(1, nState - 1):
        P_true[s, 1][min(nState - 1, s + 1)] = 0.35
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.05

    P_true[0, 1][0] = 0.4
    P_true[0, 1][1] = 0.6
    P_true[nState - 1, 1][nState - 1] = 0.6
    P_true[nState - 1, 1][nState - 2] = 0.4

    riverSwim = TabularMDP(nState, nAction, T)
    riverSwim.R = R_true  # known
    riverSwim.P = P_true  # optimal: right direction
    riverSwim.states = states
    riverSwim.reset() 
    return riverSwim


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
    
