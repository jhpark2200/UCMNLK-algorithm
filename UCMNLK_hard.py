import os, sys
# sys.path.append('../')
sys.path.append('/Users/optalab/Documents/UCMNLK')
# sys.path.append('C:/Users/uqpua/OneDrive/Desktop/UCMNLK')                                                                                                                                                                                                                          

from tqdm import tqdm
import random
import numpy as np
from env_hard import *
from algorithms.ucmnlk_hard import *
from algorithms.optimal_policy import *
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
# import parmap


def run_experiment(seed):
    T = 5000
    env = make_hardToLearnMDP(T=T)
    random.seed(seed)
    np.random.seed(seed)
    eta = 0.5 * np.log(2) + 2
    D, d= 4, 8
    lam = 82 * np.sqrt(2) * (1 + d) * eta
    N = max(20, np.sqrt(D * T / d) * np.log(np.sqrt(T)/ d / D))
    gamma = 1 - np.sqrt(d / D / T)
    print(f'N: {N}, lam: {lam}, gamma: {gamma}')
    agent = UCMNLK(env, c = 5, gamma=gamma, N=int(N), eta=eta, lam=lam, T=T)    
    episodic_return = agent.run()
    return episodic_return

if __name__ == '__main__':
    # nState = 6
    T = 5000
    # env = make_hardToLearnMDP(T=T)
    runs = 100
    seeds = [1234*(i) for i in range(runs)]
    
    with Pool(8) as pool:
        run_returns = pool.map(run_experiment, tqdm(seeds))
        pool.close()  # 더 이상 프로세스를 받지 않음
        pool.join()   # 모든 프로세스가 끝날 때까지 대기하고 리소스 해제
    # run_returns = parmap.map(run_experiment, seeds, pm_pbar=True, pm_processes=8)

    # Save results
    print(run_returns)
    for idx, i in enumerate(run_returns):
        # print(i)
        np.save('/Users/optalab/Documents/UCMNLK/data/hardtolearn/T=' + str(T) + '/UCMNLK/return' + str(idx)+'.npy', i)