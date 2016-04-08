import numpy as np # to compute and work with vectors
import random as rd # to generate random choices
import matplotlib.pyplot as plt # for plotting datas
import h5py # for storing datas in hdf5 files
from pdb import set_trace as flg

# Global param
datadir = '/home/rcaze/Data/BioCyb2013/'
figdir='/home/rcaze/Content/Figures/BioCyb2013/'


class Agent(object):
    """
    Defining a reinforcement learning agent
    """
    def __init__(self, aR=0.1, aP=0.1, plastic=False, n_choices=2, choice_temp=0.3):
        """
        qval: list of floats
            Q-values corresponding to the different arm
        temp: float
            temperature parameter
        """
        self.aR = aR
        self.aP = aP
        self.plastic = plastic
        #self.color = self._set_color()
        self.qval = range(n_choices)
        self.choice_temp = choice_temp

    def qnext(self, choice, reward):
        """
        Update a Q value given a distinct learning rate for reward and punishment

        PARAMETERS
        ----------
        qprev: float
            q-value to be changed
        reward: integer
            reward value
        aR: float
            learning rate for reward
        aP: float
            learning rate for punishment

        RETURNS
        -------
        Qnext: float
            estimation of the Q-value at the next time step
        """
        qprev = self.qval[choice]
        if reward >= qprev:
            qnext = qprev + self.aR * ( reward - qprev )
        else:
            qnext = qprev + self.aP * ( reward - qprev )
        self.qval[choice] = qnext

    def choice(self):
        """
        Generate a softmax choice given a Q-value and a temperature parameter

        RETURNS
        -------
        choice : an integer, the index of the chosen action
        """
        qval = np.array(self.qval)
        temp = self.choice_temp
        gibbs = [np.exp(qval[qind]/temp)/np.sum(np.exp(qval/temp)) for qind in range(len(qval))]
        rand = rd.random()
        choice = 0
        partsum = 0
        for i in gibbs:
            partsum += i
            if partsum >= rand:
                return choice
            choice += 1

class Testbed(object):
    """
    Testbed on a two armed bandit task
    """
    def __init__(self, agent=Agent(), pval=[0.1,0.2], n_episodes=10):
        self.agents = {agent}
        self.pval = pval
        self.n_bandits = len(pval)
        self.n_episodes = n_episodes
        self.choices = np.zeros(n_episodes)
        self.rewards = np.zeros(n_episodes)

    def run(self, save=False):
        nep = self.n_episodes
        self.obs =  np.array([[1 if rd.random() < q else -1 for i in range(nep)] for q in self.pval])
        for agent in self.agents:
            for c_episode in range(nep):
                choice = agent.choice()
                reward = self.obs[choice, c_episode]
                self.rewards[c_episode] = reward
                self.choices[c_episode] = choice
                agent.qnext(choice, reward)
                if agent.plastic:
                    agent.update_learning()

class DataBandit(object):
    def __init__(self, nit, nep, pval):
        self.nit = nit
        self.nep = nep
        self.pval = pval
        self.name = self._set_name()

    def _set_name(self):
        return 'nband{}_nep{}_nit{}.h5'.format(len(self.pval), self.nep, self.nit)

    def add_data(self, Testbed):
        with h5py.File(self.name, 'a') as hdf:
            hdf.create_dataset(Testbed.name, data=Testbed.data)

    def extract_data(self, Testbed, Agent):
        with h5py.File(self.name, 'r') as hdf:
            return np.array(hdf[Testbed.name])


