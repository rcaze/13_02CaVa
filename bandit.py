# This script is used to generate the figures for the first submission to BioCyb 2013 article

# Importing the necessary libraries, numpy, matplotlib, h5py, and random
import numpy as np # To compute and work with vectors
import random as rd # To generate random choices
import matplotlib.pyplot as plt # For plotting datas
import h5py # For storing datas in hdf5 files
from pdb import set_trace as flg

# Core functions

# Generate observations and estimations

def gobs(pval=[0.1,0.2], nep=20, nit=100):
    """
    Generate a set of observation from a binomial (or normal if p has two parameters) distribution with parameters p

    PARAMETERS
    ----------
    pval: a list
        different probabilities
    nep: positive integer
        number of episodes per trial
    nit: positive integer
        number of trials
    RETURNS
    -----
    An array of observations of dimension (nep,len(pval),nit)
    """
    return np.array([[[1 if rd.random() < q else -1 for i in range(nep)] for q in pval] for i in range(nit)])

def qnext(qprev=0, reward=1, aR=0.1, aP=0.5):
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
    if reward >= qprev:
        qnext = qprev + aR * ( reward - qprev )
    else:
        qnext = qprev + aP * ( reward - qprev )

    return qnext


# Global param
datadir = '/home/rcaze/Data/BioCyb2013/'
figdir='/home/rcaze/Content/Figures/BioCyb2013/'

# Default parameters
dpvals = [[0.1, 0.11, 0.12, 0.13, 0.14, 0.16, 0.17, 0.18, 0.19, 0.2],[0.8, 0.81, 0.82, 0.83, 0.84, 0.86, 0.87, 0.88, 0.89, 0.9]]
dagents = [[0.4,0.4],[0.4,0.1],[0.1,0.4],[]]
dnep = 1000
dnit = 100
dtemp = 0.3
dcolors = ('blue','green','red','violet')
dobs = gobs(dpvals[1], dnep, dnit)
dpex = 0.1

def checkEqual(iterator):
    """
    Test if the elements in a list are all identical
    """
    return len(set(iterator)) <= 1

def egreedy(qval=[0.5,0.9], pex = dpex):
    """
    Generate an e-greedy choice given a Q-value and a temperature parameter

    PARAMETERS
    ----------
    qval: list of floats
        Q-values each corresponding to an action
    pex: float
        Probability of switching

    RETURNS
    -------
    choice: integer
        Index of the chosen action
    """
    choice = 0
    rand = rd.random()
    if rand > pex and not checkEqual(qval):
        choice = np.argmax(qval)
        return choice
    else: #The choice is random if all option have the same Q-value
        return rd.randint(0,len(qval)-1)

def softmax(qval=[0.5,0.9], temp=dtemp):
    """
    Generate a softmax choice given a Q-value and a temperature parameter

    PARAMETERS
    ----------
    qval: list of floats
        Q-values corresponding to the different arm
    temp: float
        temperature parameter

    RETURNS
    -------
    choice : an integer, the index of the chosen action
    """
    qval = np.array(qval)
    gibbs = [np.exp(qval[qind]/temp)/np.sum(np.exp(qval/temp)) for qind in range(len(qval))]
    rand = rd.random()
    choice = 0
    partsum = 0
    for i in gibbs:
        partsum += i
        if partsum >= rand:
            return choice
        choice += 1

dfch = softmax
if dfch == softmax:
    dfname = 'nband%d_nep%d_nit%d_temp%d.h5' %(len(dpvals[0]), dnep, dnit, dtemp*10)
else:
    dfname = 'nband%d_nep%d_nit%d_egreedy_e%d.h5' %(len(dpvals[0]), dnep, dnit, dpex)

def testbed(obs=dobs, alpha=dagents[0], fch=dfch):
    """
    Launch a testbed determined by the obs array for an agent with one or two fix or plastic learning rates

    PARAMETERS
    -----
    obs: an array of observation correponding to the outcome of a choice
    alpha: a list, the two learning rates. If alpha is empty the learning rates are plastic
    fch: a function, the choice function softmax or egreedy

    RETURNS
    -------
    recch: an array recording the choice of the agent for each episode and each iteration
    recq: an array recording the internal q-values of the agent
    reca: an array recording the learning rates
    """
    nit = obs.shape[0] #Parsing of obs to extract the number of iteration/episodes/choices
    nch = obs.shape[1]
    nep = obs.shape[2]

    recch = np.zeros((nit, nep+1),np.int) #Initialize the record choice vector
    recq = np.zeros((nit, nep+1, nch)) #Initialize the record q-value vector
    reca = np.zeros((nit, nep+1, 2)) #Initialize the record alphas vector

    for cit in range(nit):
        if alpha != []: #Case of two plastic learning rate
            aR = alpha[0]
            aP = alpha[1]
        else: #Case of one or two fix learning rate
            nR, nP = 0, 0

        qest = np.zeros(nch) #Initialize Q-values at 0
        for cep in range(0,nep):
                choice = fch(qest) #Choosing given the Q value

                #being rewarded or not
                reward = obs[cit,choice,cep]

                #Updating the learning rate for the plastic agent
                if alpha == []: #Case of two plastic learning rate
                    nR += reward==1
                    nP += reward==-1
                    aP = (nR/float(nR+nP))/10
                    aR = (nP/float(nR+nP))/10

                qest[choice] = qnext(qest[choice], reward, aR, aP) #Update from the reward and choice
                recch[cit, cep+1] = choice
                recq[cit, cep+1, :] = qest
                reca[cit, cep+1, :] = [aR,aP]

    return recch[:,1:], recq[:,1:,:], reca[:,1:,:]

#Functions used for formal analysis plots

def qss(p=0.8, alpha=[0.1,0.5]):
        '''
        Compute the steady state Q value given p and the alphas

        PARAMETERS
        -----
        p: a float, the probability of reward
        alpha: a list of floats, the two learning rates for reward and punishment

        RETURNS
        -----
        qss : the q value at steady state

        '''
        aR, aP = alpha[0], alpha[1]
        qss = ((aR/aP)*p + (p-1)) / ((aR/aP)*p - (p-1))
        return qss

def xopt(pval=[0.8,0.9]):
        '''
        Compute optimal x ratio between learning rates which leads to the highest Delta Q for a given task

        PARAMETERS
        -----
        pval: a list, the probabilities of reward for each arm

        RETURNS
        -----
        xopt : a float, the q value at steady state
        dopt : a float, the optimal delta Q

        '''
        if pval[0]>pval[1]: #The bandit with the highest reward should be the last
            pval[0], pval[1] = pval[1], pval[0]
        qval = 1-np.array(pval)
        xopt = np.sqrt((qval[0]*qval[1]) / (pval[0]*pval[1]))
        dopt = (2*xopt*(pval[1]*qval[0]-pval[0]*qval[1])) / float((pval[1]*pval[0])*xopt**2+(pval[1]*qval[0]+pval[0]*pval[1])*xopt+qval[0]*qval[1])
        return xopt, dopt


#Handle function to record data

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
            hdf[Testbed.name]

    def show(self, save=False):
        """
        Plot the probability of picking the best option
        """
        plt.plot(self.choice==len(self.pval))


def h5name(pval=dpvals[0], alpha=dagents[0]):
        '''

        Generate the name of the data group to record Q values given pval and alpha

        PARAMETERS
        -----
        pval: a list, the probabilities of reward for each arm
        alpha: a list of floats, the two learning rates for reward and punishment

        RETURNS
        -----
        h5name : a char, the name of the group for the dataset

        '''
        task = "/p" + str(pval)
        if alpha==[]:
                agent = "/plas/" #The agent name can be complexify with the choice mechanism and its parameters
        else:
                agent = "/a" + str(alpha) + "/"
        return task + agent

def h5rec(data, dname, pval=dpvals[0], alpha=dagents[0], fname=dfname):
        '''

        Record either the choice, the obs, or the Q-values

        PARAMETERS
        -----
        data: an array, the data to be recorded
        dname: a char, the name of the dataset
        pval: a list, the probabilities of reward for each arm
        alpha: a list of floats, the two learning rates for reward and punishment
        fname: a char, the name of the file where to store the data

        RETURNS
        -----
        nothing

        '''
        hdf = h5py.File(datadir + fname,'a')
        gname = h5name(pval, alpha)
        if gname + dname not in hdf:
               hdf.create_dataset(gname + dname, data=data)
        hdf.close()

def h5out(dname, pval=dpvals[0], alpha=dagents[0], fname=dfname):
        '''

        Extract the datas from the h5 file

        PARAMETERS
        -----
        dname: a char, the name of the dataset
        pval: a list, the probabilities of reward for each arm
        alpha: a list of floats, the two learning rates for reward and punishment
        fname: a char, the name of the file where to store the data

        RETURNS
        -----
        data: an array, the dataset

        '''
        hdf = h5py.File(datadir + fname,'r')
        gname = h5name(pval, alpha)
        data = np.array(hdf[gname+dname])
        hdf.close()
        return data

# Wrap functions to run a series of testbed
def testbedwrap(pvals=dpvals, agents=dagents, fch=dfch, nep=dnep, nit=dnit, fname=dfname):
        '''

        Wrap different testbeds and record the data for different tasks and agents

        PARAMETERS
        -----
        pvals: a list of lists of floats, the probabilities of reward for each arm
        agents: a list of lists of floats, the two learning rates for reward and punishment
        fch: a function, the choice function
        nep: an int, the number of episodes per trial
        nit: an int, the number of trials
        temp: the temperature parameter of the softmax

        RETURNS
        -----
        fname: use for the plotting functions
        '''
        for pval in pvals: # Iterate through environments
            obs = gobs(pval, nep, nit)
            h5rec(obs, 'Observation', pval, 'Obs', fname)
            for agent in agents: # Iterate through agents
                recch, recq, reca = testbed(obs, agent, fch)
                h5rec(recch, 'Choice', pval, agent, fname)
                h5rec(recq, 'Q-values', pval, agent, fname)
                h5rec(reca, 'Learning rate', pval, agent, fname)
        return fname

#Functions for numerical analysis
def pss(chrec=[0,0,1,2,0,2,2]):
        '''

        Determine the probability of switch and staying given a choice record.

        PARAMETERS
        -----
        chrec: a list of ints, the choices made by the agent

        RETURNS
        -----
        switch: a float, probability of switching
        stay: a float, probability of staying
        '''
        switch = np.sum(chrec[1:] != chrec[:-1])
        stay = (len(chrec)-1) - switch
        return switch/float(len(chrec)-1) , stay/float(len(chrec)-1)


# Plot functions

# Useful functions
def barlabel(bars,labels):
        '''
        Labels a bar, used for bar plots

        PARAMETERS
        -----
        bars: a matplotlib object. The bar to label
        labels: a char. The label of the bar

        RETURNS
        -----
        a labelled bar

        '''
        for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x()+bar.get_width()/2., 1.02*h, '%s'%labels, ha='center',va='bottom', size=10)

def rspines(ax,places=['right','top']):
        '''
        Remove the spines and ticks in right and top, taken from the matplotlib gallery and updated by Romain Caze
        '''
        for loc, spine in ax.spines.iteritems():
                if loc in places :
                        spine.set_color('none') # don't draw spine
        #set off ticks where there is no spine
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

#Each functions correspond to a figure (Need to run simulations before)
def fig1(pvals=[[0.1,0.2],[0.8,0.9]], agents=[[0.4,0.1], [0.1,0.1], [0.1,0.4]], fname=dfname, fnamef='Fig1.svg'):
        '''
        Generate a figure of the final Q estimates for the three different types of agents

        PARAMETERS
        -----
        pvals: a list of lists of floats, the probabilities of reward for each arm
        agents: a list of lists of floats, the learning rates for reward and punishment of the different agents studied
        fname: a char, the name of the file where data` are stored
        fnamef: a char, the name of figure

        RETURNS
        -----
        nothing but record a plot similar to the one shown in the first figure of the article

        '''
        mean = [[[np.mean(h5out('Q-values', pval, agent, fname)[:,:,i]) for i, p in enumerate(pval)] for pval in pvals] for agent in agents]
        var = [[[np.var(h5out('Q-values', pval, agent, fname)[:,:,0]) for i, p in enumerate(pval)] for pval in pvals] for agent in agents]
        mean, var = np.array(mean), np.array(var)

        plt.figure(num=1, figsize=(5,3), dpi=300)

        for i, agent in enumerate(agents):
            for j, pval in enumerate(pvals):
                for k, p in enumerate(pval):
                    plt.scatter(i,mean[i][j][k], s=20, c='black')
                    plt.errorbar(i,mean[i,j,k],yerr=var[i,j,k], color='black')

        plt.xlabel(r'Agents', size=14)
        plt.xticks(np.arange(3),('Optimist','Rational','Pessimist'))
        plt.ylabel(r'$Q_{\infty}$', size=14)
        plt.ylim(-1,1)
        plt.savefig(fnamef,transparent=True)

def fig2a(pvals=dpvals, agents=dagents, fname=dfname, fnamef='Fig2a.svg'):
        '''

        Generate the figure plotting the probability of taking the best option

        PARAMETERS
        -----
        pvals: a list of lists of floats, the probabilities of reward for each arm
        agents: a list of lists of floats, the learning rates for reward and punishment of the different agents studied
        fname: a char, the name of the file where data` are stored
        fnamef: a char, the name of figure

        RETURNS
        -----
        nothing but record a plot similar to the one shown in the second figure of the article

        '''
        choices = [[np.mean(h5out('Choice', pval, agent, fname)==(len(pvals[0])-1),axis=0) for agent in agents] for pval in pvals]

        #return np.array(choices) #For now we use fig2a just for presenting datas

        plt.close('all')
        fig = plt.figure(num=2, figsize=(5,1), dpi=300)

        ax = fig.add_subplot(121)
        for i in range(len(dcolors)):
                plt.plot(choices[0][i],color=dcolors[i])

        ax.set_title('Low reward',size=10)
        ax.set_ylabel('P(Best Choice)',size=10)
        ax.set_xlim(0,dnep)
        ax.set_xlabel('Episodes',size=10)
        rspines(ax)

        ax = fig.add_subplot(122)
        for i in range(len(dcolors)):
                plt.plot(choices[1][i],color=dcolors[i])

        ax.set_title('High reward',size=10)
        ax.set_xlim(0,dnep)
        ax.set_xlabel('Episodes',size=10)
        rspines(ax)
        plt.savefig(figdir + fnamef,transparent=True)
        return np.array(choices)

def fig2b(pvals=[[0.1,0.2],[0.8,0.9]], agents=[[0.4,0.1], [0.1,0.1], [0.1,0.4]], fname=dfname, fnamef='Fig2b.svg'):
        '''

        Generate the figure plotting the probability of switch or staying

        PARAMETERS
        -----
        pvals: a list of lists of floats, the probabilities of reward for each arm
        agents: a list of lists of floats, the learning rates for reward and punishment of the different agents studied
        fname: a char, the name of the file where data` are stored
        fnamef: a char, the name of figure

        RETURNS
        -----
        switch: an array of floats, the probability of switching for the different agents
        nothing but record a plot similar to the one shown in the second figure (B) of the article

        '''
        switch = [[np.mean([pss(h5out('Choice', pval, agent, fname)[:,i])[0] for i in range(dnep)]) for agent in agents] for pval in pvals]
        switch = np.array(switch)

        fig = plt.figure(figsize=(5,1), dpi=300)
        taskname = ("Low reward", "High reward")
        labels = ('R','O','P')
        width = 0.80
        ind = np.arange(3)

        ax = fig.add_subplot(121)
        plt.bar(ind*width, switch[0,:], width, color=dcolors)
        ax.set_title(taskname[0], size=10)
        ax.set_ylabel('P(switching)', size=10)
        ax.set_ylim(0,1)
        plt.xticks(ind, labels, size=10)
        rspines(ax)

        ax = fig.add_subplot(122)
        plt.bar(ind*width, switch[1,:], width, color=dcolors)
        ax.set_title(taskname[1], size=10)
        ax.set_ylim(0,1)
        plt.xticks(ind, labels)
        rspines(ax)

        plt.savefig(figdir + fnamef,transparent=True)
        return switch

def fig2c(pvals=dpvals, agents=dagents, fname=dfname, fnamef='Fig2c.svg'):
        '''

        Generate the figure plotting the mean Q-values over a certain number of agents.

        PARAMETERS
        -----
        pvals: a list of lists of floats, the probabilities of reward for each arm
        agents: a list of lists of floats, the learning rates for reward and punishment of the different agents studied
        fname: a char, the name of the file where data` are stored
        fnamef: a char, the name of figure

        RETURNS
        -----
        nothing but record a plot for a supplementary figure with the evolution of the Q-values

        '''
        choices = [[np.mean(h5out('Q-values', pval, agent, fname),axis=0) for agent in agents] for pval in pvals]

        #return np.array(choices) #For now we use fig2a just for presenting datas

        plt.close('all')
        fig = plt.figure(num=2, figsize=(5,1), dpi=300)

        ax = fig.add_subplot(121)
        for i in range(len(dcolors)):
                plt.plot(choices[0][i],color=dcolors[i])

        ax.set_title('Low reward',size=10)
        ax.set_ylabel('Q-Values',size=10)
        ax.set_xlim(0,dnep)
        ax.set_xlabel('Episodes',size=10)
        rspines(ax)

        ax = fig.add_subplot(122)
        for i in range(len(dcolors)):
                plt.plot(choices[1][i],color=dcolors[i])

        ax.set_title('High reward',size=10)
        ax.set_xlim(0,dnep)
        ax.set_xlabel('Episodes',size=10)
        rspines(ax)
        #rspines(ax,places=['left','right','top'])
        plt.savefig(figdir + fnamef,transparent=True)
        return np.array(choices)



