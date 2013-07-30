# This script is used to generate the figures of the BioCyb2012 article
# Importing the necessary library
import numpy as np # To compute and work with vectors
import random as rd # To generate random choices
import matplotlib.pyplot as plt # For plotting datas
import h5py # For storing datas in hdf5 files
from pylab import normpdf
from pdb import set_trace as flg
import copy

# Core functions

# Generate observations and estimations

def gobs(p=[0.8], nep=20):
    '''

    Generate a set of observation from a binomial (or normal if p has two parameters) distribution with parameters p

    PARAMETERS
    -----
    p: a list. The parameters of the distribution, one parameter for a binomial and two for a normal
    nep: a positive integer. The number of episodes

    RETURNS
    -----
    A list of observations following a binomial or a normal distribution

    '''
    if len(p)==2:
        return np.random.normal(p[0],p[1],nep)
    return [1 if rd.random() < p[0] else -1 for i in range(nep)]

def vqnext(qprev=np.array([[0.1,0.2],[0.3,0.4]]), alphas=np.array([[0.1,0.2],[0.3,0.4]]), choice=np.array([0,1]), ob=np.array([1,-1])):
    '''

    Update Q-values of n agents depending on the learning rate of each agent. It is a vectorized functions

    PARAMETERS
    -----
    qprev: a list of list, number of choices * number of agents
    alphas: a list of list, two learning rates * number of agents
    choices: a list, the choices of all agents
    ob: a list, the observation for each agent

    RETURNS
    ----
    qnext: the updated q-values for all agents

    '''
    qnext = np.array(qprev)
    nagents = qprev.shape[0]
    dim = np.arange(nagents) #use for the fancy indexing

    comparison = ob <= qprev[dim,choice] #Creating an array to determine the chosen alpha
    alphas = alphas[dim,comparison.astype(np.int)] #Selecting the set of alphas for the different agents
    update = alphas * (ob - qprev[dim,choice]) #Adding the update
    qnext[dim,choice] = qprev[dim,choice] + update #Computing the selected new q value
    return qnext

# The decision mechanisms

def softmax(qval=[0.5,0.9], t=0.1):
    '''

    Generate a softmax choice given a Q-value and a temperature parameter

    PARAMETERS
    -----
    qval : a list of Q-value each corresponding to an action
    t: the temperature parameter

    RETURNS
    -----
    choice : an integer, the index of the chosen action

    '''
    qval = np.array(qval)
    gibbs = [np.exp(qval[qind]/t)/np.sum(np.exp(qval/t)) for qind in range(len(qval))]
    rand = rd.random()
    choice = 0
    partsum = 0
    for i in gibbs:
        partsum += i
        if partsum > rand:
            return choice
        choice += 1

def checkEqual(iterator):
    '''
    Test if the elements in a list are all identical
    '''
    return len(set(iterator)) <= 1

dpex=0.1

def egreedy(qval=[0.5,0.9], pex = dpex):
    '''

    Generate an e-greedy choice given a Q-value and a temperature parameter

    PARAMETERS
    -----
    qval : a list of Q-value each corresponding to an action
    pex: the probability of switching

    RETURNS
    -----
    choice : an integer, the index of the chosen action

    '''
    choice = 0
    rand = rd.random()
    if rand > pex and not checkEqual(qval):
        choice = np.argmax(qval)
        return choice
    else:
        return rd.randint(0,len(qval)-1)

# Default parameters
dpvals = [[[0.1],[0.2]],[[0.8],[0.9]]]
dagents = [[0.1,0.1],[0.4,0.1],[0.1,0.4],[]]
dnep = 800
dnit = 5000
dfch = softmax
ddatdir = '/home/rcaze/Data/BioCyb2012/'
dfname = 'sotfmax4bigHigh.h5'
dfname = ddatdir + dfname
ddatdirf='/home/rcaze/Content/Figures/BioCyb2012/'
dcolors = ('blue','green','red','violet')

# Simulating a bandit task with obs.shape[2] iterations of obs.shape[1]  episodes of obs.shape[0] choices
def testbed(pval=dpvals[0], alpha=dagents[0], nep=dnep, nit=dnit, fch=dfch):
    '''
    Launch a multi-armed bandit task for an agent with two learning rates fix or plastic

    PARAMETERS
    -----
    pval: a list, the parameters of the probability distribution governing the reward distribution
    alpha: a list, the two learning rates. If alpha is empty the learning rates are plastic.
    nep: the number of episodes during one trial
    nit: the number of iterations/agents which are launch simultaneously on a given task
    fch: the policy either softmax or e-greedy

    RETURNS
    ----
    recch: an array recording the choice of the agent for each episode and each iteration
    recq: an array recording the internal q-values of the agent
    reca: an array recording the learning rates

    '''
    obs = np.array([[gobs(cpval,nep) for cpval in pval] for cit in range(nit)])

    nit = obs.shape[0] #Parsing of obs to extract the number of iteration/episodes/choices
    nch = obs.shape[1]
    nep = obs.shape[2]
    recch = np.empty((nit,nep),np.int) #Initialize the record choice vector
    recq = np.zeros((nit,nch,nep)) #Initialize the record q-value vector
    reca = np.empty((nit,2,nep)) #Initialize the alpha of the agent

    for cit in range(nit):
        if alpha!=[]: #Assign the two fix learning rates
            aR = alpha[0]
            aP = alpha[1]
        else: #Initialize the counting or Reward and Punishment
            nR, nP = 1, 1

        qest = np.zeros((nch,nep)) #Initialize Q-values at 0
        for cep in range(0,nep):
            choice = fch(qest[:,cep]) #Choosing given the Q value
            reward = obs[cit,choice,cep] #The precomputed reward
            if alpha==[]:
                nR += reward==1
                nP += reward==-1
                aR = (cep+1-nR)/float((cep+1)*10)
                aP = (cep+1-nP)/float((cep+1)*10)

            recch[cit,cep] = choice
            recq[cit,:,cep] = qest[:,cep]
            reca[cit,:,cep] = [aR,aP]
            if cep == nep-1: #Updating the q-values if not the last episodes
                break
            qest[:,cep+1] = qest[:,cep] #Keep the same q
            qest[choice,cep+1] = qnext(qest[choice,cep], reward, aR, aP) #Update from the reward and choice

    return recch, recq, reca, obs

#Functions used for formal analysis plots

def qss(p = 0.8, alpha= [0.1,0.5]):
    '''

    Compute the steady state Q value given p and the alphas

    PARAMETERS
    -----
    p: a real number in a list, the probability of reward p = p(Q=1)
    aR: the learning rate for reward
    aP: the learning rate for punishment

    RETURNS
    -----
    qss : the q value at steady state

    '''
    aR, aP = alpha[0], alpha[1]
    qss = ((aR/aP)*p + (p-1)) / ((aR/aP)*p - (p-1))
    return qss

def xopt(p1=0.9, p0=0.8):
    '''

    Compute optimal x ratio between learning rates which leads to the highest Delta Q for a given task

    PARAMETERS
    -----
    p1: a real number, the probability of reward p = p(Q=1) for the best arm
    p0: a real number, the probability of reward for the worst arm

    RETURNS
    -----
    xopt : the q value at steady state
    dopt : the optimal delta Q

    '''
    q1 = 1-p1
    q0 = 1-p0
    xopt = np.sqrt((p1*p0) / (q1*q0))
    dopt = (2*xopt*(p1*q0-p0*q1)) / (xopt**2*p1*p0+xopt*(p1*q0+p0*p1)+q0*q1)
    return xopt, dopt


#Handle function to record data

def h5qname(p=dpvals[0][0], alpha=dagents[0]):
    '''

    Generate the name of the data group to record Q values given pval and alpha

    '''
    gname = "/Qest"
    gsname = "/p" + str(p)
    if alpha==[]:
        dname = "/plas"
    else:
        dname = "/a" + str(alpha)
    return gname + gsname + dname

def h5qrec(q=np.arange(2), p=dpvals[0][0], alpha=dagents[0], fname=dfname):
    '''

    Record the Q values

    '''
    hdf = h5py.File(fname,'a')
    name = h5qname(p, alpha)
    if name not in hdf:
        hdf.create_dataset(name, data=np.array(q))
    hdf.close()

def h5qout(p=dpvals[0][0], alpha=dagents[0], fname=dfname):
    '''

    Extract the Q values

    '''
    hdf = h5py.File(fname,'r')
    name = h5qname(p, alpha)
    q = np.array(hdf[name])
    hdf.close()
    return q


def h5cname(pval=dpvals[0], alpha=dagents[0]):
    '''

    Build the name for the hdf5 group and dataset

    '''
    gname = '/Choices'
    dname = '/p' + str(pval)
    if alpha==[]:
        dname += '_plas'
    else:
        if alpha[0]>alpha[1]:
            dname += '_opt'
        if alpha[0]<alpha[1]:
            dname += '_pes'
        if alpha[0]==alpha[1]:
            dname += '_rat'
    return gname + dname

def h5crec(choice=np.zeros(2), pval=dpvals[0], alpha=dagents[0], fname=dfname):
    '''
    Record a testbed of bandit task in an hdf5 file
    '''
    hdf = h5py.File(fname,'a')
    name = h5cname(pval, alpha)
    if name not in hdf:
        hdf.create_dataset(name, data=choice, dtype=np.int)
    hdf.close()

def h5cout(pval=dpvals[0], alpha=dagents[0], fname=dfname[0]):
    '''
    Return a testbed of bandit tasks from an hdf5 file
    '''
    hdf = h5py.File(fname,'r')
    name = h5cname(pval, alpha)
    t = np.array(hdf[name])
    hdf.close()
    return t

# Wrap functions to run a series of testbed
def testbedwrap(pvals=dpvals, agents=dagents, nep=dnep, nit=dnit, fname=dfname, fch=dfch):
    '''

    Wrap the testbed to and record the data for different environment and agents

    '''
    for pval in pvals: # Iterate through environments
        for agent in agents: # Iterate through agents
            recch, recq, reca, obs = testbed(pval, agent, nep, nit, fch)
            h5crec(recch, pval, agent, fname)
            for i,p in enumerate(pval):
                h5qrec(recq[:,i,:], p, agent, fname)

#Functions for numerical analysis

def pss(chrec=[0,0,1,2,0,2,2]):
    '''

    Determine the probability of switch and staying given a choice record.

    '''
    switch = np.sum(chrec[1:] != chrec[:-1])
    stay = (len(chrec)-1) - switch
    return [switch/float(len(chrec)-1) , stay/float(len(chrec)-1)]



def wrapss(pval=dpvals[0], alpha=dagents[0], fname="opti.hdf5"):
    '''

    Wrap of pss, return staying and switching probabilities for different reward distributions
    and different agent

    '''
    return np.mean(np.array([pss(chrec)[0] for chrec in h5cout(pval, alpha, fname)]),axis=0)

# Plot function to plot the data
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
    Remove the spines and ticks in right and top
    '''
    for loc, spine in ax.spines.iteritems():
        if loc in places :
            spine.set_color('none') # don't draw spine
    #set off ticks where there is no spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

#Each functions correspond to a figure (Need to run simulations before)
def fig1(ps=[0.1,0.2,0.8,0.9], agents=dagents, fname=dfname, fnamef=ddatdirf + 'fig1.svg'):
    '''

    Generate a figure of the final Q estimate of different agent

    Currently BROKEN for plotting

    '''
    mean = [[[np.mean(h5qout([p],agents[0], fname)[:,-1]), np.mean(h5qout([p], agent, fname)[:,-1])] for agent in dagents[1:]] for p in ps]
    var = [[[np.var(h5qout([p],agents[0], fname)[:,-1]), np.var(h5qout([p], agent, fname)[:,-1])] for agent in dagents[1:]] for p in ps]

    mean, var = np.array(mean), np.array(var)

    plt.close('all')
    fig = plt.figure(num=1, figsize=(5,3), dpi=100)

    ax = fig.add_subplot(121)
    aP = 0.1
    for i in range(len(ps)):
        #qa = [qss(p,aR,aP) for aR in [0.1,0.4]]
        plt.plot(mean[i,0], 'o-', color='black')
        plt.errorbar([0,1],mean[i,0],yerr=var[i,0], color='black')
    plt.show()
    ax.set_xlabel(r'$\alpha^+$', size=14)
    ax.set_ylabel(r'$Q_{\infty}$', size=14)
    ax.set_ylim(-1,1, size=10)
    ax.set_xticklabels(np.arange(0.1,0.41,0.3/5),size=10)
    rspines(ax)

    ax = fig.add_subplot(122)
    aP = 0.1
    for i in range(len(ps)):
        #qa = [qss(p,aR,aP) for aR in [0.1,0.4]]
        plt.plot(mean[i,1], 'o-', color='black')
        plt.errorbar([0,1],mean[i,1],yerr=var[i,0], color='black')
    plt.show()
    ax.set_xlabel(r'$\alpha^-$', size=14)
    ax.set_ylim(-1,1, size=10)
    ax.set_xticklabels(np.arange(0.1,0.41,0.3/5),size=10)
    rspines(ax)
    rspines(ax,places=['left','right','top'])

    plt.savefig(fnamef,transparent=True)


def fig2a(pvals=dpvals, agents=dagents, fname=dfname, nep=dnep, fnamef=ddatdirf+'fig2a.svg'):
    '''

    Generate the figure plotting the probability of taking the best option (weird behavior when more than 2 choices)

    '''
    choices = [[np.mean(h5cout(pvals[i], dagents[j],fname),axis=0) for j in range(len(agents))] for i in range(len(pvals))]


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
    #rspines(ax,places=['left','right','top'])

    plt.savefig(fnamef,transparent=True)
    return np.array(choices)

def fig2b(pvals=dpvals, agents=dagents, fname=dfname, fnamef=ddatdirf+'fig2b.svg'):
    '''

    Generate the figure plotting the probability of switch or staying 2a for the NIPS article

    '''
    switch = [[wrapss(pvals[i], dagents[j],fname) for i in range(len(pvals))] for j in range(len(agents))]
    switch = np.array(switch)

    #return switch #For now we use fig2b just for presenting datas

    fig = plt.figure(figsize=(5,1), dpi=300)
    taskname = ("Low reward", "High reward")
    labels = ['R','O','P','N']
    width = 0.80
    ind = np.arange(4)

    ax = fig.add_subplot(121)
    plt.bar(ind+i*width, switch[:,0], width, color=dcolors)
    ax.set_title(taskname[0], size=10)
    ax.set_ylabel('P(switching)', size=10)
    ax.set_ylim(0,1)
    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels(labels, size=10)
    rspines(ax)

    ax = fig.add_subplot(122)
    plt.bar(ind+i*width, switch[:,1], width, color=dcolors)
    ax.set_title(taskname[1], size=10)
    ax.set_ylim(0,1)
    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels(labels, size=10)
    rspines(ax)

    plt.savefig(fnamef,transparent=True)
    return switch

def fig2alt():
    '''
    Generate another figure organized by task and then by probability
    '''
    pval = [0.2,0.15,0.1]
    rat = wrapss(pval, aR=0.1, aP=0.1)
    opt = wrapss(pval, aR=0.4, aP=0.1)
    pes = wrapss(pval, aR=0.1, aP=0.4)
    #nor = wrapss(pval, plas=True)
    pss1 = [tuple(rat),tuple(opt),tuple(pes)]#,tuple(nor)]

    pval = [0.9,0.85,0.8]
    rat = wrapss(pval, aR=0.1, aP=0.1)
    opt = wrapss(pval, aR=0.4, aP=0.1)
    pes = wrapss(pval, aR=0.1, aP=0.4)
    #nor = wrapss(pval, plas=True)
    pss2 = [tuple(rat),tuple(opt),tuple(pes)]#,tuple(nor)]

    fig = plt.figure()
    colr = ['b','g','r']#,'violet']
    width = 0.20
    ind = np.arange(3)

    ax = fig.add_subplot(121,adjustable='box')
    for i, ccolr in enumerate(colr):
        plt.bar(ind+i*width, pss1[i],width, color=ccolr)

    ax.set_ylabel('P')
    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels( ('P(switching)', 'P(staying)') )
    ax.set_xlabel('Task One')

    ax = fig.add_subplot(122,adjustable='box')
    for i, ccolr in enumerate(colr):
        plt.bar(ind+i*width, pss2[i],width, color=ccolr)

    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels( ('P(switching)', 'P(staying)') )
    ax.set_xlabel('Task Two')

def fig4(fname='opti2.h5'):
    '''

    Generate a figure with the different distributions

    '''
    plt.close('all')
    mu = [-1,1]
    clr = ['red','green']
    sigma = [0.5,0.45]
    r = np.arange(-3,3.05,0.1)
    fig = plt.figure()
    for i,cmu in enumerate(mu):
        for csigma in sigma:
            dist = [normpdf(cr,cmu,csigma) for cr in r]
            plt.plot(r,dist,color=clr[i])
    plt.xlim(-3,3)
    plt.xlabel("Reward value")
    plt.ylabel("probability")
