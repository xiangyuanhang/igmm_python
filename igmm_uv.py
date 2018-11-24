import numpy as np
from collections import Counter
from arspy import ars
from scipy.special import multigammaln

#生成最初的样本
sampleNo = 100
mu = 3
sigma = 0.1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo)

def logbetapdf(beta, s, w):
    if beta<0:
        return beta
    else:
        k = len(s)
        h = -k * multigammaln(beta/2)-1
#print(s)

def sample_igmm(Y, Nsamp):
    N = len(Y)
    mu_y = np.mean(Y)
    sigSq_y = np.var(Y)
    sigSqi_y = float(1)/sigSq_y

    #Start off with one class
    c = []
    for i in range(N):
        c.append(0)

    Samp = []
    dic = {}
    Samp.append(dic)
    Samp[0]['k'] = 1
    Samp[0]['mu'] = mu_y
    Samp[0]['s'] = sigSq_y
    Samp[0]['lambdaa'] = np.random.normal(mu_y, sigSq_y, 1)
    Samp[0]['r'] = np.random.gamma(1, sigSqi_y)
    Samp[0]['beta'] = float(1)/np.random.gamma(1,1)
    Samp[0]['w'] = np.random.gamma(1, sigSq_y)
    Samp[0]['alpha'] = float(1)/np.random.gamma(1,1)
    Samp[0]['pi'] = 1.0
    Samp[0]['Ic'] = 1

    for i in range(Nsamp):
        #Make aliases for more readable code
        k = Samp[i]['k']
        mu = Samp[i]['mu']
        s = Samp[i]['s']
        beta = Samp[i]['beta']
        r = Samp[i]['r']
        w = Samp[i]['w']
        lambdaa = Samp[i]['lambdaa']
        alpha = Samp[i]['alpha']

        #find the popularity of the class
        nij = Counter(c)

        #Mu
        for j in range(k):
            inClass = [i for i in range(len(c)) if c[i] == j]
            n = len(inClass)
            if n <= 0:
                ybar = 0
            else:
                ybar = np.mean([Y[i] for i in inClass])

            tmp_sigSq = 1.0/(n*s[j]+r)
            tmp_mu = tmp_sigSq*(n*ybar*s[j]+lambdaa*r)
            Samp[i+1]['mu'].append(np.random.normal(tmp_mu, tmp_sigSq))


        #lambdaa
        tmp_sigSq = 1.0/(sigSqi_y+float(k)*r)
        tmp_mu = tmp_sigSq*(mu_y*sigSqi_y+r*np.sum(mu))
        Samp[i+1]['lambdaa'] = np.random.normal(tmp_mu, tmp_sigSq)

        #R
        mu_subtract_lambdaa = [d-lambdaa for d in mu]
        mu_sub_sq = [d^2 for d in mu_subtract_lambdaa]
        Samp[i+1]['r'] = np.random.gamma(k+1, float(k+1)/(sigSq_y+np.sum(mu_sub_sq)))

        #S
        tmp_a = []
        tmp_b = []
        for j in range(k):
            inClass = [i for i in range(len(c)) if c[i]==j]
            n = len(inClass)
            if n <= 0:
                sbar = 0
            else:
                Y_1 = [Y[i] for i in inClass]
                Y_2 = [(d - mu[j])^2 for d in Y_1]
                sbar = np.sum(Y_2)
            tmp_a.append(beta + nij[j])
            tmp_b.append(tmp_a[j]/(w*beta + sbar))
        Samp[i+1]['s'] = np.random.gamma(tmp_a, tmp_b)

        #W
        Samp[i]['w'] = np.random.gamma(k*beta+1, (k*beta+1)/(sigSqi_y+beta*np.sum(s)))

        #Beta
        Samp[i+1]['beta'] = ars.adaptive_rejection_sampling()
#sample_igmm(s, 2)