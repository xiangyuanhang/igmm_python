import numpy as np
from collections import Counter

#生成最初的样本
sampleNo = 100
mu = 3
sigma = 0.1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo)

#print(s)

def sample_igmm(Y, Nsamp):
    N = len(Y)
    mu_y = np.mean(Y)
    sigSq_y = np.var(Y)
    sigSqi_y = float(1)/sigSq_y

    #Start off with one class
    c = []
    for i in range(N):
        c.append(1)

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

            tmp_sigSq = 1.0/(n*)



#sample_igmm(s, 2)