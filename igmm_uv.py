
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.special import psi
from collections import Counter
from ars import ARS
from scipy.special import multigammaln

#生成最初的样本
Nsamp = 5000
sampleNo = 50
mu0 = 1000
sigma0 = 10
mu1 = 100
sigma1 = 10
mu2 = -100
sigma2 = 10
s0 = np.random.normal(mu0, sigma0, sampleNo)
s1 = np.random.normal(mu1, sigma1, sampleNo)
s2 = np.random.normal(mu2, sigma2, sampleNo)
s = np.hstack((s0,s1,s2))

# define the functions
def drawGammaRas(gamma1, gamma2):
    return np.random.gamma(np.multiply(gamma1,0.5), np.multiply(np.divide(gamma2, gamma1),2))
def renumber(cn, c, k, Ic, N):
    '''
    :return: (c, keep, to_add, Ic)  这里的keep要保留的mu的个数
    '''
    #print(k)
    c = np.array(c)
    Ic = int(Ic)
    #print(Ic)
    #print(cn.shape[0])
    #print(k)
    #print(cn[Ic:cn.shape[0]])
    Icn = np.argwhere(cn[Ic:cn.shape[0]]==k)+Ic #???? 应该是Ic还是Ic-1呢？应该是Ic因为在python里，Ic初始化为0
    if Icn.size != 0:
        Icn = Icn[0]
        c[Ic:int(Icn)+1] = cn[Ic:int(Icn)+1] #这个地方有问题。。。。
        Ic = Icn+1
        to_add = Icn
    else:
        c[Ic:N] = cn[Ic:N]
        to_add = np.array([])
        Ic = 0
    tab = Counter(c)
    key_after = tab.keys()
    key_after = list(key_after)
    keep = list(key_after)
    #key_before = range(max(key_after)+1)
    #key_out = key_before - key_after
    if Icn.size != 0:
        keep.remove(max(keep))
    #c_temp = np.array([])
    #np.copy(c_temp, c)
    #c_temp = np.zeros((1,len(c)))
    i = 0
    c_temp = np.copy(c)
    for key in tab:
        index = np.argwhere(c_temp==key)
        c[index] = i
        i += 1
    keep = np.array(keep)
    return (c,keep,to_add,Ic)
def drawMultinom(a):
    n = a.shape[1]
    d = np.zeros(n)
    for i in range(n):
        a_temp = a[:, i]
        temp = np.random.multinomial(1, (a_temp) / np.sum(a_temp))
        d[i] = np.argwhere(temp == 1)
    return d

def normalLike(y, mu, s):
    y = np.array(y)
    mu = np.array(mu)
    s = np.array(s)
    return np.multiply(np.sqrt(np.divide(s,2*np.pi)),np.exp(np.multiply(-s,np.divide(np.power(np.subtract(y,mu),2),2))))

def fbeta(x, s=[0.5,1.5], w=1.5):
    '''
    logbetapdf
    '''
    k = len(s)
    return -k * multigammaln(x / 2, 1) - 1 / (2 * x) + \
    (k * x - 3) / 2 * np.log(x / 2) + \
    (x / 2) * np.sum(np.add(np.log(w), np.log(s))) - x * np.sum(np.multiply(s, w / 2))

def fbetaprima(x, s=[0.5,1.5],w=1.5):
    '''
    logbetapdfprime
    '''
    k = len(s)
    return -k/2*psi(x/2)+1/(2*(x**2))+k/2*np.log(x/2)+(k*x-3)/(2*x)+\
    np.sum(np.subtract(np.add(np.log(s),np.log(w)),np.multiply(s,w)))/2

def falpha(alpha, k, n):
    return (k-3/2)*np.log(alpha)-1/(2*alpha)*multigammaln(alpha,1)-multigammaln(n+alpha,1)

def falphaprima(alpha, k, n):
    return (k-3/2)/alpha+1/(2*(alpha**2))*multigammaln(alpha,1)-1/(2*alpha)*psi(alpha)-psi(n+alpha)

def drawBeta(s, w, size=1):
    """Draw beta from its distribution (Eq.9 Rasmussen 2000) using ARS
    Make it robust with an expanding range in case of failure"""
    #nd = w.shape[0] 用于多维数据
    lb = 0.0
    flag = True
    cnt = 0
    while flag:
        xi = lb + np.logspace(-3 - cnt, 1 + cnt, 200)  # update range if needed
        flag = False
        try:
            ars = ARS(fbeta, fbetaprima, xi=xi, lb=0.0, ub=np.inf, \
                      s=s, w=w)
        except:
            cnt += 1
            flag = True

    # draw beta
    return ars.draw(size)


def drawAlpha(k, N, size=1):
    """Draw alpha from its distribution (Eq.15 Rasmussen 2000) using ARS
    Make it robust with an expanding range in case of failure"""
    flag = True
    cnt = 0
    while flag:
        xi = np.logspace(-2 - cnt, 3 + cnt, 200)  # update range if needed
        try:
            ars = ARS(falpha, falphaprima, xi=xi, lb=0, ub=np.inf, k=k, n=N)
            flag = False
        except:
            cnt += 1

    # draw alpha
    return ars.draw(size)

Y=s

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
Samp[0]['mu'] = [mu_y]
Samp[0]['s'] = [sigSqi_y]
Samp[0]['lambdaa'] = np.random.normal(mu_y, np.sqrt(sigSq_y), 1)
Samp[0]['r'] = drawGammaRas(1, sigSqi_y)
Samp[0]['beta'] = float(1)/np.random.gamma(1,1)
Samp[0]['w'] = drawGammaRas(1, sigSq_y)
Samp[0]['alpha'] = float(1)/np.random.gamma(1,1)
Samp[0]['Ic'] = 1

Ic = 0

for i in range(Nsamp):
    #Make aliases for more readable code
    dic = {}
    Samp.append(dic)
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
    Samp[i + 1]['mu'] = []
    for j in range(k):
        inClass = [i for i in range(len(c)) if c[i] == j]
        n = len(inClass)
        if n <= 0:
            ybar = 0
        else:
            ybar = np.mean([Y[i] for i in inClass])
        #print(k)
        #print(n*s[j]+r)
        tmp_sigSq = 1.0/(n*s[j]+r)
        tmp_mu = tmp_sigSq*(n*ybar*s[j]+lambdaa*r)
        mu_add = float(np.random.normal(tmp_mu, np.sqrt(tmp_sigSq)))
        Samp[i + 1]['mu'].append(mu_add)

    #lambdaa
    tmp_sigSq = 1.0/(sigSqi_y+float(k)*r)
    tmp_mu = tmp_sigSq*(mu_y*sigSqi_y+r*np.sum(mu))
    Samp[i+1]['lambdaa'] = np.random.normal(tmp_mu, np.sqrt(tmp_sigSq))

    #R
    mu_subtract_lambdaa = [d-lambdaa for d in mu]
    mu_sub_sq = [np.power(d,2) for d in mu_subtract_lambdaa]
    gamma1 = k+1
    gamma2 = float(k+1)/(sigSq_y+np.sum(mu_sub_sq))
    Samp[i+1]['r'] = drawGammaRas(gamma1, gamma2)

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
            Y_2 = [np.power(d - mu[j],2) for d in Y_1]
            sbar = np.sum(Y_2)
        tmp_a.append(beta + nij[j])
        tmp_b.append(tmp_a[j]/(w*beta + sbar))
    tmp_a = np.array(tmp_a)
    tmp_b = np.array(tmp_b)
    s_array_tmp = drawGammaRas(tmp_a, tmp_b)
    Samp[i + 1]['s'] = []
    for m in range(len(s_array_tmp)):
        Samp[i + 1]['s'].append(float(s_array_tmp[m]))

    #W
    Samp[i+1]['w'] = drawGammaRas((k*beta+1), (k*beta+1)/(sigSqi_y+beta*np.sum(s)))

    # Alpha
    ars_alpha = drawAlpha(k, n, 1)
    Samp[i + 1]['alpha'] = ars_alpha

    #Beta
    ars_beta = drawBeta(s, w, 1)
    Samp[i+1]['beta'] = ars_beta


    #C
    #samples from priors, which could be swapped in if we need a new class. Only needed for i>=Ic
    mu_prop = np.hstack((np.zeros(Ic),np.random.normal(lambdaa, np.sqrt(1/r), N-Ic)))
    s_prop = np.hstack((np.ones(Ic), drawGammaRas(beta*np.ones(N-Ic),1/w)))

    #Find the likelihoods of the observations under the *new* gaussians for i>=Ic
    unrep_like = np.multiply(alpha/(N-1+alpha),normalLike(Y, mu_prop, s_prop))
    #print('un:',unrep_like.shape)
    mu = Samp[i+1]['mu']
    s = Samp[i+1]['s']
    rep_like = np.zeros((k,N))
    #print('k:', k)
    for j in range(k):
        rep_like[j,:]=normalLike(Y, mu[j], s[j])

    #calculate the priors, specific to each datapoint, because counting over everyone else
    nij_temp = [nij[i] for i in sorted(nij.keys())]
    pri = np.matlib.repmat(np.divide(np.array(nij_temp), N - 1 + alpha), N, 1)
    pri = pri.T
    pri[np.array(c),np.array(range(N))] = np.subtract(pri[np.array(c),np.array(range(N))],1/(N-1+alpha))
    #print('pri:', pri.shape)
    #print('rep:', rep_like.shape)
    q = np.vstack((np.multiply(pri, rep_like), unrep_like))
    #print('total:',q.shape)
    cn = drawMultinom(q)
    temp = renumber(cn,c,k,Ic,N)
    c = temp[0]
    keep = temp[1]
    to_add = temp[2]
    Ic = temp[3]
    mu_array = np.array(Samp[i+1]['mu'])
    s_array = np.array(Samp[i + 1]['s'])
    #print(keep, type(keep))
    #print(to_add, type(to_add))
    if to_add.size != 0:
        mu_array_mix = np.hstack((mu_array[keep],mu_prop[to_add]))
        Samp[i+1]['mu'] = mu_array_mix
        s_array_mix = np.hstack((s_array[keep], s_prop[to_add]))
        Samp[i+1]['s'] = s_array_mix
    else:
        Samp[i + 1]['mu'] = mu_array[keep]
        Samp[i + 1]['s'] = s_array[keep]
    #Samp[i+1]['mu'].append(float(mu_prop[int(to_add)]))
    #Samp[i+1]['s'].append(float(s_prop[int(to_add)]))
    Samp[i+1]['k'] = len(Samp[i+1]['mu'])
    print(Samp[i+1]['mu'])
    print(Samp[i+1]['s'])
    #print(Counter(c))
    print("the "+str(i+1)+"th loop has completed")
    print("*" * 50)

    #update Ic
    Samp[i+1]['Ic']=Ic
    #print(Samp[i+1])

#plot k
k_list = []
for j in range(Nsamp):
    k_list.append(Samp[j]['k'])
k_array = np.array(k_list)
t_array = np.array(range(Nsamp))
fig, ax = plt.subplots()
ax.plot(t_array, k_array)
ax.set(xlabel = 'loop time', ylabel = 'k', title = 'the variation of k over loop time')
ax.grid()
fig.savefig("test.png")
plt.show()

#plot mu
mu_list = []
for j in range(Nsamp):
    mu_list.append(Samp[j]['mu'])
mu_array1 = np.array(mu_array1)
