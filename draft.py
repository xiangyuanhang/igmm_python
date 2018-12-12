from scipy.special import multigammaln
from scipy.special import psi
import numpy as np
import numpy.matlib
from ars import ARS
from collections import Counter

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
'''
logpdf_only_x = lambda x, s_here=[0.5,1.5], w_here=1.5, k=2: \
    -k * multigammaln(x / 2, 1) - 1 / (2 * x) + \
    (k * x - 3) / 2 * np.log(x / 2) + \
    (x / 2) * np.sum(np.add(np.log(w_here), np.log(s_here))) - x * np.sum(np.multiply(s_here, w_here / 2))
logpdf_only_x_prime = lambda x, s_here=[0.5,1.5], w_here=1.5, k=2:\
    -k/2*psi(x/2)+1/(2*(x**2))+k/2*np.log(x/2)+(k*x-3)/(2*x)+\
    np.sum(np.subtract(np.add(np.log(s_here)+np.log(w_here)),np.multiply(s_here,w_here)))/2

a, b = 0.1, 4
domain = (float(0), float("inf"))

ars = ARS(fbeta, fbetaprima, xi=[0.1,4], s=[0.5,1.5], w=1.5)
samples = ars.draw(100)
print(samples)
'''
N = 6
Ic = 0
c = np.array([0,0,0,0,0,0])
cn = np.array([0,0,2,0,0,1])
k = 2

def renumber(cn, c, k, Ic):
    '''
    :return: (c, keep, to_add, Ic)  这里的keep要保留的mu的个数
    '''
    Icn = np.argwhere(cn[Ic:cn.shape[0]]==k)+Ic #???? 应该是Ic还是Ic-1呢？应该是Ic因为在python里，Ic初始化为0
    if Icn.any():
        Icn = Icn[0]
        c[Ic:int(Icn)+1] = cn[Ic:int(Icn)+1] #这个地方有问题。。。。
        Ic = Icn+1
        to_add = Icn
    else:
        c[Ic:N] = cn[Ic:N]
        to_add = []
        Ic = 0
    tab = Counter(c)
    keep = len(tab)
    if Icn:
        keep = keep-1
    i = 0
    #c_temp = np.array([])
    #np.copy(c_temp, c)
    #c_temp = np.zeros((1,len(c)))
    c_temp = np.copy(c)
    for key in tab:
        index = np.argwhere(c_temp==key)
        c[index] = i
        i += 1
    return (c,keep,to_add,Ic)

a = np.array([0])
if a.size != 0:
    print(1)
