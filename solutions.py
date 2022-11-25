# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:03:03 2019

@author: Yongwen Yang
"""

import matplotlib.pyplot as plt
import numpy as np

# === codes for (a) === #
print('\nSolution to (a):')

epsilon=0.0001  ##已修改
grid=1000
alpha=0.2
beta=0.99
A=2
##第20行已修改
C=np.log(A*(1-alpha*beta))/(1-beta)+alpha*beta*np.log(A*alpha*beta)\
/((1-alpha*beta)*(1-beta))
D=alpha/(1-alpha*beta)    ##已修改
m=A*beta*D/(1+beta*D)    
k_star=m**(1/(1-alpha))
k0=np.linspace(0.1*k_star,1.9*k_star,grid)  ##已修改
V=lambda k:C+D*np.log(k)    ##已修改
pi=lambda k:m*k**alpha    
figa1,axa1=plt.subplots(num='(a)1')   
axa1.plot(k0,V(k0),'-',lw=2,label='value function')
axa1.set_xlabel('$k_t$',fontsize=14)
axa1.set_ylabel('$V(k_t)$',fontsize=14)
axa1.legend(loc='lower right',frameon=True,fontsize=14)
figa1.savefig('figure_a_1.pdf', format='pdf', dpi=1000)
figa1.show()

figa2,axa2=plt.subplots(num='(a)2')
axa2.plot(k0,pi(k0),'-',lw=2,label='policy function')
axa2.set_xlabel('$k_t$',fontsize=14)
axa2.set_ylabel(r'$\pi(k_t)$',fontsize=14)
axa2.legend(loc='lower right',frameon=True,fontsize=14)
figa2.savefig('figure_a_2.pdf', format='pdf', dpi=1000)
figa2.show()

c_star=A*k_star**alpha-pi(k_star)
print('The steady-state capital stock is %.2f, and the steady-state\
 consumption is %.2f.' % (k_star,c_star))





# === codes for (b) === #
print('\nSolution to (b):')

v0=np.zeros(grid)
v1=np.zeros(grid)
k1=np.zeros(grid)
trial=lambda k0, v0, i: np.log(A*k0[i]**alpha-k0)+beta*v0   ##已修改
for i in range(grid):   
    v1[i]=np.max(trial(k0,v0,i))   
    k1[i]=k0[np.argmax(trial(k0,v0,i))]
figb,axb=plt.subplots(nrows=2,ncols=1,figsize=(8,12),num='(b)')
axb[0].plot(k0,V(k0),'-',lw=2,label='true value function')
axb[1].plot(k0,pi(k0),'-',lw=2,label='true policy function')
axb[0].set_xlabel('$k_t$',fontsize=14)
axb[0].set_ylabel('$V(k_t)$',fontsize=14)
axb[1].set_xlabel('$k_t$',fontsize=14)
axb[1].set_ylabel(r'$\pi(k_t)$',fontsize=14)

while np.max(np.abs(v1-v0))>=epsilon:
    v0=v1.copy()
    for i in range(grid):
        v1[i]=np.max(trial(k0,v0,i))
        k1[i]=k0[np.argmax(trial(k0,v0,i))]
axb[0].plot(k0,v1,'r--',lw=2,label='value function by iteration')
axb[0].legend(loc='lower right',frameon=True,fontsize=14)
axb[1].plot(k0,k1,'r--',lw=2,label='policy function by iteration')
axb[1].legend(loc='lower right',frameon=True,fontsize=14)
figb.savefig('figure_b.pdf', format='pdf', dpi=1000)
figb.show()


# === codes for (c) === #
print('\nSolution to (c):')

beta_c=0.8
v0_c=np.zeros(grid)
v1_c=np.zeros(grid)
k1_c=np.zeros(grid)
def trial_c(k0,v0_c,i):
    return np.log(A*k0[i]**alpha-k0)+beta_c*v0_c  ##已修改
for i in range(grid):
    v1_c[i]=np.max(trial(k0,v0_c,i))
    k1_c[i]=k0[np.argmax(trial(k0,v0_c,i))]
figc,axc=plt.subplots(num='(c)')
axc.set_xlabel('$k_t$',fontsize=14)
axc.set_ylabel(r'$\pi(k_t)$',fontsize=14)

while np.max(np.abs(v1_c-v0_c))>=epsilon:
    v0_c=v1_c.copy()
    for i in range(grid):
        v1_c[i]=np.max(trial_c(k0,v0_c,i))
        k1_c[i]=k0[np.argmax(trial_c(k0,v0_c,i))]
axc.plot(k0,k1_c,'r--',lw=2,label=r'policy function with $\beta=0.8$')
axc.plot(k0,k1,'-',lw=2,label=r'policy function with $\beta=0.99$')
axc.legend(loc='lower right',frameon=True,fontsize=14)
figc.savefig('figure_c.pdf', format='pdf', dpi=1000)
figc.show()

print('When replaced with a lower discount rate, the slope of policy\
 function curve \nturns smaller, which means that people prefer current\
 consumption and leave \nless capital stock to the next period.')




# === codes for (d) === #
print('\nSolution to (d):')

v0_d=np.zeros(grid)
v1_d=np.zeros(grid)
k1_d=np.zeros(grid)
def trial_d(k0,v0_d,i):
    return ((A*k0[i]**alpha-k0)**0.5-1)/0.5+beta*v0_d  ##已修改
for i in range(grid):
    v1_d[i]=np.max(trial_d(k0,v0_d,i))  ##已修改
    k1_d[i]=k0[np.argmax(trial_d(k0,v0_d,i))]
figd,axd=plt.subplots(nrows=2,ncols=1,figsize=(8,12),num="(d)")
axd[0].set_xlabel('$k_t$',fontsize=14)
axd[0].set_ylabel('$V(k_t)$',fontsize=14)
axd[1].set_xlabel('$k_t$',fontsize=14)
axd[1].set_ylabel(r'$\pi(k_t)$',fontsize=14)

while np.max(np.abs(v1_d-v0_d))>=epsilon:
    v0_d=v1_d.copy()
    for i in range(grid):
        v1_d[i]=np.max(trial_d(k0,v0_d,i))
        k1_d[i]=k0[np.argmax(trial_d(k1,v0_d,i))]
axd[0].plot(k0,v1_d,'r--',lw=2,label=r'value function with $\mu(c_t)=\frac{c_t^{0.5}-1}{0.5}$')
axd[0].plot(k0,v1,'-',lw=2,label=r'value function with $\mu(c_t)=\log(c_t)$')
axd[0].legend(loc='lower right',frameon=True,fontsize=14)
axd[1].plot(k0,k1_d,'r--',lw=2,label=r'policy function with $\mu(c_t)=\frac{c_t^{0.5}-1}{0.5}$')
axd[1].plot(k0,k1,'-',lw=2,label=r'policy function with $\mu(c_t)=\log(c_t)$')
axd[1].legend(loc='lower right',frameon=True,fontsize=14)
figd.savefig('figure_d.pdf', format='pdf', dpi=1000)
plt.show(figd)
