# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:02:28 2017

@author: Adam
"""
#%%
from numpy import fft
import numpy as np
from scipy.signal import fftconvolve

#%%
'''
# Generate images and PSFs
pr,pn,n,r = np.zeros((128,128)),np.zeros((128,128)),np.zeros((128,128)),np.zeros((128,128))

sign,sigr = 1e-3,1e-3

x,y = np.meshgrid(np.arange(128),np.arange(128),indexing='ij')
pn+= np.exp(-((x-64)**2+(y-64)**2)/2/10**2)
pr+= np.exp(-((x-64)**2+(y-64)**2)/2/10**2)

pn*=np.sum(pn.flatten())**-1
pr*=np.sum(pr.flatten())**-1

delt1,delt2 = np.zeros_like(n),np.zeros_like(n)
delt1[55,55]=1
delt2[55,65]=1

n = fftconvolve(pn,delt1,'same') + np.random.normal(scale=sign,size=n.shape)
r = fftconvolve(pr,delt2,'same') + np.random.normal(scale=sigr,size=n.shape)

'''

#%%

def z_hat(n_hat,r_hat,pn_hat,pr_hat,sig_n,sig_r):
    '''
    Return the FFT of z given the FFTs of N,R,Pn,Pr and variances of the noise
    '''
    denom = sig_n**2*np.abs(pr_hat)**2 + sig_r**2*np.abs(pn_hat)**2
    return (pr_hat*n_hat-pn_hat*r_hat)*np.conj(pr_hat*pn_hat)/denom


def z(n,r,pn,pr,sig_n,sig_r):
    '''
    Return array of [zx,zy]
    The array is NOT fftshifted
    '''
    pr_hat,pn_hat,n_hat,r_hat = map(fft.fft2,[pr,pn,n,r])
    
    kx,ky = np.meshgrid(np.arange(n.shape[0]),np.arange(n.shape[1]))
    
    zh = z_hat(n_hat,r_hat,pn_hat,pr_hat,sig_n,sig_r)
    
    return np.real(fft.ifft2(np.array([zh*1j*kx,zh*1j*ky])))


#%%

