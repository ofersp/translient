import numpy as np
import math as ma    
from numpy.fft import fft2, ifft2, fftshift


def gaussian_profile(x0, y0, sz, cxx, cyy, cth):

    x, y = np.meshgrid(np.arange(sz[1]) - x0, np.arange(sz[0]) - y0)
    a = ma.cos(cth)**2 / (2*cxx**2) + ma.sin(cth)**2 / (2*cyy**2)
    b = ma.sin(2*cth) / (4*cyy**2) - ma.sin(2*cth) / (4*cxx**2)
    c = ma.sin(cth)**2 / (2*cxx**2) + ma.cos(cth)**2 / (2*cyy**2)
    invdet = a*c - b**2
    return np.exp(-0.5*(a*(x**2) + 2*b*(x*y) + c*(y**2))) * (ma.sqrt(invdet) / (2*ma.pi))


def z_statistic(R, N, P_r, P_n, sig_r, sig_n):

    epsilon = 1e-15
    n = np.shape(R)[0]
    sz_diff = np.array(N.shape) - np.array(P_n.shape)
        
    assert(R.shape[0] == R.shape[1])
    assert(R.shape == N.shape)    
    assert(P_r.shape == P_n.shape)
    assert(np.all(sz_diff >= 0))
    assert(np.all((sz_diff % 2) == 0))
    
    half_pad = int(sz_diff[0] / 2)
    P_r = np.pad(P_r, half_pad, 'constant')
    P_n = np.pad(P_n, half_pad, 'constant')

    R_hat, N_hat, P_r_hat, P_n_hat = map(np.fft.fft2,[R, N, P_r, P_n])
    k_x, k_y = np.meshgrid(np.arange(N.shape[0]), np.arange(N.shape[1]))
    
    denom = sig_n**2*np.abs(P_r_hat)**2 + sig_r**2*np.abs(P_n_hat)**2
    num = (P_r_hat*N_hat - P_n_hat*R_hat)*np.conj(P_r_hat * P_n_hat)
    num_over_denom = num / (denom + epsilon)
    z_hat_x = (4*ma.pi/n) * k_x * num_over_denom
    z_hat_y = (4*ma.pi/n) * k_y * num_over_denom

    z_x = np.fft.fftshift(np.imag(np.fft.ifft2(z_hat_x)))
    z_y = np.fft.fftshift(np.imag(np.fft.ifft2(z_hat_y)))
    
    return z_x, z_y


def s_statistic(R, N, P_r, P_n, sig_r, sig_n):

    epsilon = 1e-15
    n = np.shape(R)[0]
    sz_diff = np.array(N.shape) - np.array(P_n.shape)
        
    assert(R.shape[0] == R.shape[1])
    assert(R.shape == N.shape)    
    assert(P_r.shape == P_n.shape)
    assert(np.all(sz_diff >= 0))
    assert(np.all((sz_diff % 2) == 0))
    
    half_pad = int(sz_diff[0] / 2)
    P_r = np.pad(P_r, half_pad, 'constant')
    P_n = np.pad(P_n, half_pad, 'constant')

    R_hat, N_hat, P_r_hat, P_n_hat = map(np.fft.fft2,[R, N, P_r, P_n])
    k_x, k_y = np.meshgrid(np.arange(N.shape[0]), np.arange(N.shape[1]))
    
    denom = sig_n**2*np.abs(P_r_hat)**2 + sig_r**2*np.abs(P_n_hat)**2
    num = (P_r_hat*N_hat - P_n_hat*R_hat)*np.conj(P_r_hat * P_n_hat)
    num_over_denom = num / (denom + epsilon)
    s_hat = num_over_denom
    s = np.real(np.fft.fftshift(np.fft.ifft2(s_hat)))
    
    return s


def delta_hat(q_, sz):
    n = sz[0]
    assert(sz[1] == n)
    x, y = np.meshgrid(range(n), range(n))
    delta = np.exp(-((x-q_[0])**2 + (y-q_[1])**2) / 1.0)
    delta = delta / np.sum(delta)
    return fft2(delta)


def d_statistic_nll_linearized(R_hat, N_hat, P_r_hat, P_n_hat, sig_r, sig_n, q, amb, apb_Delta, theta):
    
    epsilon = 1e-15
    sz = R_hat.shape
    n = sz[0]
    assert(N_hat.shape == sz and P_r_hat.shape == sz and P_n_hat.shape == sz and sz[1] == n)

    delta_hat_q = delta_hat(q, sz)    
    dq = 1e-8
    dqx = np.array([dq, 0])
    dqy = np.array([0, dq])
    d_delta_hat_q_dqx = (delta_hat(q+dqx, sz) - delta_hat(q-dqx, sz)) / (2*dq)
    d_delta_hat_q_dqy = (delta_hat(q+dqy, sz) - delta_hat(q-dqy, sz)) / (2*dq)
    
    num = P_n_hat*R_hat - P_r_hat*N_hat - \
          P_r_hat*P_n_hat*(delta_hat_q*amb - \
          (apb_Delta/2)*ma.cos(theta)*d_delta_hat_q_dqx + \
          (apb_Delta/2)*ma.sin(theta)*d_delta_hat_q_dqy)
    num = np.abs(num)**2                   
    denom = np.abs(sig_r*P_n_hat)**2 + np.abs(sig_n*P_r_hat)**2
    
    nll_per_k = num / (denom + epsilon)
    nll = np.sum(nll_per_k / (2*n**2)) + 0.5*np.sum(np.log(2*ma.pi*denom*n**2))
    return nll

"""
def d_statistic_nll_linearized_(R_hat, N_hat, P_r_hat, P_n_hat, sig_r, sig_n, q, amb, apb_Delta, theta):
    
    epsilon = 1e-15
    sz = R_hat.shape
    n = sz[0]
    assert(N_hat.shape == sz and P_r_hat.shape == sz and P_n_hat.shape == sz and sz[1] == n)

    k_x, k_y = np.meshgrid(np.arange(n), np.arange(n))

    alpha_n_delta_q_alpha_r_delta_p = np.exp(-2*ma.pi*1j*(k_x*q[0] + k_y*q[1])/n) * \
        (amb + apb_Delta*ma.pi*1j*(k_x*ma.cos(theta) + k_y*ma.sin(theta))/n)

    num = P_r_hat*N_hat - P_n_hat*R_hat - P_r_hat*P_n_hat*alpha_n_delta_q_alpha_r_delta_p
    num = np.abs(num)**2                   
    denom = np.abs(sig_r*P_n_hat)**2 + np.abs(sig_n*P_r_hat)**2
    
    nll_per_k = num / (denom + epsilon)
    nll = np.sum(nll_per_k / n**2)
    return nll
"""

def d_statistic_nll_full(R_hat, N_hat, P_r_hat, P_n_hat, sig_r, sig_n, xx, Delta_x, Delta_y, alpha_r, alpha_n):
    
    epsilon = 1e-15
    sz = R_hat.shape
    n = sz[0]
    assert(N_hat.shape == sz and P_r_hat.shape == sz and P_n_hat.shape == sz and sz[1] == n)

    delta_hat_q = delta_hat(xx - np.array([Delta_x, Delta_y])/2, sz)
    delta_hat_p = delta_hat(xx + np.array([Delta_x, Delta_y])/2, sz)
    
    num = P_r_hat*N_hat - P_n_hat*R_hat + \
          P_n_hat*P_r_hat*(alpha_r*delta_hat_q - alpha_n*delta_hat_p)
    num = np.abs(num)**2                   
    denom = np.abs(sig_r*P_n_hat)**2 + np.abs(sig_n*P_r_hat)**2
    
    nll_per_k = num / (denom + epsilon)
    nll = np.sum(nll_per_k / (2*n**2)) + 0.5*np.sum(np.log(2*ma.pi*denom*n**2))
    return nll


def performance_sims(P_r, P_n, T_P_r, T_P_n, sig_r, sig_n, k):
    
    n = len(T_P_r)
    z_pos = np.zeros((n, n, k))
    z_neg = np.zeros((n, n, k))
    s_pos = np.zeros((n, n, k))
    s_neg = np.zeros((n, n, k))

    for i in range(k):    
        R = T_P_r + np.random.randn(n, n)*sig_r
        N = T_P_n + np.random.randn(n, n)*sig_n
        z_x, z_y = z_statistic(R, N, P_r, P_n, sig_r, sig_n)
        s = s_statistic(R, N, P_r, P_n, sig_r, sig_n)    
        z_pos[:, :, i] = z_x**2 + z_y**2
        s_pos[:, :, i] = s**2

    for i in range(k):    
        R = np.random.randn(n, n)*sig_r
        N = np.random.randn(n, n)*sig_n
        z_x, z_y = z_statistic(R, N, P_r, P_n, sig_r, sig_n)
        s = s_statistic(R, N, P_r, P_n, sig_r, sig_n)    
        z_neg[:, :, i] = z_x**2 + z_y**2
        s_neg[:, :, i] = s**2
        
    return z_pos, z_neg, s_pos, s_neg


def roc_curves(z_pos, z_neg, s_pos, s_neg, z_max=4e6, s_max=8e5, roc_bins=10000):

    k = z_pos.shape[2]
    z_tpr = np.zeros(roc_bins,)
    z_fpr = np.zeros(roc_bins,)

    max_z_pos = np.max(z_pos, axis=(0,1))
    max_z_neg = np.max(z_neg, axis=(0,1))                   
    for i, thresh in enumerate(np.linspace(0, z_max, roc_bins)):
        z_tpr[i] = np.sum(max_z_pos > thresh) / k
        z_fpr[i] = np.sum(max_z_neg > thresh) / k

    s_tpr = np.zeros(roc_bins,)
    s_fpr = np.zeros(roc_bins,)

    max_s_pos = np.max(s_pos, axis=(0,1))
    max_s_neg = np.max(s_neg, axis=(0,1))                   
    for i, thresh in enumerate(np.linspace(0, s_max, roc_bins)):
        s_tpr[i] = np.sum(max_s_pos > thresh) / k
        s_fpr[i] = np.sum(max_s_neg > thresh) / k
        
    return z_tpr, z_fpr, s_tpr, s_fpr