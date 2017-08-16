import math as m
import numpy as np
from scipy.signal import fftconvolve

def render_psf_os(psf_sigma_x, psf_sigma_y, psf_theta, pixel_scale, over_sampling):

    psf_sigma_max = max(psf_sigma_x, psf_sigma_y)
    radius_os_pix = 3*int(over_sampling*psf_sigma_max/(2.0*pixel_scale))
    range_os_pix = range(-radius_os_pix, radius_os_pix+1)
    x, y = np.meshgrid(range_os_pix, range_os_pix)
    a = (m.cos(psf_theta)**2)/(2*psf_sigma_x**2) + (m.sin(psf_theta)**2)/(2*psf_sigma_y**2)
    b = m.sin(2*psf_theta)/(4*psf_sigma_y**2) - m.sin(2*psf_theta)/(4*psf_sigma_x**2)
    c = m.sin(psf_theta)**2/(2*psf_sigma_x**2) + m.cos(psf_theta)**2/(2*psf_sigma_y**2)
    psf_os = np.exp(-a*(x**2) - 2*b*(x*y) - c*(y**2))
    psf_os = psf_os / np.sum(psf_os)
    return psf_os

def render_source_image(
    pos_x = 0,
    pos_y = 0,    
    alpha = 5,
    noise_bg_sigma = 0.01,
    psf_sigma_x = 6.0,
    psf_sigma_y = 8.0,
    psf_theta = 0.0,
    width = 25,
    height = 25,
    pixel_scale = 0.5,
    over_sampling = 5):

    assert(over_sampling % 2 == 1)

    half_width_os_pix = int(over_sampling*width/(2.0*pixel_scale))
    half_height_os_pix = int(over_sampling*height/(2.0*pixel_scale))
    width_os_pix = 1 + 2*half_width_os_pix
    height_os_pix = 1 + 2*half_height_os_pix
    pos_x_os_pix = int(round((over_sampling*pos_x/pixel_scale) + half_width_os_pix))
    pos_y_os_pix = int(round((over_sampling*pos_y/pixel_scale) + half_width_os_pix))
    im_os = np.zeros((height_os_pix, width_os_pix))
    im_os[pos_y_os_pix, pos_x_os_pix] = 1.0
    psf_os = render_psf_os(psf_sigma_x, psf_sigma_y, psf_theta, pixel_scale, over_sampling)
    im_os_conv_psf_os = fftconvolve(im_os, psf_os, 'same')
    ds_kernel = np.ones((over_sampling, over_sampling))
    im_os_tag = fftconvolve(im_os_conv_psf_os, ds_kernel, 'same')
    x_sample = np.arange(0, int(half_width_os_pix), over_sampling).astype(np.int32)
    y_sample = np.arange(0, int(half_height_os_pix), over_sampling).astype(np.int32)
    x_sample = np.hstack((-np.flip(x_sample, 0), x_sample[1:])) + half_width_os_pix
    y_sample = np.hstack((-np.flip(y_sample, 0), y_sample[1:])) + half_height_os_pix
    xx, yy = np.meshgrid(x_sample, y_sample)
    im_ds = im_os_tag[yy.flatten(), xx.flatten()].reshape(*xx.shape)
    im = im_ds + np.random.randn(*im_ds.shape)*noise_bg_sigma
    return im

if __name__ == "__main__":

    src_im = render_source_image()
    import matplotlib.pyplot as plt
    plt.imshow(src_im)
    plt.show()