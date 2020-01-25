import numpy as np
import matplotlib.pyplot as plt
from general import calculate_weights, retrieve_async, retrieve_sync


def add_noise(x, noise):
    noise_inds = np.random.permutation(x.size)[:noise]
    x[noise_inds] = x[noise_inds]*-1
    return x


#     2a     #
images = np.load('images.npy')

# Parameters
nrows = 46
ncols = 25
num_iter = 1000
noise_fac = 200
num_images = 7

fig, ax = plt.subplots(nrows=1, ncols=7, sharex='all', sharey='all',
                       figsize=(10, 2),
                       gridspec_kw={'wspace': 0},
                       subplot_kw={'xticks': [], 'yticks': []})

#     2b     #

for im_ind in range(num_images):
    im = images[:, im_ind]
    im_reshape = im.reshape(nrows, ncols, order='F')
    ax[im_ind].imshow(im_reshape, cmap='binary')
    ax[im_ind].set_title('#%d' %im_ind)
fig.savefig('figures/images.pdf', format='pdf')

#     2c    #

w = calculate_weights(images)

#     2d    #

fig, ax = plt.subplots(nrows=3, ncols=7, sharex='all', sharey='all',
                       # figsize=(8, 4),
                       gridspec_kw={'wspace': 0.1, 'hspace': 0.3},
                       subplot_kw={'xticks': [], 'yticks': []})
ax[0, 0].set_ylabel('Original image')
ax[1, 0].set_ylabel('Cued image')
ax[2, 0].set_ylabel('Retrieved image')

for im_ind in range(num_images):
    im = images[:, im_ind]
    ax[0, im_ind].imshow(im.reshape(nrows, ncols, order='F'), cmap='binary')
    im_noise = add_noise(im.copy(), noise_fac)
    ax[1, im_ind].imshow(im_noise.reshape(nrows, ncols, order='F'), cmap='binary')
    im_ret_noise = retrieve_sync(im_noise.copy(), w, num_iter)
    ax[2, im_ind].imshow(im_ret_noise.reshape(nrows, ncols, order='F'), cmap='binary')
    cc = np.corrcoef(im, im_ret_noise)[1, 0]
    ax[2, im_ind].set_title('r = %.2f' %cc, fontsize=10)

plt.tight_layout()
fig.savefig('figures/images-retrieval.pdf', format='pdf')

#     2e    #

cc_im_mat = np.corrcoef(images.T)
fig, ax = plt.subplots()
im_h = ax.imshow(cc_im_mat)
cbar = fig.colorbar(im_h)
ax.set_ylabel('image #')
ax.set_xlabel('image #')
cbar.set_label('Correlation coefficient')
fig.savefig('figures/correlation-matrix.pdf', format='pdf')

#     2f    #

noise_level = np.arange(0, 1175, 25)

cc_noise_im = np.zeros((num_images, noise_level.size))
num_iter=100

for im_ind in range(num_images):
    im = images[:, im_ind]
    for n_ind, noise_fac in enumerate(noise_level):
        print('Processing image #%d noise level = %d' %(im_ind, noise_fac))
        im_noise = add_noise(im.copy(), noise_fac)
        im_ret_noise = retrieve_sync(im_noise.copy(), w, num_iter)
        cc_noise_im[im_ind, n_ind] = np.corrcoef(im, im_ret_noise)[1,0]

fig, ax = plt.subplots()
noise_im_h = ax.pcolor(noise_level, np.arange(num_images+1), cc_noise_im)
cbar = fig.colorbar(noise_im_h)
ax.set_xlabel('noise level')
ax.set_ylabel('image #')
cbar.set_label('Correlation coefficient')
fig.savefig('figures/noiselevel-correlation.pdf', format='pdf')

plt.show()