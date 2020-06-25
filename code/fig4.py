import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Function to reconstruct the tables with the acquisition information
from dipy.core.gradients import gradient_table

# Functions to sample the diffusion-weighted gradient directions
from dipy.core.sphere import disperse_charges, HemiSphere

# procedures to process diffusion tensor
import dipy.reconst.dti as dti

# procedures to process diffusion tensor
import dipy.reconst.fwdti as fwdti

# fractional anisotropy and mean diffusivity functions
from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity

# Function to simulate a fiber phamtom immersed in wate
from functions import generate_phantom, dual_tensor

# functions to perform standard DTI
import dipy.reconst.dti as dti

# Manifold class
from manifold import Manifold

# Beltrami Model and Fit classes
from beltrami import  BeltramiModel


# ------------------------------------------------------------------------------
print('Parameters: ')
# ------------------------------------------------------------------------------

# Parameters that define the number of simulations and acquisition scheme
n_pts = 32      # number of gradient directions per scheme
n_bzeros = 6    # number of simulated bzeros
bval_low = 0.5  # weigthing factor of lower shell
bval_high = 1   # weighting factor for higher shell
SNR = 40
Fsigma=None          


# ------------------------------------------------------------------------------
# Simulating S0 signal
# ------------------------------------------------------------------------------

# Parameters that define the unweighted signal at 3T (assuming no T1 relaxation)
# T2 relaxation (ms)
T2_tissue = 80  # Wansapura et al., 1999
T2_water = 500  # Piechnik et al., 2009

# Proton density (percentage units)
PD_tissue = 70  # Abbas et al., 2015
PD_water = 100  # proton density of free water is always 100

# Echo time of a single-shell acquisition at 3T
TE = 74

# Assuming no T1 relaxation, the non-weighted signal for a voxel that has only
# tisssue or only water:
k = 10  # scaling factor
St = k * PD_tissue * np.exp(-TE / T2_tissue)
Sw = k * PD_water * np.exp(-TE / T2_water)

print('S0 signal for a voxel with only tissue is ' + str(St))
print('S0 signal for a voxel with only water is ' + str(Sw))
print(('S0 signal for voxel with ' + str(0.5*100) + ' % tissue: ' +
       str(St * 0.5 + Sw * 0.5)))


# ------------------------------------------------------------------------------
# Simulating acquisition gradients
# ------------------------------------------------------------------------------

# Sample the spherical cordinates of 32 evenly distributed
# diffusion-weighted directions
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
directions = hsph_updated.vertices

bvals = bval_low * np.ones(n_pts + n_bzeros)
bvals[0:n_bzeros] = 0
bvecs = np.zeros((n_pts + n_bzeros, 3))
bvecs[n_bzeros:, :] = directions
gtab = gradient_table(bvals, bvecs, b0_threshold=0)

bvals_2 = bval_high * np.ones(n_pts)
bvecs_2 = directions
bvals = np.hstack((bvals, bvals_2))
bvecs = np.vstack((bvecs, bvecs_2))
gtab = gradient_table(bvals, bvecs, b0_threshold=0)


# ------------------------------------------------------------------------------
print('Generating phantom...')
# ------------------------------------------------------------------------------

evals, evecs, F, phantom, mask1 = generate_phantom(gtab, snr=SNR, S0t=St, S0w=Sw, 
                                                  dir_sigma=Fsigma)

# ground truth
gt_fw = (1 - F) * mask1
gt_fa = fractional_anisotropy(evals) * mask1
gt_md = mean_diffusivity(evals) * mask1

print('Data shape: ' + str(phantom.shape))

# ------------------------------------------------------------------------------
print('Running standard DTI...')
# ------------------------------------------------------------------------------
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(phantom)
dti_fa = tenfit.fa * mask1
dti_md = tenfit.md * mask1
dti_fw = np.zeros(dti_fa.shape)


fw_means = np.array([])
fa_means = np.array([])
md_means = np.array([])
fw_lows = np.array([])
fa_lows = np.array([])
md_lows = np.array([])
fw_highs = np.array([])
fa_highs = np.array([])
md_highs = np.array([])


# ------------------------------------------------------------------------------
print('Running Beltrami (S0 init)...')
# ------------------------------------------------------------------------------
bmodel = BeltramiModel(gtab, init_method='S0', Stissue=St, Swater=Sw,
                       iterations=200, learning_rate=0.0005)
bfit = bmodel.fit(phantom, mask=mask1)
belt_fw = bfit.fw * mask1
belt_fa = bfit.fa * mask1
belt_md = bfit.md * mask1
belt_fw0 = bfit.fw0 * mask1
belt_fa0 = bfit.fa0 * mask1
belt_md0 = bfit.md0 * mask1

mask = F == 0.6

# Final error
median_fw = np.median(belt_fw[mask])
p25_fw, p75_fw = np.percentile(belt_fw[mask], [25, 75])
low_fw = median_fw - p25_fw
high_fw = p75_fw - median_fw

median_fa = np.median(belt_fa[mask])
p25_fa, p75_fa = np.percentile(belt_fa[mask], [25, 75])
low_fa = median_fa - p25_fa
high_fa = p75_fa - median_fa

median_md = np.median(belt_md[mask])
p25_md, p75_md = np.percentile(belt_md[mask], [25, 75])
low_md = median_md - p25_md
high_md = p75_md - median_md

# Initial error
median_fw0 = np.median(belt_fw0[mask])
p25_fw0, p75_fw0 = np.percentile(belt_fw0[mask], [25, 75])
low_fw0 = median_fw0 - p25_fw0
high_fw0 = p75_fw0 - median_fw0

median_fa0 = np.median(belt_fa0[mask])
p25_fa0, p75_fa0 = np.percentile(belt_fa0[mask], [25, 75])
low_fa0 = median_fa0 - p25_fa0
high_fa0 = p75_fa0 - median_fa0

median_md0 = np.median(belt_md0[mask])
p25_md0, p75_md0 = np.percentile(belt_md0[mask], [25, 75])
low_md0 = median_md0 - p25_md0
high_md0 = p75_md0 - median_md0

#--------------------------------------------
fw_means = np.append(fw_means, median_fw0)
fw_means = np.append(fw_means, median_fw)
fa_means = np.append(fa_means, median_fa0)
fa_means = np.append(fa_means, median_fa)
md_means = np.append(md_means, median_md0)
md_means = np.append(md_means, median_md)

fw_lows = np.append(fw_lows, low_fw0)
fw_lows = np.append(fw_lows, low_fw)
fa_lows = np.append(fa_lows, low_fa0)
fa_lows = np.append(fa_lows, low_fa)
md_lows = np.append(md_lows, low_md0)
md_lows = np.append(md_lows, low_md)

fw_highs = np.append(fw_highs, high_fw0)
fw_highs = np.append(fw_highs, high_fw)
fa_highs = np.append(fa_highs, high_fa0)
fa_highs = np.append(fa_highs, high_fa)
md_highs = np.append(md_highs, high_md0)
md_highs = np.append(md_highs, high_md)

# ------------------------------------------------------------------------------
print('Running Beltrami (MD init)...')
# ------------------------------------------------------------------------------
bmodel = BeltramiModel(gtab, init_method='MD',
                       iterations=200, learning_rate=0.0005)
bfit = bmodel.fit(phantom, mask=mask1)
belt_fw = bfit.fw * mask1
belt_fa = bfit.fa * mask1
belt_md = bfit.md * mask1
belt_fw0 = bfit.fw0 * mask1
belt_fa0 = bfit.fa0 * mask1
belt_md0 = bfit.md0 * mask1

# Final error
median_fw = np.median(belt_fw[mask])
p25_fw, p75_fw = np.percentile(belt_fw[mask], [25, 75])
low_fw = median_fw - p25_fw
high_fw = p75_fw - median_fw

median_fa = np.median(belt_fa[mask])
p25_fa, p75_fa = np.percentile(belt_fa[mask], [25, 75])
low_fa = median_fa - p25_fa
high_fa = p75_fa - median_fa

median_md = np.median(belt_md[mask])
p25_md, p75_md = np.percentile(belt_md[mask], [25, 75])
low_md = median_md - p25_md
high_md = p75_md - median_md

# Initial error
median_fw0 = np.median(belt_fw0[mask])
p25_fw0, p75_fw0 = np.percentile(belt_fw0[mask], [25, 75])
low_fw0 = median_fw0 - p25_fw0
high_fw0 = p75_fw0 - median_fw0

median_fa0 = np.median(belt_fa0[mask])
p25_fa0, p75_fa0 = np.percentile(belt_fa0[mask], [25, 75])
low_fa0 = median_fa0 - p25_fa0
high_fa0 = p75_fa0 - median_fa0

median_md0 = np.median(belt_md0[mask])
p25_md0, p75_md0 = np.percentile(belt_md0[mask], [25, 75])
low_md0 = median_md0 - p25_md0
high_md0 = p75_md0 - median_md0

#--------------------------------------------
fw_means = np.append(fw_means, median_fw0)
fw_means = np.append(fw_means, median_fw)
fa_means = np.append(fa_means, median_fa0)
fa_means = np.append(fa_means, median_fa)
md_means = np.append(md_means, median_md0)
md_means = np.append(md_means, median_md)

fw_lows = np.append(fw_lows, low_fw0)
fw_lows = np.append(fw_lows, low_fw)
fa_lows = np.append(fa_lows, low_fa0)
fa_lows = np.append(fa_lows, low_fa)
md_lows = np.append(md_lows, low_md0)
md_lows = np.append(md_lows, low_md)

fw_highs = np.append(fw_highs, high_fw0)
fw_highs = np.append(fw_highs, high_fw)
fa_highs = np.append(fa_highs, high_fa0)
fa_highs = np.append(fa_highs, high_fa)
md_highs = np.append(md_highs, high_md0)
md_highs = np.append(md_highs, high_md)

# ------------------------------------------------------------------------------
print('Running Beltrami (hybrid init)...')
# ------------------------------------------------------------------------------
bmodel = BeltramiModel(gtab, init_method='hybrid', Stissue=St, Swater=Sw,
                       iterations=200, learning_rate=0.0005)
bfit = bmodel.fit(phantom, mask=mask1)
belt_fw = bfit.fw * mask1
belt_fa = bfit.fa * mask1
belt_md = bfit.md * mask1
belt_fw0 = bfit.fw0 * mask1
belt_fa0 = bfit.fa0 * mask1
belt_md0 = bfit.md0 * mask1

# Final error
median_fw = np.median(belt_fw[mask])
p25_fw, p75_fw = np.percentile(belt_fw[mask], [25, 75])
low_fw = median_fw - p25_fw
high_fw = p75_fw - median_fw

median_fa = np.median(belt_fa[mask])
p25_fa, p75_fa = np.percentile(belt_fa[mask], [25, 75])
low_fa = median_fa - p25_fa
high_fa = p75_fa - median_fa

median_md = np.median(belt_md[mask])
p25_md, p75_md = np.percentile(belt_md[mask], [25, 75])
low_md = median_md - p25_md
high_md = p75_md - median_md

# Initial error
median_fw0 = np.median(belt_fw0[mask])
p25_fw0, p75_fw0 = np.percentile(belt_fw0[mask], [25, 75])
low_fw0 = median_fw0 - p25_fw0
high_fw0 = p75_fw0 - median_fw0

median_fa0 = np.median(belt_fa0[mask])
p25_fa0, p75_fa0 = np.percentile(belt_fa0[mask], [25, 75])
low_fa0 = median_fa0 - p25_fa0
high_fa0 = p75_fa0 - median_fa0

median_md0 = np.median(belt_md0[mask])
p25_md0, p75_md0 = np.percentile(belt_md0[mask], [25, 75])
low_md0 = median_md0 - p25_md0
high_md0 = p75_md0 - median_md0

#--------------------------------------------
fw_means = np.append(fw_means, median_fw0)
fw_means = np.append(fw_means, median_fw)
fa_means = np.append(fa_means, median_fa0)
fa_means = np.append(fa_means, median_fa)
md_means = np.append(md_means, median_md0)
md_means = np.append(md_means, median_md)

fw_lows = np.append(fw_lows, low_fw0)
fw_lows = np.append(fw_lows, low_fw)
fa_lows = np.append(fa_lows, low_fa0)
fa_lows = np.append(fa_lows, low_fa)
md_lows = np.append(md_lows, low_md0)
md_lows = np.append(md_lows, low_md)

fw_highs = np.append(fw_highs, high_fw0)
fw_highs = np.append(fw_highs, high_fw)
fa_highs = np.append(fa_highs, high_fa0)
fa_highs = np.append(fa_highs, high_fa)
md_highs = np.append(md_highs, high_md0)
md_highs = np.append(md_highs, high_md)

# ------------------------------------------------------------------------------
print('Running NLS FW-DTI (Hoy)...')
# ------------------------------------------------------------------------------

# rescaling gtab for NLS algorithm
gtab = gradient_table(1000 * bvals, bvecs, b0_threshold=0)

nls_model = fwdti.FreeWaterTensorModel(gtab)
nls_fit = nls_model.fit(phantom)
nls_fa = nls_fit.fa * mask1
nls_md = nls_fit.md * mask1 * 10**3
nls_fw = nls_fit.f * mask1

# Final error
median_fw = np.median(nls_fw[mask])
p25_fw, p75_fw = np.percentile(nls_fw[mask], [25, 75])
low_fw = median_fw - p25_fw
high_fw = p75_fw - median_fw

median_fa = np.median(nls_fa[mask])
p25_fa, p75_fa = np.percentile(nls_fa[mask], [25, 75])
low_fa = median_fa - p25_fa
high_fa = p75_fa - median_fa

median_md = np.median(nls_md[mask])
p25_md, p75_md = np.percentile(nls_md[mask], [25, 75])
low_md = median_md - p25_md
high_md = p75_md - median_md

fw_means = np.append(fw_means, median_fw)
fa_means = np.append(fa_means, median_fa)
md_means = np.append(md_means, median_md)

fw_lows = np.append(fw_lows, low_fw)
fa_lows = np.append(fa_lows, low_fa)
md_lows = np.append(md_lows, low_md)

fw_highs = np.append(fw_highs, high_fw)
fa_highs = np.append(fa_highs, high_fa)
md_highs = np.append(md_highs, high_md)

# ------------------------------------------------------------------------------
print('Plotting figure...')
# ------------------------------------------------------------------------------

# fig = plt.figure(figsize=(6.9, 3.6))
fig = plt.figure(figsize=(10, 5.2))
gs1 = fig.add_gridspec(ncols=4, nrows=3, hspace=0.05, wspace=0.05, left=0.1, right=0.63)

# axes
# ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs1[0, 0])
ax3 = fig.add_subplot(gs1[0, 1])
ax4 = fig.add_subplot(gs1[0, 2])
# ax5 = fig.add_subplot(gs1[1, 0])
ax10 = fig.add_subplot(gs1[1, 0])
ax11 = fig.add_subplot(gs1[1, 1])
ax12= fig.add_subplot(gs1[1, 2])
# ax9 = fig.add_subplot(gs1[2, 0])
ax6 = fig.add_subplot(gs1[2, 0])
ax7 = fig.add_subplot(gs1[2, 1])
ax8 = fig.add_subplot(gs1[2, 2])

# adding axes for FW-DTI NLS maps
ax16 = fig.add_subplot(gs1[0, 3])
ax18 = fig.add_subplot(gs1[1, 3])
ax17 = fig.add_subplot(gs1[2, 3])

# column titles
# ax1.set_title('GT')
ax2.set_title('GT')
ax3.set_title('Hybrid init')
ax4.set_title('RGD FWE')
ax16.set_title('NLS FWE')

# ylablels
ax2.set_ylabel('FW')
ax6.set_ylabel('FA')
ax10.set_ylabel('MD\n' + r'$[\mu m^2 ms^{-1}]$')

# removing ticks
all_axes = [ax2, ax3, ax4, ax6, ax7, ax8, ax10, ax11, ax12, ax16,
            ax17, ax18]

for ax in all_axes:
    ax.set_xticks([])
    ax.set_yticks([])

# ax1.imshow(gt_fw[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
ax2.imshow(gt_fw[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
ax3.imshow(belt_fw0[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
mp = ax4.imshow(belt_fw[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
cax = fig.add_axes([0.64, 0.63, 0.015, 0.25])
cticks = [0.25, 0.5, 0.75, 1]
fig.colorbar(mp, cax=cax, ticks=cticks)

# ax5.imshow(gt_fa[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
ax6.imshow(gt_fa[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
ax7.imshow(belt_fa0[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
mp = ax8.imshow(belt_fa[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
cax = fig.add_axes([0.64, 0.11, 0.015, 0.25])
cticks = [0.25, 0.5, 0.75, 1]
fig.colorbar(mp, cax=cax, ticks=cticks)


# ax9.imshow(gt_md[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=3)
ax10.imshow(gt_md[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=3)
ax11.imshow(belt_md0[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=3)
mp = ax12.imshow(belt_md[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=3)
cticks = [0.75, 1.5, 2.25, 3]
cax = fig.add_axes([0.64, 0.37, 0.015, 0.25])
fig.colorbar(mp, cax=cax, ticks=cticks)

# NLS
ax16.imshow(nls_fw[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
ax17.imshow(nls_fa[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=1)
ax18.imshow(nls_md[:, 10, :].T, origin='lower', cmap='hot', vmin=0, vmax=3)

# error plots ------------------------------------------------------------------
gs2 = fig.add_gridspec(ncols=1, nrows=3, left=0.71, right=0.88, hspace=0.05)

ax13 = fig.add_subplot(gs2[0, 0])
ax15 = fig.add_subplot(gs2[1, 0])
ax14 = fig.add_subplot(gs2[2, 0])

msize = 7

colors = {0: 'red', 1: 'blue', 2: 'green', 3:'m', 4:'darkorange'}
labels = {0: 'S0 init', 1: 'MD init', 2: 'Hybrid', 3:'NLS', 4:'GT'}

n = [1, 2, 4, 5, 7, 8, 10]
x = n[0:2]
y = fw_means[0:2]
low_err = fw_lows[0:2]
high_err = fw_highs[0:2]
ax13.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[0], label=labels[0], ms=msize)

x = n[2:4]
y = fw_means[2:4]
low_err = fw_lows[2:4]
high_err = fw_highs[2:4]
ax13.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[1], label=labels[1], ms=msize)

x = n[4:6]
y = fw_means[4:6]
low_err = fw_lows[4:6]
high_err = fw_highs[4:6]
ax13.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[2], label=labels[2], ms=msize)

x = n[6:7]
y = fw_means[6:7]
low_err = fw_lows[6:7]
high_err = fw_highs[6:7]
ax13.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[3], label=labels[3], ms=msize)

x = np.arange(0, 12)
y = np.median(gt_fw[mask]) * np.ones(x.shape)
ax13.plot(x, y, color=colors[4], label=labels[4], linestyle='dashed', ms=msize)

ax13.legend(loc='center left', bbox_to_anchor=(0.01, 0.69), prop={'size': 8},
            frameon=False)

# ------------------------------------------------------------------------------
x = n[0:2]
y = fa_means[0:2]
low_err = fa_lows[0:2]
high_err = fa_highs[0:2]
ax14.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[0], ms=msize)

x = n[2:4]
y = fa_means[2:4]
low_err = fa_lows[2:4]
high_err = fa_highs[2:4]
ax14.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[1], ms=msize)

x = n[4:6]
y = fa_means[4:6]
low_err = fa_lows[4:6]
high_err = fa_highs[4:6]
ax14.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[2], ms=msize)

x = n[6:7]
y = fa_means[6:7]
low_err = fa_lows[6:7]
high_err = fa_highs[6:7]
ax14.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[3], ms=msize)

x = np.arange(0, 12)
y = np.median(gt_fa[mask]) * np.ones(x.shape)
ax14.plot(x, y, color=colors[4], label=labels[4], linestyle='dashed', ms=msize)

# ------------------------------------------------------------------------------
x = n[0:2]
y = md_means[0:2]
low_err = md_lows[0:2]
high_err = md_highs[0:2]
ax15.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[0], ms=msize)

x = n[2:4]
y = md_means[2:4]
low_err = md_lows[2:4]
high_err = md_highs[2:4]
ax15.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[1], ms=msize)

x = n[4:6]
y = md_means[4:6]
low_err = md_lows[4:6]
high_err = md_highs[4:6]
ax15.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[2], ms=msize)

x = n[6:7]
y = md_means[6:7]
low_err = md_lows[6:7]
high_err = md_highs[6:7]
ax15.errorbar(x, y, fmt='.', yerr=[low_err, high_err], color=colors[3], ms=msize)

x = np.arange(0, 12)
y = np.median(gt_md[mask]) * np.ones(x.shape)
ax15.plot(x, y, color=colors[4], label=labels[4], linestyle='dashed', ms=msize)

#-------------------
# ax13.set_title('median\n(init. vs est.)')

ax13.set_xlim([0, 11])
ax14.set_xlim([0, 11])
ax15.set_xlim([0, 11])

xticks = [1, 2, 4, 5, 7, 8, 10]
ax13.set_xticks(xticks)
ax14.set_xticks(xticks)
ax15.set_xticks(xticks)

ax13.set_xticklabels([])
ax14.set_xticklabels([])

ax13.set_ylim([0, 1.5])
ax14.set_ylim([0, 1])
ax15.set_ylim([0, 1.1])

yticks = (0.1, 0.5, 0.9)
ax13.set_yticks(yticks)
ax14.set_yticks(yticks)
ax15.set_yticks(yticks)

xticklabels = ['init', 'RGD', 'init', 'RGD', 'init', 'RGD', 'NLS']
ax14.set_xticklabels(xticklabels, rotation=90)

ax13.yaxis.tick_right()
ax14.yaxis.tick_right()
ax15.yaxis.tick_right()

plt.show()

# ------------------------------------------------------------------------------
print('Saving...')  # (comment / uncomment)
# ------------------------------------------------------------------------------
fout = '/home/mrk/Desktop/'
fname = 'fig3_final'
fig.savefig(fout + fname + '.png', format='png', dpi=600, bbox_inches='tight')
fig.savefig(fout + fname + '.eps', format='eps', dpi=600, bbox_inches='tight')
print('All done!')