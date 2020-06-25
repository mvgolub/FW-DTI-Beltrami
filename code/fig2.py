import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, HemiSphere

from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity

from beltrami import (fraction_init_s0, fraction_init_md, fraction_init_hybrid,
                      tensor_init)

from functions import dual_tensor, generate_eigvalues


# ------------------------------------------------------------------------------
# Defining simulation parameters
# ------------------------------------------------------------------------------

# Parameters that define the number of simulations
n_pts = 32     # number of gradient directions
nMDs = 10      # number of simulated MD values
nFrac = 10     # number of simulated F values
nDTdirs = 5  # number of diffusion tensor principal directions
nReps = 5    # number of noise instances
SNR = 40       # noise level simulated for nReps instances

# Ground truth parameters of a single voxel signal
L1 = 1.6
L2 = 0.5
L3 = 0.3
evals = np.array([L1, L2, L3])
Dw = 3

FWs = np.linspace(0.1, 1, num=nFrac)
Fs = 1 - FWs

FA = fractional_anisotropy(evals)
MD = mean_diffusivity(evals)
MDs = np.linspace(0.1, 1.6, num=nMDs)

ratios = MDs / MD
evals = (evals[..., None] * ratios).T
FAs = fractional_anisotropy(evals)
L1s = evals[:, 0]
L2s = evals[:, 1]
L3s = evals[:, 2]

print('----------------------------- Parameters -----------------------------')
# print('ground truth FWs are ' + str(FWs))
print('Ground truth FA is ' + str(FA))
# print('ground truth MDs are ' + str(MDs))


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

# gyromagnetic ratio of proton (ms^-1 mT^-1)
# gamma = 2.6752218744 * 10**2
gamma = 267.52218744 * 10**6

# Strength of diffusion gradients (mT m^-1)
# Gmax = 45
Gmax = 0.045

# Simulating the S0 signal for each compartment (tissue and water), assuming a
# single-shell acquisition with b-value = 1000 
# b = 1                      # units of milisecond / micrometer^2
# bval = np.copy(b) * 10**12  # (ms m^-2)  to compute echo time TE in ms
b = 1
bval = 1000 * 10**6

# Assuming:
# - a simple spin echo sequense
# - the diffusion gradient pulses are rectangular
# - interval between pulses (Delta) =~ pulse duration (d)
# Then d =~ 2 * TE, replacing in b-value formula and solving with respect to TE,
# gives the ideal TEs for each desired b-value
TE = 2 * (3 / 2 * bval / (gamma**2 * Gmax**2))**(1/3)  # ms
TE *= 10**3
TE = 74

# Assuming no T1 relaxation, the non-weighted signal for a voxel that has only
# tisssue or only water:
k = 10
St = k * PD_tissue * np.exp(-TE / T2_tissue)
Sw = k * PD_water * np.exp(-TE / T2_water)
S0s = Fs * St + FWs * Sw
F_cor = Fs * Sw / ((1 - Fs) * St + Fs * Sw)

print('S0 signal for a voxel with only tissue is ' + str(St))
print('S0 signal for a voxel with only water is ' + str(Sw))
print(('S0 signal for voxel with ' + str(0.5*100) + ' % tissue: ' +
       str(St * 0.5 + Sw * 0.5) + '\n(TE = ' + str(np.round(TE)) + ' ms, ' +
       'b-value = ' + str(np.round(bval*10**-9)) + ' s/mm^2)'))
print('-----------------------------------------------------------------------')


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

# A single acquisition scheme will be tested, with b-value = 1000 and 6 b0s
bval = b  # units of milisecond / micrometer^2
bvals = bval * np.ones(n_pts + 6)
bvals[0:6] = 0
bvecs = np.zeros((n_pts + 6, 3))
bvecs[6:, :] = directions
gtab = gradient_table(bvals, bvecs, b0_threshold=0)


# The experiment is repeated for 100 diffusion tensor directions and 100
# noise repeats
theta = np.pi * np.random.rand(nDTdirs)
phi = 2 * np.pi * np.random.rand(nDTdirs)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
DTdirs = hsph_updated.vertices


# ------------------------------------------------------------------------------
print('Generating single voxel DWIs...')
# ------------------------------------------------------------------------------

# Allocating array that holds all signals,
# shape: (nMDs, nFrac, nDTdirs*nReps, n_pts + 1)
DWIs = np.zeros((nMDs, nFrac, nDTdirs*nReps, bvals.size))

for md_i in np.arange(nMDs):

    # for current ground truth MD
    L1 = L1s[md_i]
    L2 = L2s[md_i]
    L3 = L3s[md_i]

    mevals = np.array([[L1, L2, L3],
                       [Dw, Dw, Dw]])
    
    for f_i in np.arange(nFrac):

        # for current volume fraction
        fractions = np.array([F_cor[f_i] * 100, (1 - F_cor[f_i]) * 100])

        for d_i in np.arange(nDTdirs):

            # for current simulated tensor direction
            DTdir = DTdirs[d_i]
            angles = [DTdir, (1, 0, 0)]

            for n_i in np.arange(d_i * nReps, (d_i+1) * nReps):

                # for current noise instance
                signal, _ = dual_tensor(gtab, mevals, S0t=St, S0w=Sw,
                                             angles=angles, fractions=fractions,
                                             snr=SNR)

                DWIs[md_i, f_i, n_i, :] = signal


# ------------------------------------------------------------------------------
print('Initializing tissue fractions and tissue tensor...')
# ------------------------------------------------------------------------------

# initialize volume fractions
f0_s0, fmin, fmax = fraction_init_s0(DWIs, gtab, Stissue=St, Swater=Sw)
f0_md, _, _ = fraction_init_md(DWIs, gtab, tissue_MD=0.6)
f0_hybrid, _, _ = fraction_init_hybrid(DWIs, gtab, Stissue=St, Swater=Sw,
                                       tissue_MD=0.6)

# correct signal and initialize tissue tensor
dti_params_s0 = tensor_init(DWIs, gtab, f0_s0)
dti_params_md = tensor_init(DWIs, gtab, f0_md)
dti_params_hybrid = tensor_init(DWIs, gtab, f0_hybrid)


# ------------------------------------------------------------------------------
print('Plotting figure...')
# ------------------------------------------------------------------------------
msize = 5  # Marker size
lw = 1.5   # Line width

colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'darkorange'}
labels = {0: 'S0 init', 1: 'MD init', 2: 'Hyb.', 3: 'GT'}

fig = plt.figure(figsize=(7.2, 5.7))
gs = gridspec.GridSpec(3, 4)
# gs.update(hspace=0.08, wspace=0.05)
gs.update(hspace=0.1, wspace=0.1)


ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[0, 3])

ax5 = plt.subplot(gs[1, 0])
ax6 = plt.subplot(gs[1, 1])
ax7 = plt.subplot(gs[1, 2])
ax8 = plt.subplot(gs[1, 3])

ax9 = plt.subplot(gs[2, 0])
ax10 = plt.subplot(gs[2, 1])
ax11 = plt.subplot(gs[2, 2])
ax12 = plt.subplot(gs[2, 3])

# y labels
ax1.set_ylabel('FW')
ax5.set_ylabel('MD\n' + r'$[\mu m^2 ms^{-1}]$')
ax9.set_ylabel('FA')

# x labels
ax9.set_xlabel('GT FW')
ax10.set_xlabel('GT MD\n' + r'$[\mu m^2 ms^{-1}]$')
ax11.set_xlabel('GT MD\n' + r'$[\mu m^2 ms^{-1}]$')
ax12.set_xlabel('GT MD\n' + r'$[\mu m^2 ms^{-1}]$')

# axes limits
MD_lims = [0, 1.7]
FW_lims = [0, 1.1]

# xlims
ax1.set_xlim(FW_lims)
ax2.set_xlim(MD_lims)
ax3.set_xlim(MD_lims)
ax4.set_xlim(MD_lims)

ax5.set_xlim(FW_lims)
ax6.set_xlim(MD_lims)
ax7.set_xlim(MD_lims)
ax8.set_xlim(MD_lims)

ax9.set_xlim(FW_lims)
ax10.set_xlim(MD_lims)
ax11.set_xlim(MD_lims)
ax12.set_xlim(MD_lims)

# ylims
ax1.set_ylim(FW_lims)
ax2.set_ylim(FW_lims)
ax3.set_ylim(FW_lims)
ax4.set_ylim(FW_lims)

ax5.set_ylim([0, 2.6])
ax6.set_ylim([0, 2.6])
ax7.set_ylim([0, 2.6])
ax8.set_ylim([0, 2.6])

ax9.set_ylim(FW_lims)
ax10.set_ylim(FW_lims)
ax11.set_ylim(FW_lims)
ax12.set_ylim(FW_lims)

# ticks an tick labels
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])
ax5.set_xticklabels([])
ax6.set_xticklabels([])
ax7.set_xticklabels([])
ax8.set_xticklabels([])

ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax6.set_yticklabels([])
ax7.set_yticklabels([])
ax8.set_yticklabels([])
ax10.set_yticklabels([])
ax11.set_yticklabels([])
ax12.set_yticklabels([])

MD_ticks = np.linspace(0.1, 1.6, num=4)
FW_ticks = np.linspace(0.1, 1, num=4)

ax1.set_xticks(FW_ticks)
ax5.set_xticks(FW_ticks)
ax9.set_xticks(FW_ticks)

ax2.set_xticks(MD_ticks)
ax3.set_xticks(MD_ticks)
ax4.set_xticks(MD_ticks)
ax6.set_xticks(MD_ticks)
ax7.set_xticks(MD_ticks)
ax8.set_xticks(MD_ticks)
ax10.set_xticks(MD_ticks)
ax11.set_xticks(MD_ticks)
ax12.set_xticks(MD_ticks)

ax1.set_yticks(FW_ticks)
ax2.set_yticks(FW_ticks)
ax3.set_yticks(FW_ticks)
ax4.set_yticks(FW_ticks)

ax5.set_yticks(np.linspace(0.1, 2.5, num=4))
ax6.set_yticks(np.linspace(0.1, 2.5, num=4))
ax7.set_yticks(np.linspace(0.1, 2.5, num=4))
ax8.set_yticks(np.linspace(0.1, 2.5, num=4))

ax9.set_yticks(FW_ticks)
ax10.set_yticks(FW_ticks)
ax11.set_yticks(FW_ticks)
ax12.set_yticks(FW_ticks)

# Data for fixed MD = 0.6
md_i = 3

# FW
median_fw_s0 = np.median(1 - f0_s0[md_i, ...], axis=-1)
median_fw_md = np.median(1 - f0_md[md_i, ...], axis=-1)
median_fw_hybrid = np.median(1 - f0_hybrid[md_i, ...], axis=-1)
p25_fw_s0, p75_fw_s0 = np.percentile(1 - f0_s0[md_i, ...], [25, 75], axis=-1)
p25_fw_md, p75_fw_md = np.percentile(1 - f0_md[md_i, ...], [25, 75], axis=-1)
p25_fw_hybrid, p75_fw_hybrid = np.percentile(1 - f0_hybrid[md_i, ...], [25, 75], axis=-1)
low_fw_s0 = median_fw_s0 - p25_fw_s0
low_fw_md = median_fw_md - p25_fw_md
low_fw_hybrid = median_fw_hybrid - p25_fw_hybrid
high_fw_s0 = p75_fw_s0 - median_fw_s0
high_fw_md = p75_fw_md - median_fw_md
high_fw_hybrid = p75_fw_hybrid - median_fw_hybrid

# FA
gt_fa = FAs[md_i] * np.ones(nFrac)  # ground truth
fa_s0 = fractional_anisotropy(dti_params_s0[..., 0:3])
fa_md = fractional_anisotropy(dti_params_md[..., 0:3])
fa_hybrid = fractional_anisotropy(dti_params_hybrid[..., 0:3])
median_fa_s0 = np.median(fa_s0[md_i, ...], axis=-1)
median_fa_md = np.median(fa_md[md_i, ...], axis=-1)
median_fa_hybrid = np.median(fa_hybrid[md_i, ...], axis=-1)
p25_fa_s0, p75_fa_s0 = np.percentile(fa_s0[md_i, ...], [25, 75], axis=-1)
p25_fa_md, p75_fa_md = np.percentile(fa_md[md_i, ...], [25, 75], axis=-1)
p25_fa_hybrid, p75_fa_hybrid = np.percentile(fa_hybrid[md_i, ...], [25, 75], axis=-1)
low_fa_s0 = median_fa_s0 - p25_fa_s0
low_fa_md = median_fa_md - p25_fa_md
low_fa_hybrid = median_fa_hybrid - p25_fa_hybrid
high_fa_s0 = p75_fa_s0 - median_fa_s0
high_fa_md = p75_fa_md - median_fa_md
high_fa_hybrid = p75_fa_hybrid - median_fa_hybrid

# MD
gt_md = MDs[md_i] * np.ones(nFrac)
md_s0 = mean_diffusivity(dti_params_s0[..., 0:3])
md_md = mean_diffusivity(dti_params_md[..., 0:3])
md_hybrid = mean_diffusivity(dti_params_hybrid[..., 0:3])
median_md_s0 = np.median(md_s0[md_i, ...], axis=-1)
median_md_md = np.median(md_md[md_i, ...], axis=-1)
median_md_hybrid = np.median(md_hybrid[md_i, ...], axis=-1)
p25_md_s0, p75_md_s0 = np.percentile(md_s0[md_i, ...], [25, 75], axis=-1)
p25_md_md, p75_md_md = np.percentile(md_md[md_i, ...], [25, 75], axis=-1)
p25_md_hybrid, p75_md_hybrid = np.percentile(md_hybrid[md_i, ...], [25, 75], axis=-1)
low_md_s0 = median_md_s0 - p25_md_s0
low_md_md = median_md_md - p25_md_md
low_md_hybrid = median_md_hybrid - p25_md_hybrid
high_md_s0 = p75_md_s0 - median_md_s0
high_md_md = p75_md_md - median_md_md
high_md_hybrid = p75_md_hybrid - median_md_hybrid


ax1.errorbar(FWs, median_fw_s0, fmt='.', yerr=[low_fw_s0, high_fw_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax1.errorbar(FWs, median_fw_md, fmt='.', yerr=[low_fw_md, high_fw_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax1.errorbar(FWs, median_fw_hybrid, fmt='.',
              yerr=[low_fw_hybrid, high_fw_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax1.plot(FWs, FWs, color=colors[3], label=labels[3], linewidth=lw, ls='--')
ax1.set_title('GT MD = ' + str(MDs[md_i]))
ax1.legend(loc='center left', bbox_to_anchor=(-0.08, 0.73), prop={'size': 9},
           frameon=False)

#-------------------------------------------------------------------------------
ax5.errorbar(FWs, median_md_s0, fmt='.', yerr=[low_md_s0, high_md_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax5.errorbar(FWs, median_md_md, fmt='.', yerr=[low_md_md, high_md_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax5.errorbar(FWs, median_md_hybrid, fmt='.',
              yerr=[low_md_hybrid, high_md_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax5.plot(FWs, gt_md, color=colors[3], label=labels[3], linewidth=lw, ls='--')

print(FWs[-1])
print(median_md_s0[-1])
print(median_md_md[-1])
print(median_md_hybrid[-1])

#-------------------------------------------------------------------------------
ax9.errorbar(FWs, median_fa_s0, fmt='.', yerr=[low_fa_s0, high_fa_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax9.errorbar(FWs, median_fa_md, fmt='.', yerr=[low_fa_md, high_fa_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax9.errorbar(FWs, median_fa_hybrid, fmt='.',
              yerr=[low_fa_hybrid, high_fa_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax9.plot(FWs, gt_fa, color=colors[3], label=labels[3], linewidth=lw, ls='--')


# Data for fixed FW = 0.2
f_i = 1
# FW
gt_fw = FWs[f_i] * np.ones(nMDs)
median_fw_s0 = np.median(1 - f0_s0[:, f_i, ...], axis=-1)
median_fw_md = np.median(1 - f0_md[:, f_i, ...], axis=-1)
median_fw_hybrid = np.median(1 - f0_hybrid[:, f_i, ...], axis=-1)
p25_fw_s0, p75_fw_s0 = np.percentile(1 - f0_s0[:, f_i, ...], [25, 75], axis=-1)
p25_fw_md, p75_fw_md = np.percentile(1 - f0_md[:, f_i, ...], [25, 75], axis=-1)
p25_fw_hybrid, p75_fw_hybrid = np.percentile(1 - f0_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_fw_s0 = median_fw_s0 - p25_fw_s0
low_fw_md = median_fw_md - p25_fw_md
low_fw_hybrid = median_fw_hybrid - p25_fw_hybrid
high_fw_s0 = p75_fw_s0 - median_fw_s0
high_fw_md = p75_fw_md - median_fw_md
high_fw_hybrid = p75_fw_hybrid - median_fw_hybrid

# FA
gt_fa = FAs[f_i] * np.ones(nMDs)
fa_s0 = fractional_anisotropy(dti_params_s0[ ..., 0:3])
fa_md = fractional_anisotropy(dti_params_md[..., 0:3])
fa_hybrid = fractional_anisotropy(dti_params_hybrid[..., 0:3])
median_fa_s0 = np.median(fa_s0[:, f_i, ...], axis=-1)
median_fa_md = np.median(fa_md[:, f_i, ...], axis=-1)
median_fa_hybrid = np.median(fa_hybrid[:, f_i, ...], axis=-1)
p25_fa_s0, p75_fa_s0 = np.percentile(fa_s0[:, f_i, ...], [25, 75], axis=-1)
p25_fa_md, p75_fa_md = np.percentile(fa_md[:, f_i, ...], [25, 75], axis=-1)
p25_fa_hybrid, p75_fa_hybrid = np.percentile(fa_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_fa_s0 = median_fa_s0 - p25_fa_s0
low_fa_md = median_fa_md - p25_fa_md
low_fa_hybrid = median_fa_hybrid - p25_fa_hybrid
high_fa_s0 = p75_fa_s0 - median_fa_s0
high_fa_md = p75_fa_md - median_fa_md
high_fa_hybrid = p75_fa_hybrid - median_fa_hybrid

# MD
md_s0 = mean_diffusivity(dti_params_s0[..., 0:3])
md_md = mean_diffusivity(dti_params_md[..., 0:3])
md_hybrid = mean_diffusivity(dti_params_hybrid[..., 0:3])
median_md_s0 = np.median(md_s0[:, f_i, ...], axis=-1)
median_md_md = np.median(md_md[:, f_i, ...], axis=-1)
median_md_hybrid = np.median(md_hybrid[:, f_i, ...], axis=-1)
p25_md_s0, p75_md_s0 = np.percentile(md_s0[:, f_i, ...], [25, 75], axis=-1)
p25_md_md, p75_md_md = np.percentile(md_md[:, f_i, ...], [25, 75], axis=-1)
p25_md_hybrid, p75_md_hybrid = np.percentile(md_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_md_s0 = median_md_s0 - p25_md_s0
low_md_md = median_md_md - p25_md_md
low_md_hybrid = median_md_hybrid - p25_md_hybrid
high_md_s0 = p75_md_s0 - median_md_s0
high_md_md = p75_md_md - median_md_md
high_md_hybrid = p75_md_hybrid - median_md_hybrid

ax2.errorbar(MDs, median_fw_s0, fmt='.', yerr=[low_fw_s0, high_fw_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax2.errorbar(MDs, median_fw_md, fmt='.', yerr=[low_fw_md, high_fw_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax2.errorbar(MDs, median_fw_hybrid, fmt='.',
              yerr=[low_fw_hybrid, high_fw_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax2.plot(MDs, gt_fw, color=colors[3], label=labels[3], linewidth=lw, ls='--')
ax2.set_title('GT FW = ' + str(FWs[f_i]))

#-------------------------------------------------------------------------------
ax6.errorbar(MDs, median_md_s0, fmt='.', yerr=[low_md_s0, high_md_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax6.errorbar(MDs, median_md_md, fmt='.', yerr=[low_md_md, high_md_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax6.errorbar(MDs, median_md_hybrid, fmt='.',
              yerr=[low_md_hybrid, high_md_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax6.plot(MDs, MDs, color=colors[3], label=labels[3], linewidth=lw, ls='--')

#-------------------------------------------------------------------------------
ax10.errorbar(MDs, median_fa_s0, fmt='.', yerr=[low_fa_s0, high_fa_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax10.errorbar(MDs, median_fa_md, fmt='.', yerr=[low_fa_md, high_fa_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax10.errorbar(MDs, median_fa_hybrid, fmt='.',
              yerr=[low_fa_hybrid, high_fa_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax10.plot(MDs, gt_fa, color=colors[3], label=labels[3], linewidth=lw, ls='--')


# Data for fixed FW = 0.5
f_i = 4
# FW
gt_fw = FWs[f_i] * np.ones(nMDs)
median_fw_s0 = np.median(1 - f0_s0[:, f_i, ...], axis=-1)
median_fw_md = np.median(1 - f0_md[:, f_i, ...], axis=-1)
median_fw_hybrid = np.median(1 - f0_hybrid[:, f_i, ...], axis=-1)
p25_fw_s0, p75_fw_s0 = np.percentile(1 - f0_s0[:, f_i, ...], [25, 75], axis=-1)
p25_fw_md, p75_fw_md = np.percentile(1 - f0_md[:, f_i, ...], [25, 75], axis=-1)
p25_fw_hybrid, p75_fw_hybrid = np.percentile(1 - f0_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_fw_s0 = median_fw_s0 - p25_fw_s0
low_fw_md = median_fw_md - p25_fw_md
low_fw_hybrid = median_fw_hybrid - p25_fw_hybrid
high_fw_s0 = p75_fw_s0 - median_fw_s0
high_fw_md = p75_fw_md - median_fw_md
high_fw_hybrid = p75_fw_hybrid - median_fw_hybrid

# FA
gt_fa = FAs[f_i] * np.ones(nMDs)
fa_s0 = fractional_anisotropy(dti_params_s0[..., 0:3])
fa_md = fractional_anisotropy(dti_params_md[..., 0:3])
fa_hybrid = fractional_anisotropy(dti_params_hybrid[..., 0:3])
median_fa_s0 = np.median(fa_s0[:, f_i, ...], axis=-1)
median_fa_md = np.median(fa_md[:, f_i, ...], axis=-1)
median_fa_hybrid = np.median(fa_hybrid[:, f_i, ...], axis=-1)
p25_fa_s0, p75_fa_s0 = np.percentile(fa_s0[:, f_i, ...], [25, 75], axis=-1)
p25_fa_md, p75_fa_md = np.percentile(fa_md[:, f_i, ...], [25, 75], axis=-1)
p25_fa_hybrid, p75_fa_hybrid = np.percentile(fa_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_fa_s0 = median_fa_s0 - p25_fa_s0
low_fa_md = median_fa_md - p25_fa_md
low_fa_hybrid = median_fa_hybrid - p25_fa_hybrid
high_fa_s0 = p75_fa_s0 - median_fa_s0
high_fa_md = p75_fa_md - median_fa_md
high_fa_hybrid = p75_fa_hybrid - median_fa_hybrid

# MD
md_s0 = mean_diffusivity(dti_params_s0[..., 0:3])
md_md = mean_diffusivity(dti_params_md[..., 0:3])
md_hybrid = mean_diffusivity(dti_params_hybrid[..., 0:3])
median_md_s0 = np.median(md_s0[:, f_i, ...], axis=-1)
median_md_md = np.median(md_md[:, f_i, ...], axis=-1)
median_md_hybrid = np.median(md_hybrid[:, f_i, ...], axis=-1)
p25_md_s0, p75_md_s0 = np.percentile(md_s0[:, f_i, ...], [25, 75], axis=-1)
p25_md_md, p75_md_md = np.percentile(md_md[:, f_i, ...], [25, 75], axis=-1)
p25_md_hybrid, p75_md_hybrid = np.percentile(md_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_md_s0 = median_md_s0 - p25_md_s0
low_md_md = median_md_md - p25_md_md
low_md_hybrid = median_md_hybrid - p25_md_hybrid
high_md_s0 = p75_md_s0 - median_md_s0
high_md_md = p75_md_md - median_md_md
high_md_hybrid = p75_md_hybrid - median_md_hybrid

ax3.errorbar(MDs, median_fw_s0, fmt='.', yerr=[low_fw_s0, high_fw_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax3.errorbar(MDs, median_fw_md, fmt='.', yerr=[low_fw_md, high_fw_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax3.errorbar(MDs, median_fw_hybrid, fmt='.',
              yerr=[low_fw_hybrid, high_fw_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax3.plot(MDs, gt_fw, color=colors[3], label=labels[3], linewidth=lw, ls='--')
ax3.set_title('GT FW = ' + str(FWs[f_i]))

#-------------------------------------------------------------------------------
ax7.errorbar(MDs, median_md_s0, fmt='.', yerr=[low_md_s0, high_md_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax7.errorbar(MDs, median_md_md, fmt='.', yerr=[low_md_md, high_md_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax7.errorbar(MDs, median_md_hybrid, fmt='.',
              yerr=[low_md_hybrid, high_md_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax7.plot(MDs, MDs, color=colors[3], label=labels[3], linewidth=lw, ls='--')

#-------------------------------------------------------------------------------
ax11.errorbar(MDs, median_fa_s0, fmt='.', yerr=[low_fa_s0, high_fa_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax11.errorbar(MDs, median_fa_md, fmt='.', yerr=[low_fa_md, high_fa_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax11.errorbar(MDs, median_fa_hybrid, fmt='.',
              yerr=[low_fa_hybrid, high_fa_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax11.plot(MDs, gt_fa, color=colors[3], label=labels[3], linewidth=lw, ls='--')


# Data for fixed FW = 0.8
f_i = 7
# FW
gt_fw = FWs[f_i] * np.ones(nMDs)
median_fw_s0 = np.median(1 - f0_s0[:, f_i, ...], axis=-1)
median_fw_md = np.median(1 - f0_md[:, f_i, ...], axis=-1)
median_fw_hybrid = np.median(1 - f0_hybrid[:, f_i, ...], axis=-1)
p25_fw_s0, p75_fw_s0 = np.percentile(1 - f0_s0[:, f_i, ...], [25, 75], axis=-1)
p25_fw_md, p75_fw_md = np.percentile(1 - f0_md[:, f_i, ...], [25, 75], axis=-1)
p25_fw_hybrid, p75_fw_hybrid = np.percentile(1 - f0_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_fw_s0 = median_fw_s0 - p25_fw_s0
low_fw_md = median_fw_md - p25_fw_md
low_fw_hybrid = median_fw_hybrid - p25_fw_hybrid
high_fw_s0 = p75_fw_s0 - median_fw_s0
high_fw_md = p75_fw_md - median_fw_md
high_fw_hybrid = p75_fw_hybrid - median_fw_hybrid

# FA
gt_fa = FAs[f_i] * np.ones(nMDs)
fa_s0 = fractional_anisotropy(dti_params_s0[..., 0:3])
fa_md = fractional_anisotropy(dti_params_md[..., 0:3])
fa_hybrid = fractional_anisotropy(dti_params_hybrid[..., 0:3])
median_fa_s0 = np.median(fa_s0[:, f_i, ...], axis=-1)
median_fa_md = np.median(fa_md[:, f_i, ...], axis=-1)
median_fa_hybrid = np.median(fa_hybrid[:, f_i, ...], axis=-1)
p25_fa_s0, p75_fa_s0 = np.percentile(fa_s0[:, f_i, ...], [25, 75], axis=-1)
p25_fa_md, p75_fa_md = np.percentile(fa_md[:, f_i, ...], [25, 75], axis=-1)
p25_fa_hybrid, p75_fa_hybrid = np.percentile(fa_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_fa_s0 = median_fa_s0 - p25_fa_s0
low_fa_md = median_fa_md - p25_fa_md
low_fa_hybrid = median_fa_hybrid - p25_fa_hybrid
high_fa_s0 = p75_fa_s0 - median_fa_s0
high_fa_md = p75_fa_md - median_fa_md
high_fa_hybrid = p75_fa_hybrid - median_fa_hybrid

# MD
md_s0 = mean_diffusivity(dti_params_s0[..., 0:3])
md_md = mean_diffusivity(dti_params_md[..., 0:3])
md_hybrid = mean_diffusivity(dti_params_hybrid[..., 0:3])
median_md_s0 = np.median(md_s0[:, f_i, ...], axis=-1)
median_md_md = np.median(md_md[:, f_i, ...], axis=-1)
median_md_hybrid = np.median(md_hybrid[:, f_i, ...], axis=-1)
p25_md_s0, p75_md_s0 = np.percentile(md_s0[:, f_i, ...], [25, 75], axis=-1)
p25_md_md, p75_md_md = np.percentile(md_md[:, f_i, ...], [25, 75], axis=-1)
p25_md_hybrid, p75_md_hybrid = np.percentile(md_hybrid[:, f_i, ...], [25, 75], axis=-1)
low_md_s0 = median_md_s0 - p25_md_s0
low_md_md = median_md_md - p25_md_md
low_md_hybrid = median_md_hybrid - p25_md_hybrid
high_md_s0 = p75_md_s0 - median_md_s0
high_md_md = p75_md_md - median_md_md
high_md_hybrid = p75_md_hybrid - median_md_hybrid

ax4.errorbar(MDs, median_fw_s0, fmt='.', yerr=[low_fw_s0, high_fw_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax4.errorbar(MDs, median_fw_md, fmt='.', yerr=[low_fw_md, high_fw_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax4.errorbar(MDs, median_fw_hybrid, fmt='.',
              yerr=[low_fw_hybrid, high_fw_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax4.plot(MDs, gt_fw, color=colors[3], label=labels[3], linewidth=lw, ls='--')
ax4.set_title('GT FW = ' + str(FWs[f_i]))

#-------------------------------------------------------------------------------
ax8.errorbar(MDs, median_md_s0, fmt='.', yerr=[low_md_s0, high_md_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax8.errorbar(MDs, median_md_md, fmt='.', yerr=[low_md_md, high_md_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax8.errorbar(MDs, median_md_hybrid, fmt='.',
              yerr=[low_md_hybrid, high_md_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax8.plot(MDs, MDs, color=colors[3], label=labels[3], linewidth=lw, ls='--')

#-------------------------------------------------------------------------------
ax12.errorbar(MDs, median_fa_s0, fmt='.', yerr=[low_fa_s0, high_fa_s0],
              color=colors[0], ecolor=colors[0], linewidth=lw,
              label=labels[0], markersize=msize)

ax12.errorbar(MDs, median_fa_md, fmt='.', yerr=[low_fa_md, high_fa_md],
              color=colors[1], ecolor=colors[1], linewidth=lw,
              label=labels[1], markersize=msize)

ax12.errorbar(MDs, median_fa_hybrid, fmt='.',
              yerr=[low_fa_hybrid, high_fa_hybrid],
              color=colors[2], ecolor=colors[2], linewidth=lw,
              label=labels[2], markersize=msize)

ax12.plot(MDs, gt_fa, color=colors[3], label=labels[3], linewidth=lw, ls='--')

plt.show()

# # ------------------------------------------------------------------------------
# print('Saving...')  # (comment / uncomment)
# # ------------------------------------------------------------------------------
# fout = '/home/mrk/Desktop/'
# fname = 'fig1_final'
# fig.savefig(fout + fname + '.png', format='png', dpi=600, bbox_inches='tight')
# fig.savefig(fout + fname + '.eps', format='eps', dpi=600, bbox_inches='tight')
# print('All done!')
