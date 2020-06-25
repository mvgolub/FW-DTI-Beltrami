from pathlib import Path
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import (TensorModel, fractional_anisotropy,
                              mean_diffusivity)
from dipy.reconst.fwdti import FreeWaterTensorModel
from functions import (simulate_volume, simulate_fw_lesion, simulate_md_lesion,
                       simulate_fw_md_lesion, simulate_volume2, simulate_volume3)
from beltrami import BeltramiModel, BeltramiFit, model_prediction
from matplotlib import pyplot as plt

# modified NLS FW-DTI (that uses the same initialization as Beltrami)
from nls import FreeWaterTensorModel_mod


print('Loading GT parameters...')
cwd = Path.cwd()
parent_dir = cwd.parent
gt_dir = parent_dir / 'data' / 'GT' / 'from_cenir'

# GT parameters
params = nib.load(str(gt_dir / 'nls_params.nii.gz')).get_data()

# GT S0
S0 = nib.load(str(gt_dir / 'S0.nii.gz')).get_data()

# mask
mask = nib.load(str(gt_dir / 'binary_mask.nii.gz')).get_data()
mask = mask.astype(bool)

# simulating acquisition gradients
n_pts = 32      # number of gradient directions per scheme
n_bzeros = 6    # number of simulated bzeros
bval_low = 0.5  # weigthing factor of lower shell
bval_high = 1   # weighting factor for higher shell

theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
directions = hsph_updated.vertices

bvals_1 = bval_high * np.ones(n_pts + n_bzeros)
bvals_1[0:n_bzeros] = 0
bvecs_1 = np.zeros((n_pts + n_bzeros, 3))
bvecs_1[n_bzeros:, :] = directions

bvals_2 = bval_low * np.ones(n_pts)
bvecs_2 = directions
bvals = np.hstack((bvals_1, bvals_2))
bvecs = np.vstack((bvecs_1, bvecs_2))

gtab_single = gradient_table(bvals_1, bvecs_1, b0_threshold=0)
gtab_multi = gradient_table(bvals, bvecs, b0_threshold=0)

# lesion location and size
c = [21, 49, 10]
r = 7

# simulating lesions in GT
params_fwles, _ = simulate_fw_lesion(params, c, r)
params_mdles, lesion_mask = simulate_md_lesion(params, c, r)
lesion_mask = lesion_mask.astype(bool)

# # slicing
# z = 10
# n = 3
# params_fwles = params_fwles[..., 9:12, :]
# params_mdles = params_mdles[..., 9:12, :]
# mask = mask[..., 9:12]
# lesion_mask = lesion_mask[..., 9:12]
# S0 = S0[..., 9:12]

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

print(Sw)
print(St)

print('simulating signal...')
SNR = 40
sig_fwles_single = np.zeros(S0.shape + (gtab_single.bvals.size, ))
sig_mdles_single = np.zeros(S0.shape + (gtab_single.bvals.size, ))
sig_fwles_multi = np.zeros(S0.shape + (gtab_multi.bvals.size, ))
sig_mdles_multi = np.zeros(S0.shape + (gtab_multi.bvals.size, ))

sig_fwles_single[mask, :] = simulate_volume3(params_fwles[mask, :],
                                             gtab_single, S0t=St, S0w=Sw,
                                             snr=SNR, Diso=3)

sig_mdles_single[mask, :] = simulate_volume3(params_mdles[mask, :],
                                             gtab_single, S0t=St, S0w=Sw,
                                             snr=SNR, Diso=3)

sig_fwles_multi[mask, :] = simulate_volume3(params_fwles[mask, :],
                                            gtab_multi, S0t=St, S0w=Sw,
                                            snr=SNR, Diso=3)

sig_mdles_multi[mask, :] = simulate_volume3(params_mdles[mask, :],
                                            gtab_multi, S0t=St, S0w=Sw,
                                            snr=SNR, Diso=3)


print('signal shape (single-shell) = ' + str(sig_fwles_single.shape))
print('signal shape (multi-shell) = ' + str(sig_fwles_multi.shape))


# ground truth
gt_fw_fwles = (1 - params_fwles[..., 12]) * mask
gt_fa_fwles= fractional_anisotropy(params_fwles[..., 0:3]) * mask
gt_md_fwles = mean_diffusivity(params_fwles[..., 0:3]) * mask

gt_fw_mdles = (1 - params_mdles[..., 12]) * mask
gt_fa_mdles= fractional_anisotropy(params_mdles[..., 0:3]) * mask
gt_md_mdles = mean_diffusivity(params_mdles[..., 0:3]) * mask 

St = 400
Sw = 900

print('-----------------------------------------------------------------------')
print('Processing data with FW lesion...')
# ------------------------------------------------------------------------------
print('Running standard DTI (single-shell)...')
# ------------------------------------------------------------------------------
tenmodel = TensorModel(gtab_single)
tenfit = tenmodel.fit(sig_fwles_single)
dti_fa = tenfit.fa * mask
dti_md = tenfit.md * mask
dti_fw = np.zeros(dti_fa.shape)

# ------------------------------------------------------------------------------
print('Running Beltrami (single-shell)...')
# ------------------------------------------------------------------------------
bmodel = BeltramiModel(gtab_single, init_method='hybrid', Stissue=St, Swater=Sw,
                       learning_rate=0.0005, iterations=200)
bfit = bmodel.fit(sig_fwles_single, mask=mask)
belt_fw_single = bfit.fw * mask
belt_fa_single = bfit.fa * mask
belt_md_single = bfit.md * mask

# ------------------------------------------------------------------------------
print('Running Beltrami (multi-shell)...')
# ------------------------------------------------------------------------------
bmodel = BeltramiModel(gtab_multi, init_method='hybrid', Stissue=St, Swater=Sw,
                       learning_rate=0.0005, iterations=200)
bfit = bmodel.fit(sig_fwles_multi, mask=mask)
belt_fw_multi = bfit.fw * mask
belt_fa_multi = bfit.fa * mask
belt_md_multi = bfit.md * mask

belt_fw0_multi = bfit.fw0 * mask
belt_fa0_multi = bfit.fa0 * mask
belt_md0_multi = bfit.md0 * mask

# # ------------------------------------------------------------------------------
# print('Running NLS FW-DTI...')
# # ------------------------------------------------------------------------------
# bvals_scaled = gtab_multi.bvals * 1000
# gtab = gradient_table(bvals_scaled, gtab_multi.bvecs, b0_threshold=0)
# nlsmodel = FreeWaterTensorModel(gtab, fit_method='NLS')
# nlsfit = nlsmodel.fit(sig_fwles_multi, mask=mask)
# nls_fw = nlsfit.f * mask
# nls_fa = nlsfit.fa * mask
# nls_md = 10**3 * nlsfit.md * mask

# ------------------------------------------------------------------------------
print('Running modified NLS FW-DTI...')
# ------------------------------------------------------------------------------
bvals_scaled = gtab_multi.bvals * 1000
gtab = gradient_table(bvals_scaled, gtab_multi.bvecs, b0_threshold=0)
nlsmodel = FreeWaterTensorModel_mod(gtab, Stissue=St, Swater=Sw, fit_method='NLS')
nlsfit = nlsmodel.fit(sig_fwles_multi, mask=mask)
nls_fw = nlsfit.f * mask
nls_fa = nlsfit.fa * mask
nls_md = 10**3 * nlsfit.md * mask

# errors for FW lesion
# FW
belt_fw_sing_fwles_err = belt_fw_single[lesion_mask]
belt_fw_mult_fwles_err = belt_fw_multi[lesion_mask]
nls_fw_fwles_err = nls_fw[lesion_mask]
init_fw_fwles_err = belt_fw0_multi[lesion_mask]

# MD
belt_md_sing_fwles_err = belt_md_single[lesion_mask]
belt_md_mult_fwles_err = belt_md_multi[lesion_mask]
nls_md_fwles_err = nls_md[lesion_mask]
init_md_fwles_err = belt_md0_multi[lesion_mask]

# plt.figure('MD error historgrams', figsize=(5.9, 5.9))

# plt.subplot(221)
# plt.hist(gt_md_fwles[lesion_mask], bins=30)
# plt.title('gt')
# plt.ylabel('# of counts')

# plt.subplot(222)
# plt.hist(belt_md_single[lesion_mask], bins=30)
# plt.title('Beltrami single')
# plt.ylabel('# of counts')

# plt.subplot(223)
# plt.hist(belt_md_multi[lesion_mask], bins=30)
# plt.title('Beltrami multi')
# plt.xlabel('MD error in lesion mask')
# plt.ylabel('# of counts')

# plt.subplot(224)
# plt.hist(nls_md[lesion_mask], bins=30)
# plt.title('NLS')
# plt.xlabel('MD error in lesion mask')
# plt.ylabel('# of counts')

# plt.show()

# plt.figure()
# plt.imshow(gt_md_fwles[:, :, 10].T * lesion_mask[:, :, 10].T, origin='lower', cmap='gray', vmin=0, vmax=3)
# plt.show()

# FA
belt_fa_sing_fwles_err = belt_fa_single[lesion_mask]
belt_fa_mult_fwles_err = belt_fa_multi[lesion_mask]
nls_fa_fwles_err = nls_fa[lesion_mask]
init_fa_fwles_err = belt_fa0_multi[lesion_mask]

print('-----------------------------------------------------------------------')
print('Processing data with MD lesion...')
# ------------------------------------------------------------------------------
print('Running standard DTI (single-shell)...')
# ------------------------------------------------------------------------------
tenmodel = TensorModel(gtab_single)
tenfit = tenmodel.fit(sig_mdles_single)
dti_fa = tenfit.fa * mask
dti_md = tenfit.md * mask
dti_fw = np.zeros(dti_fa.shape)

# ------------------------------------------------------------------------------
print('Running Beltrami (single-shell)...')
# ------------------------------------------------------------------------------
bmodel = BeltramiModel(gtab_single, init_method='hybrid', Stissue=St, Swater=Sw,
                       learning_rate=0.0005, iterations=200)
bfit = bmodel.fit(sig_mdles_single, mask=mask)
belt_fw_single = bfit.fw * mask
belt_fa_single = bfit.fa * mask
belt_md_single = bfit.md * mask

# ------------------------------------------------------------------------------
print('Running Beltrami (multi-shell)...')
# ------------------------------------------------------------------------------
bmodel = BeltramiModel(gtab_multi, init_method='hybrid', Stissue=St, Swater=Sw,
                       learning_rate=0.0005, iterations=200)
bfit = bmodel.fit(sig_mdles_multi, mask=mask)
belt_fw_multi = bfit.fw * mask
belt_fa_multi = bfit.fa * mask
belt_md_multi = bfit.md * mask

belt_fw0_multi = bfit.fw0 * mask
belt_fa0_multi = bfit.fa0 * mask
belt_md0_multi = bfit.md0 * mask

# # ------------------------------------------------------------------------------
# print('Running NLS FW-DTI...')
# # ------------------------------------------------------------------------------
# bvals_scaled = gtab_multi.bvals * 1000
# gtab = gradient_table(bvals_scaled, gtab_multi.bvecs, b0_threshold=0)
# nlsmodel = FreeWaterTensorModel(gtab, fit_method='NLS')
# nlsfit = nlsmodel.fit(sig_mdles_multi, mask=mask)
# nls_fw = nlsfit.f * mask
# nls_fa = nlsfit.fa * mask
# nls_md = 10**3 * nlsfit.md * mask

# ------------------------------------------------------------------------------
print('Running modified NLS FW-DTI...')
# ------------------------------------------------------------------------------
bvals_scaled = gtab_multi.bvals * 1000
gtab = gradient_table(bvals_scaled, gtab_multi.bvecs, b0_threshold=0)
nlsmodel = FreeWaterTensorModel_mod(gtab, Stissue=St, Swater=Sw, fit_method='NLS')
nlsfit = nlsmodel.fit(sig_mdles_multi, mask=mask)
nls_fw = nlsfit.f * mask
nls_fa = nlsfit.fa * mask
nls_md = 10**3 * nlsfit.md * mask

# errors for MD lesion
# FW
belt_fw_sing_mdles_err = belt_fw_single[lesion_mask]
belt_fw_mult_mdles_err = belt_fw_multi[lesion_mask]
nls_fw_mdles_err = nls_fw[lesion_mask]
init_fw_mdles_err = belt_fw0_multi[lesion_mask]

# MD
belt_md_sing_mdles_err = belt_md_single[lesion_mask]
belt_md_mult_mdles_err = belt_md_multi[lesion_mask]
nls_md_mdles_err = nls_md[lesion_mask]
init_md_mdles_err = belt_md0_multi[lesion_mask]

# FA
belt_fa_sing_mdles_err = belt_fa_single[lesion_mask]
belt_fa_mult_mdles_err = belt_fa_multi[lesion_mask]
nls_fa_mdles_err = nls_fa[lesion_mask]
init_fa_mdles_err = belt_fa0_multi[lesion_mask]

print('Plotting...')

fig = plt.figure(figsize=(6.9, 5.9))
gs3 = fig.add_gridspec(ncols=2, nrows=3, left=0.12, right=0.80)

# axes
ax1 = fig.add_subplot(gs3[0, 0])
ax2 = fig.add_subplot(gs3[0, 1])
ax3 = fig.add_subplot(gs3[1, 0])
ax4 = fig.add_subplot(gs3[1, 1])
ax5 = fig.add_subplot(gs3[2, 0])
ax6 = fig.add_subplot(gs3[2, 1])

# column titles
fs = 10
ax1.set_title('Data with FW lesion', fontsize=fs)
ax2.set_title('Data with MD lesion', fontsize=fs)

# ylablels
ax1.set_ylabel('FW', fontsize=fs)
ax3.set_ylabel('MD\n'+  r'$[\mu m^2 ms^{-1}]$', fontsize=fs)
ax5.set_ylabel('FA', fontsize=fs)

# removing ticks
all_axes = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in all_axes:
    ax.set_xticks([])

colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'c', 4: 'darkorange'}
labels = {0: 'initial guess', 1: 'RGD FWE\n(single-shell)',
          2: 'RGD FWE\n(multi-shell)', 3: 'NLS FWE*\n(multi-shell)', 4: 'GT'}

# FW errors for FW lesion
x = 2
y = np.median(gt_fw_fwles[lesion_mask])
p25, p75 = np.percentile(gt_fw_fwles[lesion_mask], [25, 75])
low = y - p25
high = p75 - y
ax1.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[4], label=labels[4], capsize=3, ms=7)

x = 4
y = np.median(init_fw_fwles_err)
p25, p75 = np.percentile(init_fw_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax1.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[0], label=labels[0], capsize=3, ms=7)

x = 6
y = np.median(belt_fw_sing_fwles_err)
p25, p75 = np.percentile(belt_fw_sing_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax1.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[1], label=labels[1], capsize=3, ms=7)

x = 8
y = np.median(belt_fw_mult_fwles_err)
p25, p75 = np.percentile(belt_fw_mult_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax1.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[2], label=labels[2], capsize=3, ms=7)

x = 10
y = np.median(nls_fw_fwles_err)
p25, p75 = np.percentile(nls_fw_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax1.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[3], label=labels[3], capsize=3, ms=7)

x = np.arange(0, 13)
y = np.median(gt_fw_fwles[lesion_mask]) * np.ones(x.shape)
ax1.plot(x, y, color='gray', linestyle='dashed', ms=7)


# FW errors for MD lesion
x = 2
y = np.median(gt_fw_mdles[lesion_mask])
p25, p75 = np.percentile(gt_fw_mdles[lesion_mask], [25, 75])
low = y - p25
high = p75 - y
ax2.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[4], label=labels[4], capsize=3, ms=7)

x = 4
y = np.median(init_fw_mdles_err)
p25, p75 = np.percentile(init_fw_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax2.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[0], label=labels[0], capsize=3, ms=7)

x = 6
y = np.median(belt_fw_sing_mdles_err)
p25, p75 = np.percentile(belt_fw_sing_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax2.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[1], label=labels[1], capsize=3, ms=7)

x = 8
y = np.median(belt_fw_mult_mdles_err)
p25, p75 = np.percentile(belt_fw_mult_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax2.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[2], label=labels[2], capsize=3, ms=7)

x = 10
y = np.median(nls_fw_mdles_err)
p25, p75 = np.percentile(nls_fw_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax2.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[3], label=labels[3], capsize=3, ms=7)

x = np.arange(0, 13)
y = np.median(gt_fw_mdles[lesion_mask]) * np.ones(x.shape)
ax2.plot(x, y, color='gray', linestyle='dashed', ms=7)

p25, p75 = np.percentile(gt_fw_mdles[lesion_mask], [25, 75])
ax2.fill_between(x, p25, p75, alpha=0.2, color='gray')

ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.58), prop={'size': 8},
            frameon=True)

# MD errors for FW lesion
x = 2
y = np.median(gt_md_fwles[lesion_mask])
p25, p75 = np.percentile(gt_md_fwles[lesion_mask], [25, 75])
low = y - p25
high = p75 - y
ax3.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[4], label=labels[4], capsize=3, ms=7)

x = 4
y = np.median(init_md_fwles_err)
p25, p75 = np.percentile(init_md_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax3.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[0], label=labels[0], capsize=3, ms=7)

x = 6
y = np.median(belt_md_sing_fwles_err)
p25, p75 = np.percentile(belt_md_sing_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax3.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[1], label=labels[1], capsize=3, ms=7)

x = 8
y = np.median(belt_md_mult_fwles_err)
p25, p75 = np.percentile(belt_md_mult_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax3.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[2], label=labels[2], capsize=3, ms=7)

x = 10
y = np.median(nls_md_fwles_err)
p25, p75 = np.percentile(nls_md_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax3.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[3], label=labels[3], capsize=3, ms=7)

x = np.arange(0, 13)
y = np.median(gt_md_fwles[lesion_mask]) * np.ones(x.shape)
ax3.plot(x, y, color='gray', linestyle='dashed', ms=7)

p25, p75 = np.percentile(gt_md_fwles[lesion_mask], [25, 75])
ax3.fill_between(x, p25, p75, alpha=0.2, color='gray')

# MD errors for MD lesion
x = 2
y = np.median(gt_md_mdles[lesion_mask])
p25, p75 = np.percentile(gt_md_mdles[lesion_mask], [25, 75])
low = y - p25
high = p75 - y
ax4.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[4], label=labels[4], capsize=3, ms=7)

x = 4
y = np.median(init_md_mdles_err)
p25, p75 = np.percentile(init_md_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax4.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[0], label=labels[0], capsize=3, ms=7)

x = 6
y = np.median(belt_md_sing_mdles_err)
p25, p75 = np.percentile(belt_md_sing_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax4.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[1], label=labels[1], capsize=3, ms=7)

x = 8
y = np.median(belt_md_mult_mdles_err)
p25, p75 = np.percentile(belt_md_mult_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax4.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[2], label=labels[2], capsize=3, ms=7)

x = 10
y = np.median(nls_md_mdles_err)
p25, p75 = np.percentile(nls_md_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax4.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[3], label=labels[3], capsize=3, ms=7)

x = np.arange(0, 13)
y = np.median(gt_md_mdles[lesion_mask]) * np.ones(x.shape)
ax4.plot(x, y, color='gray', linestyle='dashed', ms=7)


# FA errors for FW lesion
x = 2
y = np.median(gt_fa_fwles[lesion_mask])
p25, p75 = np.percentile(gt_fa_fwles[lesion_mask], [25, 75])
low = y - p25
high = p75 - y
ax5.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[4], label=labels[4], capsize=3, ms=7)

x = 4
y = np.median(init_fa_fwles_err)
p25, p75 = np.percentile(init_fa_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax5.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[0], label=labels[0], capsize=3, ms=7)

x = 6
y = np.median(belt_fa_sing_fwles_err)
p25, p75 = np.percentile(belt_fa_sing_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax5.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[1], label=labels[1], capsize=3, ms=7)

x = 8
y = np.median(belt_fa_mult_fwles_err)
p25, p75 = np.percentile(belt_fa_mult_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax5.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[2], label=labels[2], capsize=3, ms=7)

x = 10
y = np.median(nls_fa_fwles_err)
p25, p75 = np.percentile(nls_fa_fwles_err, [25, 75])
low = y - p25
high = p75 - y
ax5.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[3], label=labels[3], capsize=3, ms=7)

x = np.arange(0, 13)
y = np.median(gt_fa_fwles[lesion_mask]) * np.ones(x.shape)
ax5.plot(x, y, color='gray', linestyle='dashed', ms=7)

p25, p75 = np.percentile(gt_fa_fwles[lesion_mask], [25, 75])
ax5.fill_between(x, p25, p75, alpha=0.2, color='gray')


# FA errors for MD lesion
x = 2
y = np.median(gt_fa_mdles[lesion_mask])
p25, p75 = np.percentile(gt_fa_mdles[lesion_mask], [25, 75])
low = y - p25
high = p75 - y
ax6.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[4], label=labels[4], capsize=3, ms=7)

x = 4
y = np.median(init_fa_mdles_err)
p25, p75 = np.percentile(init_fa_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax6.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[0], label=labels[0], capsize=3, ms=7)

x = 6
y = np.median(belt_fa_sing_mdles_err)
p25, p75 = np.percentile(belt_fa_sing_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax6.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[1], label=labels[1], capsize=3, ms=7)

x = 8
y = np.median(belt_fa_mult_mdles_err)
p25, p75 = np.percentile(belt_fa_mult_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax6.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[2], label=labels[2], capsize=3, ms=7)

x = 10
y = np.median(nls_fa_mdles_err)
p25, p75 = np.percentile(nls_fa_mdles_err, [25, 75])
low = y - p25
high = p75 - y
ax6.errorbar(x, y, fmt='.', yerr=np.array([[low, high]]).T, color=colors[3], label=labels[3], capsize=3, ms=7)

x = np.arange(0, 13)
y = np.median(gt_fa_mdles[lesion_mask]) * np.ones(x.shape)
ax6.plot(x, y, color='gray', linestyle='dashed', ms=7)

p25, p75 = np.percentile(gt_fa_mdles[lesion_mask], [25, 75])
ax6.fill_between(x, p25, p75, alpha=0.2, color='gray')

# lines = [l1, l2, l3, l4, l5]
# fig.legend(lines, list(labels.values()), loc='center left',
#             bbox_to_anchor=(0.80, 0.69), prop={'size': 9}, frameon=True)

ax1.set_xlim([0, 12])
ax2.set_xlim([0, 12])
ax3.set_xlim([0, 12])
ax4.set_xlim([0, 12])
ax5.set_xlim([0, 12])
ax6.set_xlim([0, 12])

ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1.3])
ax4.set_ylim([0, 1.3])
ax5.set_ylim([0, 1])
ax6.set_ylim([0, 1])

fwticks = [0.1, 0.5, 0.9]
mdticks = [0.2, 0.6, 1.1]
# mdticks = np.linspace(0.1, 1.2, num=3)
ax1.set_yticks(np.round([np.median(gt_fw_mdles[lesion_mask]), np.median(gt_fw_fwles[lesion_mask]), 0.9], 2))
ax2.set_yticks(np.round([np.median(gt_fw_mdles[lesion_mask]), np.median(gt_fw_fwles[lesion_mask]), 0.9], 2))
ax3.set_yticks(np.round([0.2, np.median(gt_md_fwles[lesion_mask]), 1.1], 2))
ax4.set_yticks(np.round([0.2, np.median(gt_md_fwles[lesion_mask]) , 1.1], 2))
ax5.set_yticks(np.round([0.1, np.median(gt_fa_fwles[lesion_mask]), 0.9], 2))
ax6.set_yticks(np.round([0.1, np.median(gt_fa_mdles[lesion_mask]), 0.9], 2))

plt.show()

# ------------------------------------------------------------------------------
print('Saving...')  # (comment / uncomment)
# ------------------------------------------------------------------------------
fout = '/home/mrk/Desktop/'
fname = 'fig7_final'
fig.savefig(fout + fname + '.png', format='png', dpi=600, bbox_inches='tight')
fig.savefig(fout + fname + '.eps', format='eps', dpi=600, bbox_inches='tight')
print('All done!')
