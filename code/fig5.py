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
affine = nib.load(str(gt_dir / 'nls_params.nii.gz')).affine

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
params_mdles, _ = simulate_md_lesion(params, c, r)

# # slicing
# z = 10
# n = 3
# params_fwles = params_fwles[:, :, 8:12, :]
# params_mdles = params_mdles[:, :, 8:12, :]
# mask = mask[..., 8:12]
# S0 = S0[..., 8:12]


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

print('Simulated S0w and S0t:')
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


# ------------------------------------------------------------------------------
# saving signal with lesion (single-shell) # to run with matlab version and compare
out_FW = parent_dir / 'matlab' / 'FW_lesion'
out_MD = parent_dir / 'matlab' / 'MD_lesion'
fwles_img = nib.Nifti1Image(sig_fwles_single.astype(np.float64), affine)
nib.save(fwles_img, str(out_FW / 'dwi.nii'))
mdles_img = nib.Nifti1Image(sig_mdles_single.astype(np.float64), affine)
nib.save(mdles_img, str(out_MD / 'dwi.nii'))
# saving mask
mask_img = nib.Nifti1Image(mask.astype(np.float32), affine)
nib.save(mask_img, str(out_FW / 'binary_mask.nii'))
nib.save(mask_img, str(out_MD / 'binary_mask.nii'))
# saving bvals and bvecs
bvals_tmp = gtab_single.bvals * 1000
bvecs_tmp = gtab_single.bvecs
np.savetxt(str(out_FW / 'bvals'), bvals_tmp)
np.savetxt(str(out_FW / 'bvecs'), bvecs_tmp)
np.savetxt(str(out_MD / 'bvals'), bvals_tmp)
np.savetxt(str(out_MD / 'bvecs'), bvecs_tmp)
# ------------------------------------------------------------------------------


im = sig_fwles_single

plt.figure()
plt.subplot(131)
plt.imshow(im[..., 8, 0].T, origin='lower', cmap='gray')

plt.subplot(132)
plt.imshow(im[..., 8, 16].T, origin='lower', cmap='gray')

plt.subplot(133)
plt.hist(im[mask, 0], bins=30)

plt.show()

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

# ------------------------------------------------------------------------------
print('Running NLS FW-DTI...')
# ------------------------------------------------------------------------------
bvals_scaled = gtab_multi.bvals * 1000
gtab = gradient_table(bvals_scaled, gtab_multi.bvecs, b0_threshold=0)
nlsmodel = FreeWaterTensorModel(gtab, fit_method='NLS')
nlsfit = nlsmodel.fit(sig_fwles_multi, mask=mask)
nls_fw = nlsfit.f * mask
nls_fa = nlsfit.fa * mask
nls_md = 10**3 * nlsfit.md * mask

# ------------------------------------------------------------------------------
print('Running modified NLS FW-DTI...')
# ------------------------------------------------------------------------------
nlsmodel = FreeWaterTensorModel_mod(gtab, Stissue=St, Swater=Sw, fit_method='NLS')
nlsfit = nlsmodel.fit(sig_fwles_multi, mask=mask)
nlsmod_fw = nlsfit.f * mask
nlsmod_fa = nlsfit.fa * mask
nlsmod_md = 10**3 * nlsfit.md * mask

print('Plotting...')
fig1 = plt.figure(figsize=(6.83, 4.3))
gs1 = fig1.add_gridspec(ncols=6, nrows=3, hspace=0.0, wspace=0.0)

# axes
ax1 = fig1.add_subplot(gs1[0, 0])
ax2 = fig1.add_subplot(gs1[0, 1])
ax3 = fig1.add_subplot(gs1[0, 2])
ax4 = fig1.add_subplot(gs1[0, 3])
ax5 = fig1.add_subplot(gs1[0, 4])
ax6 = fig1.add_subplot(gs1[0, 5])
ax7 = fig1.add_subplot(gs1[1, 0])
ax8 = fig1.add_subplot(gs1[1, 1])
ax9 = fig1.add_subplot(gs1[1, 2])
ax10 = fig1.add_subplot(gs1[1, 3])
ax11 = fig1.add_subplot(gs1[1, 4])
ax12 = fig1.add_subplot(gs1[1, 5])
ax13 = fig1.add_subplot(gs1[2, 0])
ax14 = fig1.add_subplot(gs1[2, 1])
ax15 = fig1.add_subplot(gs1[2, 2])
ax16 = fig1.add_subplot(gs1[2, 3])
ax17 = fig1.add_subplot(gs1[2, 4])
ax18 = fig1.add_subplot(gs1[2, 5])

ax1.imshow(gt_fw_fwles[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax2.imshow(dti_fw[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax3.imshow(belt_fw_single[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax4.imshow(belt_fw_multi[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax5.imshow(nlsmod_fw[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax6.imshow(nls_fw[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)

ax7.imshow(gt_md_fwles[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=3)
ax8.imshow(dti_md[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=3)
ax9.imshow(belt_md_single[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=3)
ax10.imshow(belt_md_multi[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=3)
ax11.imshow(nlsmod_md[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=3)
ax12.imshow(nls_md[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=3)

ax13.imshow(gt_fa_fwles[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax14.imshow(dti_fa[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax15.imshow(belt_fa_single[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax16.imshow(belt_fa_multi[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax17.imshow(nlsmod_fa[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax18.imshow(nls_fa[:, :, 8].T, origin='lower', cmap='gray', vmin=0, vmax=1)

# column titles
fs = 9

ax1.set_title('GT', fontsize=fs)
ax2.set_title('DTI\n(single-shell)', fontsize=fs)
ax3.set_title('RGD FWE\n(single-shell)', fontsize=fs)
ax4.set_title('RGD FWE\n(multi-shell) ', fontsize=fs)
ax5.set_title('NLS FWE*\n(multi-shell)', fontsize=fs)
ax6.set_title('NLS FWE\n(multi-shell)', fontsize=fs)

# ylablels
ax1.set_ylabel('FW', fontsize=fs)
ax7.set_ylabel('MD', fontsize=fs)
ax13.set_ylabel('FA', fontsize=fs)

# removing ticks
all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13,
            ax14, ax15, ax16, ax17, ax18]

for ax in all_axes:
    ax.set_xticks([])
    ax.set_yticks([])

# ax3.text(0.12, 1.42, 'FW lesion', transform=ax3.transAxes, bbox={'facecolor': 'none'})
# ax13.text(0.12, 1.42, 'MD lesion', transform=ax13.transAxes, bbox={'facecolor': 'none'})

ax3.annotate('', fontsize=fs, xy=(21, 49),
                    color='cyan',
                    xycoords='data', xytext=(25, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='cyan'))

ax4.annotate('', fontsize=fs, xy=(21, 49),
                    color='cyan',
                    xycoords='data', xytext=(25, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='cyan'))

ax5.annotate('', fontsize=fs, xy=(21, 49),
                    color='cyan',
                    xycoords='data', xytext=(25, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='cyan'))

ax6.annotate('', fontsize=fs, xy=(21, 49),
                    color='cyan',
                    xycoords='data', xytext=(25, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='cyan'))

ax8.annotate('', fontsize=fs, xy=(21, 49),
                    color='red',
                    xycoords='data', xytext=(25, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))

ax9.annotate('', fontsize=fs, xy=(21, 49),
                    color='red',
                    xycoords='data', xytext=(25, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))

# ax14.annotate('FP', fontsize=fs, xy=(21, 49),
#                     color='red',
#                     xycoords='data', xytext=(25, 0),
#                     textcoords='offset points',
#                     arrowprops=dict(arrowstyle="->",
#                                     color='red'))

plt.show()

# ------------------------------------------------------------------------------
print('Saving...')  # (comment / uncomment)
# ------------------------------------------------------------------------------
fout = '/home/mrk/Desktop/'
fname = 'fig5_final'
fig1.savefig(fout + fname + '.png', format='png', dpi=600, bbox_inches='tight')
fig1.savefig(fout + fname + '.eps', format='eps', dpi=600, bbox_inches='tight')
print('All done!')

# metrics_dir = parent_dir / 'data' / 'CENIR' / 'metrics' / 'FW-lesion'

# # ground truth
# img = nib.Nifti1Image(gt_fw_fwles, affine)
# nib.save(img, str(metrics_dir / 'FW_gt.nii.gz'))

# img = nib.Nifti1Image(gt_fa_fwles, affine)
# nib.save(img, str(metrics_dir / 'FA_gt.nii.gz'))

# img = nib.Nifti1Image(gt_md_fwles, affine)
# nib.save(img, str(metrics_dir / 'MD_gt.nii.gz'))

# # estimated with beltrami multi-shell
# img = nib.Nifti1Image(belt_fw_multi, affine)
# nib.save(img, str(metrics_dir / 'FW_belt_multi.nii.gz'))

# img = nib.Nifti1Image(belt_fa_multi, affine)
# nib.save(img, str(metrics_dir / 'FA_belt_multi.nii.gz'))

# img = nib.Nifti1Image(belt_md_multi, affine)
# nib.save(img, str(metrics_dir / 'MD_belt_multi.nii.gz'))

# # estimated with NLS*
# img = nib.Nifti1Image(nlsmod_fw, affine)
# nib.save(img, str(metrics_dir / 'FW_nlsmod.nii.gz'))

# img = nib.Nifti1Image(nlsmod_fa, affine)
# nib.save(img, str(metrics_dir / 'FA_nlsmod.nii.gz'))

# img = nib.Nifti1Image(nlsmod_md, affine)
# nib.save(img, str(metrics_dir / 'MD_nlsmod.nii.gz'))

print('All done!')
