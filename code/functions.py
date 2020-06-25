import numpy as np
from matplotlib import pyplot as plt

from dipy.core.ndindex import ndindex
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table

from dipy.reconst.dki import _positive_evals
from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.reconst.dti import (design_matrix, lower_triangular,
                              fractional_anisotropy, mean_diffusivity) 

from dipy.sims.voxel import (_check_directions, single_tensor, multi_tensor, 
                             all_tensor_evecs, add_noise)

from beltrami import model_prediction


def generate_eigvalues(trace):
    """
    generates random eigenvalues whose sum is equal to trace
    """

    weight_1 = np.random.uniform(0.001, 0.3)
    weight_2 = np.random.uniform(0.3, 0.6)

    L1 = trace * weight_1
    L2 = trace * weight_2
    L3 = trace - (L1 + L2)

    eigvals = np.sort(np.vstack((L1, L2, L3)), axis=0)

    return np.flip(eigvals, axis=0).T


def dual_tensor(gtab, mevals, S0t=50, S0w=100, angles=[(0, 0), (90, 0)],
                fractions=[50, 50], snr=None):
    """
    custom fucntion similar to Dipy's 'sims.voxel.multi_tensor(...)', that
    simulates a dual tensor (free water model), modified to assume that the
    non-weihghted signal S0 is differnet for each compartment (S0t and S0w)
    """

    if np.round(np.sum(fractions), 2) != 100.0:
        raise ValueError('Fractions should sum to 100')

    fractions = [f / 100. for f in fractions]

    sticks = _check_directions(angles)

    S_tissue = fractions[0] * single_tensor(gtab, S0=S0t, evals=mevals[0],
                                            evecs=all_tensor_evecs(sticks[0]),
                                            snr=None)

    S_water = fractions[1] * single_tensor(gtab, S0=S0w, evals=mevals[1],
                                           evecs=all_tensor_evecs(sticks[1]),
                                           snr=None)

    S = S_tissue + S_water

    return add_noise(S, snr, S0w), sticks


def generate_phantom(gtab, S0t=50, S0w=100, snr=None, dir_sigma=None):
    """
    generates a (11 x 11 x 11 x K) dw-MRI dataset, using the free water model, to
    simulated a rectangular fiber immersed in water, dir_sigma defines the
    degree of angular dispersion of the principal eigenvectors that compose the
    fiber i.e. small sigmas result in a smooth tensor field (well aligned fiber)
    large sigmas result in random tensor field.
    """
    def circular_mask(rows, cols, center, radius):
        X, Y = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    phantom = np.zeros((21, 21, 21, gtab.bvals.size))

    # Defining the principal eigenvectors (in spherical coordinates)
    # of the tensor field, by default all tensors are perfectly aligned with
    # the z axis (i.e vertical well aligned fiber)
    theta = 0 * np.ones(phantom.shape[:-1])
    phi = 90 * np.ones(phantom.shape[:-1])

    if dir_sigma is not None:
        # Noisy principal directions
        theta = np.random.normal(0, dir_sigma, size=phantom.shape[:-1])
        phi = np.random.normal(90, dir_sigma, size=phantom.shape[:-1])

    # The ground truth eigenvalues are the same for every voxel 
    L1 = 1.6 
    L2 = 0.5
    L3 = 0.3
    Dw = 3
    mevals = np.array([[L1, L2, L3],
                       [Dw, Dw, Dw]])

    # # Defining indices of the tissue volume fraction, voxels with the same
    # # "F_inds" will have the same fraction volume "F" assigned to them,
    # # e.g where F_ind == 0, corresponds to outside the fiber with F = 0.1
    # F_inds = np.zeros((phantom.shape[:-1]))
    # # voxels in the center of the fiber:
    # F_inds[4:7, 4:7, :] = 1
    # F_inds[5, 5, :] = 2
    # # voxels in the edges of the fiber:
    # F_inds[[3, 5, 5, 7], [5, 3, 7, 5], :] = 3
    # F_inds[[3, 3, 4, 4, 6, 6, 7, 7], [4, 6, 3, 7, 3, 7, 4, 6], :] = 4
    # # voxels in the corners of the fiber
    # F_inds[[3, 3, 7, 7], [3, 7, 3, 7], :] = 5

    # # The tissue volume fraction itself is defined as follows:
    # # - voxels that are completly inside the fiber (F_inds = 1)
    # #   have high tissue fraction
    # # - voxels that are on the edges of the fiber suffer some FW partial volume
    # #   effects, but still retain half of the tissue signal
    # # - corner voxels are almost completley supressed by FW partial volume, and
    # #   have small tissue contribution
    # # - all remaining voxels are outside the fiber and have F = 0.1
    # F = 0.1 * np.ones(phantom.shape[:-1])
    # F[F_inds == 1] = 0.74  # center voxels
    # F[F_inds == 2] = 0.9   
    # F[F_inds == 3] = 0.58  # edge voxels  
    # F[F_inds == 4] = 0.42
    # F[F_inds == 5] = 0.26  # corner voxels
    
    nx = phantom.shape[0]
    ny = phantom.shape[1]
    c = [np.round(nx / 2), np.round(ny / 2)] 
    r1 = 2
    r2 = 6
    r3 = 7
    mask1 = circular_mask(nx, ny, c, r1)
    mask2 = circular_mask(nx, ny, c, r2)
    mask3 = circular_mask(nx, ny, c, r3)
    np.logical_xor(mask1, mask2, out=mask2)
    np.logical_xor(np.logical_or(mask1, mask2), mask3, out=mask3)
    
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(1 * mask1)
    # plt.subplot(132)
    # plt.imshow(1 * mask2)
    # plt.subplot(133)
    # plt.imshow(1 * mask3)
    # plt.show()

    F = 0.1 * np.ones(phantom.shape[:-1])
    F[mask1, :] = 0.9
    F[mask2, :] = 0.6
    F[mask3, :] = 0.3

    F_cor = F * S0w / (F * S0w + (1 - F) * S0t)

    # Simulating the signal for each voxel (looping through the volume)
    # matrices that hold the ground truth eigenvalues and eigenvectors of the
    # tissue compartment
    evals = np.zeros(phantom.shape[:-1] + (3, ))
    evals[..., 0] = L1
    evals[..., 1] = L2
    evals[..., 2] = L3

    evecs = np.zeros(phantom.shape[:-1] + (3, 3))

    for index in np.ndindex(phantom.shape[:-1]):

        # simulated tensor direction, both tensors (FW and tissue) have the
        # same direction
        direction = (theta[index], phi[index])
        angles = [direction, direction]

        # Tisssue volume fraction
        f = F_cor[index]
        fractions = [100 * f, 100 * (1 - f)]

        # Simulated signal
        signal, sticks = dual_tensor(gtab, mevals, S0t=S0t, S0w=S0w,
                                    angles=angles, fractions=fractions,
                                    snr=snr)
        phantom[index + (slice(None), )] = signal

        # Ground truth evecs
        evecs[index + (slice(None), )] = all_tensor_evecs(sticks[0])

    fiber_mask = np.zeros(phantom.shape[:-1]).astype(bool)
    mask = mask1 + mask2 + mask3
    fiber_mask[mask, :] = True
    # Return ground truth parameters and the phantom 
    return (evals, evecs, F, phantom, fiber_mask)


def simulate_volume(model_params, gtab, S0t=50, S0w=100, Diso=3, snr=None):
    evals = model_params[..., :3]
    direcs = model_params[..., 3:6]  # principal eigen vector
    F = model_params[..., 12]
    F_cor = F * S0w / (F * S0w + (1 - F) * S0t)  # corrected fractions
    volume = np.zeros(model_params.shape[:-1] + (gtab.bvals.size, )) 
    for index in ndindex(model_params.shape[:-1]):
        L1 = evals[index + (0, )]
        L2 = evals[index + (1, )]
        L3 = evals[index + (2, )]
        mevals = np.array([[L1, L2, L3],
                           [Diso, Diso, Diso]])
        x, y, z = direcs[index + (slice(None), )]
        r, theta, phi = cart2sphere(x, y, z)
        direction = (theta, phi)
        angles = [direction, direction]
        f = F_cor[index]
        fractions = [100 * f, 100 * (1 - f)]
        signal, sticks = dual_tensor(gtab, mevals, S0t=S0t, S0w=S0w,
                                    angles=angles, fractions=fractions,
                                    snr=snr)
        volume[index + (slice(None), )] = signal
    return volume


def simulate_volume2(model_params, gtab, S0=1, Diso=3, snr=None):
    evals = model_params[..., :3]
    evecs = model_params[..., 3:12].reshape(model_params.shape[:-1] + (3, 3))
    fraction = model_params[..., 12][..., None]
    qform = vec_val_vect(evecs, evals)
    lower_tissue = lower_triangular(qform, 1)
    lower_water = np.copy(lower_tissue)
    lower_water[..., 0] = Diso
    lower_water[..., 1] = 0
    lower_water[..., 2] = Diso
    lower_water[..., 3] = 0
    lower_water[..., 4] = 0
    lower_water[..., 5] = Diso
    H = design_matrix(gtab)
    Stissue = fraction * np.exp(np.einsum('...j,ij->...i', lower_tissue, H))
    Swater = (1 - fraction) * np.exp(np.einsum('...j,ij->...i', lower_water, H))
    # mask = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    signal = (Stissue + Swater) * S0[..., None] #* mask[..., None]
    signal[..., gtab.b0s_mask] = S0[..., None]
    return add_noise(signal, snr, np.mean(S0))


def simulate_volume3(model_params, gtab, S0t=50, S0w=100, Diso=3, snr=None):
    evals = model_params[..., :3]
    evecs = model_params[..., 3:12].reshape(model_params.shape[:-1] + (3, 3))
    fraction = model_params[..., 12][..., None]
    F_cor = fraction * S0w / (fraction * S0w + (1 - fraction) * S0t)  # corrected fractions
    qform = vec_val_vect(evecs, evals)
    lower_tissue = lower_triangular(qform, 1)
    lower_water = np.copy(lower_tissue)
    lower_water[..., 0] = Diso
    lower_water[..., 1] = 0
    lower_water[..., 2] = Diso
    lower_water[..., 3] = 0
    lower_water[..., 4] = 0
    lower_water[..., 5] = Diso
    H = design_matrix(gtab)
    Stissue = S0t * F_cor * np.exp(np.einsum('...j,ij->...i', lower_tissue, H))
    Swater = S0w * (1 - F_cor) * np.exp(np.einsum('...j,ij->...i', lower_water, H))
    signal = Stissue + Swater
    return add_noise(signal, snr, S0w)


def simulate_fw_lesion(model_params,  center, radius):
    x, y, z = model_params.shape[:-1]
    X, Y, Z = np.ogrid[:x, :y, :z]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    mask = dist_from_center <= radius
    fw = 0.6
    out = np.copy(model_params)
    out[mask, 12] = 1 - fw
    return out, mask


def simulate_md_lesion(model_params,  center, radius):
    x, y, z = model_params.shape[:-1]
    X, Y, Z = np.ogrid[:x, :y, :z]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    mask = dist_from_center <= radius
    # mask[np.any(model_params[..., 0:3] <= 0, axis=-1)] = False
    # model_params[np.any(model_params[..., 0:3] <= 0, axis=-1), :-1] = 0.1

    md = 1.1
    gt_md = mean_diffusivity(model_params[mask, 0:3])
    ratio = md / gt_md
    out = np.copy(model_params)
    out[mask, 0:3] *= ratio[..., None]
    return out, mask


def simulate_fw_md_lesion(model_params,  center, radius):
    x, y, z = model_params.shape[:-1]
    X, Y, Z = np.ogrid[:x, :y, :z]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    mask = dist_from_center <= radius
    mask[np.any(model_params[..., 0:3] <= 0, axis=-1)] = False

    fw = 0.65
    md = 1.5
    gt_md = mean_diffusivity(model_params[mask, 0:3])
    ratio = md / gt_md
    out = np.copy(model_params)
    out[mask, 0:3] *= ratio[..., None]
    out[mask, 12] = 1 - fw
    return out


# print('\n-------------------------------------\n')
# L1 = 0.9
# L2 = 0.7
# L3 = 0.7
# evals = np.array([L1, L2, L3])
# md = mean_diffusivity(evals)
# fa = fractional_anisotropy(evals)

# mds = np.linspace(0.1, 1.6, num=10)
# fs = mds / md
# new_evals = (evals[..., None] * fs).T


# for eigs in new_evals:
#     print('FA: ' + str(fractional_anisotropy(eigs)))
#     print('MD: ' + str(mean_diffusivity(eigs)))
#     print(eigs)
#     print('\n')
