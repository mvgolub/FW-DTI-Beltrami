import numpy as np
from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.reconst.dki import _positive_evals
from dipy.core.gradients import gradient_table
from dipy.reconst.base import ReconstModel
from dipy.reconst.dti import (TensorFit, design_matrix, lower_triangular,
                              eig_from_lo_tri, MIN_POSITIVE_SIGNAL,
                              ols_fit_tensor, fractional_anisotropy,
                              mean_diffusivity)
from dipy.core.onetime import auto_attr
from matplotlib import pyplot as plt


MAX_DIFFFUSIVITY = 5
MIN_DIFFUSIVITY = 0.01

from matplotlib import pyplot as plt

def model_prediction(model_params, gtab, S0, Diso):
    evals = model_params[..., :3]
    evecs = model_params[..., 3:12].reshape(model_params.shape[:-1] + (3, 3))
    fraction = model_params[..., 12][..., None]
    qform = vec_val_vect(evecs, evals)
    lower_tissue = lower_triangular(qform, S0)
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
    mask = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    return (Stissue + Swater) * mask[..., None]


class Manifold():

    def __init__(self, design_matrix, model_params, attenuations, fmin, fmax,
                 Diso=3, beta=1, mask=None, zooms=None):

        # Manifold shape
        self.shape = model_params.shape[:-1]

        # Diffusion parameters
        evals = model_params[..., 0:3]
        evecs = model_params[..., 3:12].reshape(self.shape + (3, 3))
        lowtri = lower_triangular(vec_val_vect(evecs, evals), b0=None)

        # Scaled diffusion parameters
        self.X = np.copy(lowtri)
        self.X[..., [1, 3, 4]] *= np.sqrt(2)

        # Design matrix
        self.design_matrix = design_matrix

        # Scaled design matrix
        self.dH = np.copy(design_matrix)
        self.dH[..., [1, 3, 4]] *= np.sqrt(2)

        # Metric ratio
        self.beta = beta

        # Mask
        if mask is None:
            self.mask = np.ones(model_params.shape[:-1]).astype(bool)
        else:
            self.mask = mask.astype(bool)
        
        # Masks for derivatives,
        # to avoid unstable derivatives at the mask boundary
        nx, ny, nz = self.mask.shape
        shift_fx = np.append(np.arange(1, nx), nx-1)
        shift_fy = np.append(np.arange(1, ny), ny-1)
        shift_fz = np.append(np.arange(1, nz), nz-1)
        shift_bx = np.append(0, np.arange(nx-1))
        shift_by = np.append(0, np.arange(ny-1))
        shift_bz = np.append(0, np.arange(nz-1))
        self.mask_forward_x = self.mask[shift_fx, ...] * self.mask
        self.mask_forward_y = self.mask[:, shift_fy, :] * self.mask
        self.mask_forward_z = self.mask[..., shift_fz] * self.mask
        self.mask_backward_x = self.mask[shift_bx, ...] * self.mask
        self.mask_backward_y = self.mask[:, shift_by, :] * self.mask
        self.mask_backward_z = self.mask[..., shift_bz] * self.mask

        # Voxel resolution
        if zooms is None:
            self.zooms = np.array([1., 1., 1.])
        else:
            self.zooms = zooms / np.min(zooms)

        # flattened free water tensor components
        self.flat_Diso = np.zeros(self.flat_lowtri.shape)
        self.flat_Diso[..., [0, 2, 5]] = Diso

        # flattened attenuations
        self.flat_attenuations = attenuations[self.mask, :]

        # flattened tissue fraction
        self.flat_fraction = model_params[self.mask, 12][..., None]

        # flattened lower and upper limits for tisue fraction
        self.flat_fmin = fmin[self.mask][..., None]
        self.flat_fmax = fmax[self.mask][..., None]

        # Increment matrices
        self.flat_beltrami = np.zeros(self.flat_fraction.shape[:-1] + (6, ))
        self.flat_fidelity = np.zeros(self.flat_fraction.shape[:-1] + (6, ))
        self.flat_df = np.zeros(self.flat_fraction.shape)

        # cost
        self.flat_cost = np.zeros(self.flat_fraction.shape)
        self.flat_g = np.zeros(self.flat_fraction.shape)


    @staticmethod
    def forward_difference(array, d, axis):
        n = array.shape[axis]
        shift = np.append(np.arange(1, n), n-1)
        if axis == 0:
            return (array[shift, ...] - array) / d
        elif axis == 1:
            return (array[:, shift, ...] - array) / d
        elif axis == 2:
            return (array[..., shift, :] - array) / d


    @staticmethod
    def backward_difference(array, d, axis):
        n = array.shape[axis]
        shift = np.append(np.arange(1, n), n-1)
        if axis == 0:
            return (array - array[shift, ...]) / d
        elif axis == 1:
            return (array - array[:, shift, ...]) / d
        elif axis == 2:
            return (array - array[..., shift, :]) / d
    

    @property
    def flat_lowtri(self):
        out = np.copy(self.X[self.mask, :])
        out[..., [1, 3, 4]] *= 1 / np.sqrt(2)
        return out


    def compute_beltrami(self):

        # Computing derivatives
        dx, dy, dz = self.zooms
        X_dx = (Manifold.forward_difference(self.X, dx, 0)
                * self.mask_forward_x[..., None])
        X_dy = (Manifold.forward_difference(self.X, dy, 1)
                * self.mask_forward_y[..., None])
        X_dz = (Manifold.forward_difference(self.X, dz, 2)
                * self.mask_forward_z[..., None])

        # Computing the Manifold metric (Euclidean)  
        g11 = np.sum(X_dx * X_dx, axis=-1) * self.beta + 1.
        g12 = np.sum(X_dx * X_dy, axis=-1) * self.beta
        g22 = np.sum(X_dy * X_dy, axis=-1) * self.beta + 1.
        g13 = np.sum(X_dx * X_dz, axis=-1) * self.beta
        g23 = np.sum(X_dy * X_dz, axis=-1) * self.beta
        g33 = np.sum(X_dz * X_dz, axis=-1) * self.beta + 1.

        # Computing inverse metric
        gdet = (g12 * g13 * g23 * 2 + g11 * g22 * g33
                - g22 * g13**2
                - g33 * g12**2
                - g11 * g23**2)
        # # unstable values
        unstable_g = np.logical_or(gdet <= 0, gdet >= 1000) * self.mask
        gdet[unstable_g] = 1
        g11[unstable_g] = 1
        g12[unstable_g] = 0
        g22[unstable_g] = 1
        g13[unstable_g] = 0
        g23[unstable_g] = 0
        g33[unstable_g] = 1
        # the inverse
        ginv11 = (g22 * g33 - g23**2) / gdet
        ginv22 = (g11 * g33 - g13**2) / gdet
        ginv33 = (g11 * g22 - g12**2) / gdet
        ginv12 = (g13 * g23 - g12 * g33) / gdet
        ginv13 = (g12 * g23 - g13 * g22) / gdet
        ginv23 = (g12 * g13 - g11 * g23) / gdet

        # Computing Beltrami increments
        # auxiliary matrices
        g = np.sqrt(gdet)[..., None]
        g11 = ginv11[..., None]
        g12 = ginv12[..., None]
        g22 = ginv22[..., None]
        g13 = ginv13[..., None]
        g23 = ginv23[..., None]
        g33 = ginv33[..., None]
        Ax = g11 * X_dx + g12 * X_dy + g13 * X_dz
        Ay = g12 * X_dx + g22 * X_dy + g23 * X_dz
        Az = g13 * X_dx + g23 * X_dy + g33 * X_dz
        
        beltrami = (Manifold.backward_difference(g * Ax, dx, 0)
                    * self.mask_backward_x[..., None])
        beltrami += (Manifold.backward_difference(g * Ay, dy, 1)
                     * self.mask_backward_y[..., None])
        beltrami += (Manifold.backward_difference(g * Az, dz, 2)
                     * self.mask_backward_z[..., None])
        beltrami *= 1 / g 
        
        self.flat_beltrami[...] = beltrami[self.mask]

        # Save the unstable voxels masks
        self.unstable_mask = unstable_g[self.mask][..., None]

        # Save srt(det(g))
        self.flat_g[..., 0] = g[self.mask, 0]


    def compute_fidelity(self):
        Awater = np.exp(np.einsum('...j,ij->...i', self.flat_Diso,
                                   self.design_matrix))
        Atissue = np.exp(np.einsum('...j,ij->...i', self.flat_lowtri,
                                   self.design_matrix))
        Cwater = (1 - self.flat_fraction) * Awater
        Ctissue = self.flat_fraction * Atissue
        Amodel = Ctissue + Cwater
        Adiff = Amodel - self.flat_attenuations
        np.einsum('...i,ij->...j', -1 * Adiff * Ctissue,
                  self.dH, out=self.flat_fidelity)
        np.sum(-1 * (Atissue - Awater) * Adiff, axis=-1,
               out=self.flat_df[..., 0])


    def compute_cost(self, alpha):
        Awater = np.exp(np.einsum('...j,ij->...i', self.flat_Diso,
                                   self.design_matrix))
        Atissue = np.exp(np.einsum('...j,ij->...i', self.flat_lowtri,
                                   self.design_matrix))
        Cwater = (1 - self.flat_fraction) * Awater
        Ctissue = self.flat_fraction * Atissue
        Amodel = Ctissue + Cwater
        k = Amodel.shape[-1]
        self.flat_cost[..., 0] = np.sum((Amodel - self.flat_attenuations)**2, axis=-1) / k
        self.flat_cost *= 1/2
        # self.flat_cost += self.flat_g


    @property
    def update_mask(self):
        # Do not voxels with high free water
        # csf_mask = self.flat_fraction <= 0.2
        # return ~np.logical_or(self.unstable_mask, csf_mask)
        return ~self.unstable_mask

    def update(self, dt, alpha):
        self.compute_beltrami()
        self.compute_fidelity()

        # Only update stable voxels
        self.flat_beltrami *= self.update_mask
        self.flat_fidelity *= self.update_mask
        self.flat_df *= self.update_mask

        # Update parameters
        self.X[self.mask, :] += dt * (self.flat_fidelity +
                                      self.flat_beltrami * alpha)
        self.flat_fraction += dt * self.flat_df

        # constrain the tissue fraction to its lower and upper bounds
        np.clip(self.flat_fraction, self.flat_fmin, self.flat_fmax,
                out=self.flat_fraction)
        
        # update cost
        self.compute_cost(alpha)
        

    @auto_attr
    def parameters(self):
    
        dti_params = eig_from_lo_tri(self.flat_lowtri)
        out = np.zeros(self.shape + (13, ))
        out[self.mask, 0:12] = dti_params
        out[self.mask, 12] = self.flat_fraction[..., 0]
        return out


class BeltramiModel(ReconstModel):

    def __init__(self, gtab, init_method='MD', **kwargs):
        ReconstModel.__init__(self, gtab)
        if not callable(init_method):
            try:
                init_method = init_methods[init_method]
            except KeyError:
                e_s = '"' + str(init_method) + '" is not a known init '
                e_s += 'method, the init method should either be a '
                e_s += 'function or one of the available init methods'
                raise ValueError(e_s)
        self.init_method = init_method
        self.kwargs = kwargs
        self.design_matrix = design_matrix(self.gtab)
        init_keys = ('Diso', 'Stissue', 'Swater', 'min_tissue_diff',
                     'max_tissue_diff', 'tissue_MD')
        self.init_kwargs = {k:kwargs[k] for k in init_keys if k in kwargs}
        fit_keys = ('iterations', 'learning_rate', 'zooms', 'metric_ratio'
                    'reg_weight', 'Diso')
        self.fit_kwargs = {k:kwargs[k] for k in fit_keys if k in kwargs}


    def predict(self, model_params, S0=1):
        Diso = self.init_kwargs.get('Diso', 3)
        return model_prediction(model_params, self.gtab, S0, Diso)


    def fit(self, data, mask=None):

        if mask is not None:
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = mask.astype(bool, copy=False)
        else:
            mask = np.ones(data.shape[:-1]).astype(bool, copy=False)

        # Initializing tissue volume fraction
        data = np.maximum(data, MIN_POSITIVE_SIGNAL)
        masked_data = data[mask, :]
        f0 = np.zeros(data.shape[:-1])
        fmin = np.zeros(data.shape[:-1])
        fmax = np.ones(data.shape[:-1])
        f0[mask], fmin[mask], fmax[mask] = self.init_method(masked_data,
                                                            self.gtab,
                                                            **self.init_kwargs)
        np.clip(f0, fmin, fmax, out=f0) 

        # Initializing tissue tensor
        init_params = np.zeros(data.shape[:-1] + (13, ))
        Diso = self.init_kwargs.get('Diso', 3)
        min_tissue_diff = self.init_kwargs.get('min_tissue_diff', 0.001)
        max_tissue_diff = self.init_kwargs.get('max_tissue_diff', 2.5)
        init_params[mask, 0:12] = tensor_init(masked_data, self.gtab, f0[mask],
                                             min_tissue_diff=min_tissue_diff,
                                             max_tissue_diff=max_tissue_diff,
                                             Diso=Diso)
        init_params[mask, 12] = f0[mask]

        md_tissue = np.mean(init_params[..., :3], axis=-1)
        init_params[md_tissue >= 1.5, -1] = 0
        init_params[md_tissue >= 1.5, :3] = 0
        init_params[md_tissue >= 1.5, 3:-1] = 0

        # Run gradient descent
        atten, gtab = get_attenuations(data, self.gtab)
        D = design_matrix(gtab)
        beltrami_params = gradient_descent(D, init_params,
                                           atten, fmin, fmax, mask,
                                           **self.fit_kwargs)
        
        fit = BeltramiFit(self, beltrami_params)
    
        # Add the initialization parameters to Class instance (for debugging)
        fit.initial_guess = init_params
        fit.finterval = np.stack((fmin, fmax), axis=-1)

        return fit


class BeltramiFit(TensorFit):

    def __init__(self, model, model_params):
        TensorFit.__init__(self, model, model_params, model_S0=None)
    
    @property
    def f(self):
        return self.model_params[..., 12]


    @property
    def fw(self):
        return (1 - self.model_params[..., 12])


    @property
    def fwmin(self):
        return (1 - self.finterval[..., 1])


    @property
    def fwmax(self):
        return (1 - self.finterval[..., 0])


    @property
    def fw0(self):
        return 1 - self.initial_guess[..., 12]


    @property
    def fa0(self):
        return fractional_anisotropy(self.initial_guess[..., 0:3])


    @property
    def md0(self):
        return mean_diffusivity(self.initial_guess[..., 0:3])


    def predict(self, gtab, S0=1):
        Diso = self.model.fit_kwargs.get('Diso', 3)
        return model_prediction(self.model_params, gtab, S0, Diso)



def get_attenuations(signal, gtab):

    # Averaging S0 and getting normalized attenuations
    b0_inds = gtab.b0s_mask
    S0 = np.mean(signal[..., b0_inds], axis=-1)
    Sk = signal[..., ~b0_inds]
    Ak = Sk / S0[..., None]

    # Correcting non realistic attenuations
    bvals = gtab.bvals[~b0_inds]
    bvecs = gtab.bvecs[~b0_inds]
    Amin = np.exp(-bvals * MAX_DIFFFUSIVITY)
    Amin = np.tile(Amin, Ak.shape[:-1] + (1, ))
    Amax = np.exp(-bvals * MIN_DIFFUSIVITY)
    Amax = np.tile(Amax, Ak.shape[:-1] + (1, ))
    np.clip(Ak, Amin, Amax, out=Ak)

    # Adding 'dummy' b0 zero data to attenuations and gtab
    bvals = np.insert(bvals, 0 , 0)
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    this_gtab = gradient_table(bvals, bvecs) 
    this_Ak = np.ones(Ak.shape[:-1] + (Ak.shape[-1] + 1, ))
    this_gtab = gradient_table(bvals, bvecs, b0_threshold=0)
    this_Ak[..., 1:] = Ak

    return (this_Ak, this_gtab)


def fraction_init_s0(signal, gtab, Diso=3, Stissue=None, Swater=None,
                     min_tissue_diff=0.001, max_tissue_diff=2.5):

    S0 = np.mean(signal[..., gtab.b0s_mask], axis=-1)
    if Stissue is None or Swater is None:
        Stissue = np.percentile(S0, 75) 
        Swater = np.percentile(S0, 95)
        print('Stissue = ' + str(Stissue))
        print('Swater = ' + str(Swater))

    # Normalized attenuations
    Ak, this_gtab = get_attenuations(signal, gtab)
    Ak = Ak[..., 1:]
    bvals = this_gtab.bvals[1:]  # non zero bvals
    Awater = np.exp(-bvals * Diso)
    Awater = np.tile(Awater, Ak.shape[:-1] + (1, ))

    # Min and Max attenuations expected in tissue
    Atissue_min = np.exp(-bvals * max_tissue_diff)
    Atissue_max = np.exp(-bvals * min_tissue_diff)

    # Initial volume fraction
    f0 = 1 - np.log(S0 / Stissue) / np.log(Swater / Stissue)

    # Min and Max volume fraction
    fmin = np.min(Ak - Awater, axis=-1) / np.max(Atissue_max - Awater, axis=-1)
    fmax = np.max(Ak - Awater, axis=-1) / np.min(Atissue_min - Awater, axis=-1)
    fmin[fmin <= 0] = 0.0001
    fmin[fmin >= 1] = 1 - 0.0001
    fmax[fmax <= 0] = 0.0001
    fmax[fmax >= 1] = 1 - 0.0001

    return (f0, fmin, fmax)


def fraction_init_md(signal, gtab, Diso=3, tissue_MD=0.6):

    # bvals = gtab.bvals[~gtab.b0s_mask]
    bvals = gtab.bvals
    bvecs = gtab.bvecs
    mean_bval = np.max(bvals)
    # print(mean_bval)

    mbvals = bvals[np.logical_or(bvals==0, bvals==mean_bval)]
    mbvecs = bvecs[np.logical_or(bvals==0, bvals==mean_bval), :]
    mgtab = gradient_table(mbvals, mbvecs, b0_threshold=0)
    msignal = signal[..., np.logical_or(bvals==0, bvals==mean_bval)]

    # Conventional DTI
    dti_params = ols_fit_tensor(design_matrix(mgtab), msignal)
    eigvals = dti_params[..., 0:3]
    MD = np.mean(eigvals, axis=-1)  # mean diffusivity

    # Initial volume fraction
    Awater = np.exp(-mean_bval * Diso)
    Atissue = np.exp(-mean_bval * tissue_MD)
    f0 = (np.exp(-mean_bval * MD) - Awater) / (Atissue - Awater)

    # Min and Max volume fractions
    fmin = np.ones(f0.shape) * 0.0001
    fmax = np.ones(f0.shape) * (1 - 0.0001)

    return (f0, fmin, fmax)


def fraction_init_hybrid(signal, gtab, Diso=3, Stissue=None, Swater=None,
                         min_tissue_diff=0.001, max_tissue_diff=2.5,
                         tissue_MD=0.6):

    f_S0, fmin, fmax = fraction_init_s0(signal, gtab, Diso=Diso,
                                        Stissue=Stissue, Swater=Swater,
                                        min_tissue_diff=min_tissue_diff,
                                        max_tissue_diff=max_tissue_diff)
    f_MD, _, _ = fraction_init_md(signal, gtab, Diso=Diso,
                                  tissue_MD=tissue_MD)
    # hybrid initialization
    alpha = np.copy(f_S0)
    np.clip(alpha, 0.0001, 0.9999, out=alpha)
    np.clip(f_S0, fmin, fmax, out=f_S0)
    np.clip(f_MD, 0.0001, 0.9999, out=f_MD)
    f0 = (f_MD**alpha) * (f_S0**(1 - alpha))
    # f0 = (f_S0**(f_MD)) * f_MD**(1 - f_MD)

    return (f0, fmin, fmax)


def tensor_init(signal, gtab, fraction, Diso=3, min_tissue_diff=0.001,
                max_tissue_diff=2.5):

    Ak, this_gtab = get_attenuations(signal, gtab)

    # nonzero bvals and bvecs
    bvals = this_gtab.bvals
    bvecs = this_gtab.bvecs
 
    # Min and Max attenuations expected in tissue
    Atissue_min = np.exp(-bvals * max_tissue_diff)
    Atissue_min = np.tile(Atissue_min, Ak.shape[:-1] + (1, ))
    Atissue_max = np.exp(-bvals * min_tissue_diff)
    Atissue_max = np.tile(Atissue_max, Ak.shape[:-1] + (1, ))

    # correcting the attenuations for free water
    f = fraction[..., None]
    Awater = np.exp(-bvals * Diso)
    Awater = np.tile(Awater, Ak.shape[:-1] + (1, ))
    Atissue = (Ak - (1-f) * Awater) / f
    # np.clip(Atissue, Atissue_min, Atissue_max, out=Atissue)
    np.clip(Atissue, 0.0001, 0.9999, out=Atissue)

    # applying standard DTI to corrected signal
    dti_params = ols_fit_tensor(design_matrix(this_gtab), Atissue)

    return dti_params



def gradient_descent(design_matrix, initial_guess, attenuations, fmin, fmax,
                     mask, iterations=100, learning_rate=0.01, metric_ratio=1,
                     reg_weight=1, Diso=3, zooms=None):
    
    # cropping the non zero information from the data
    Ak = attenuations[..., 1:]
    H = design_matrix[1:, :-1]

    # Initializing manifold
    manifold = Manifold(H, initial_guess, Ak, fmin, fmax, Diso=Diso,
                        beta=metric_ratio, mask=mask, zooms=zooms)

    cost = np.zeros(iterations)
    for i in range(iterations):

        # At half itarations, turn off the regualrization term
        if i == iterations // 2:
            reg_weight = 0

        # Update the manifold
        manifold.update(learning_rate, reg_weight)
        cost[i] = np.mean(manifold.flat_cost)
        # print(manifold.flat_cost[1000])

    # Return the estimated parameters
    plt.figure('Cost')
    plt.plot(cost, '.')
    plt.xlabel('iterations')
    plt.ylabel('Total Cost')
    return manifold.parameters



init_methods = {'S0':fraction_init_s0,
                's0':fraction_init_s0,
                'b0':fraction_init_s0,
                'md':fraction_init_md,
                'MD':fraction_init_md,
                'mean_diffusivity':fraction_init_md,
                'hybrid':fraction_init_hybrid,
                'interp':fraction_init_hybrid,
                'log_linear':fraction_init_hybrid
                }