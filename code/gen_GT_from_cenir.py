from pathlib import Path
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.fwdti import FreeWaterTensorModel
from matplotlib import pyplot as plt


print('Loading data...')
cwd = Path.cwd()
parent_dir = cwd.parent
data_dir = parent_dir / 'data'
cenir_dir = data_dir / 'CENIR'
fdwi = cenir_dir / 'dwi_bias_corrected.nii.gz'
fbvals = cenir_dir / 'bvals'
fbvecs = cenir_dir / 'bvecs'
fmask = cenir_dir / 'binary_mask.nii.gz'

# Gradient table
bvals, bvecs = read_bvals_bvecs(str(fbvals), str(fbvecs))
gtab = gradient_table(bvals, bvecs, b0_threshold=0)

# Data and mask
data = nib.load(str(fdwi)).get_data()
affine = nib.load(str(fdwi)).affine

mask = nib.load(str(fmask)).get_data()
mask = mask.astype(bool)

# slicing data (for faster computation)
data = data[17:89, 7:97, 36:56, :]
mask = mask[17:89, 7:97, 36:56]

# masking data
masked = data * mask[..., np.newaxis]

print('data shape after masking and cropping: ' + str(masked.shape))

# extracting ground truth parameters with NLS FW-DTI
print('Running NLS FW-DTI...')
nlsmodel = FreeWaterTensorModel(gtab)
nlsfit = nlsmodel.fit(masked, mask=mask)
nlsparams = nlsfit.model_params
nlsparams[..., 0:3] *= 10**3
nlsparams[..., 12] *= -1
nlsparams[..., 12] += 1

# saving
print('Saving...')
GT_dir = data_dir / 'GT' / 'from_cenir'

# saving estimated parameters
nlsparams_img = nib.Nifti1Image(nlsparams, affine)
nib.save(nlsparams_img, str(GT_dir / 'nls_params.nii.gz'))

# saving binary mask
mask_img = nib.Nifti1Image(mask.astype(np.float32), affine)
nib.save(mask_img, str(GT_dir / 'binary_mask.nii.gz'))

# saving S0 image
S0 = np.mean(masked[..., gtab.b0s_mask], axis=-1)
S0_img = nib.Nifti1Image(S0, affine)
nib.save(S0_img, str(GT_dir / 'S0.nii.gz'))
