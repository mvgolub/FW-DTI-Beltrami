# FW-DTI-Beltrami

## Authors
Golub, Marc; Universidade de Lisboa Instituto Superior Tecnico, Institute
for Systems and Robotics, Department of Bioengineering

Henriques, Rafael; Champalimaud Centre for the Unknown,
Champalimaud Neuroscience Programme

Nunes, Rita; Universidade de Lisboa Instituto Superior Tecnico, Institute
for Systems and Robotics, Department of Bioengineering

## Requirements
A standard installation of DIPY is necessary to run the scripts in `/code` (https://dipy.org/documentation/1.1.1./installation/#installing-a-release).

To download the `dwi_bias_corrected.nii.gz` in `/data`, Git Large File Storage (LFS) is required (https://git-lfs.github.com/) due to GitHub's limit of 100 MB per file. Alternatively, the raw data can also be obtained at https://digital.lib.washington.edu/researchworks/handle/1773/33311, however, this version was not corrected for B1 inhomogeneities like in this study.

## Inroduction
This repository contains code that was used to obtain the results submitted in a manuscript titled  "Free water DTI estimates from single b-value data might seem plausible but must be interpreted with care" for MRM (currently under review). The purpose of this study was to compare state of the art Free Water Elimination DTI techniques and quantify the specificity of FWE-DTI estimates when extracted from single- and multi-shell dwMRI data.

## Implemented FWE-DTI routines
Regularized Gradient Descent (RGD) FWE-DTI, originally reffered as Beltrami regularization FWE-DTI, was implemented as described  in [1]. State of the art initialization methods for the FW fraction were also implemented [2]. The nonlinear least squares FWE-DTI routine already impelmeted in DIPY [3, 4] was used for multi-shell data. The conventional least squares DTI routine (also implemented in DIPY) was also used.

## Data
A dwMRI dataset of a healthy volunteer acquired on a Siemens Prisma 3T scanner and made available by CENIR, ICM, Paris was used (available at https://digital.lib.washington.edu/researchworks/handle/1773/33311. This dataset had been previously pre-processed to correct for eddy currents distortions and motion using Topup’s FSL tools [5, 6] and incorporated into DIPY as an example dataset. In addition, the selected volumes were corrected for B1 field inhomogeneities with the dwibiascorrect tool from MRTrix3, using the FSL FAST option [7].

## Processing and Simulations
A qualitative assessement (Figure 1) was first obtained by applying the implemented methods to single- and multi-shell data and visually comparing the estimated scalar maps.

Single voxel simulations (Figure 2) were perfomed on synthetic single-shell dwMRI signals with known ground truth (GT), generated with existing DIPY tools, in order to compare the performance of different initialization methods for the FW fraction described in [2]

A multi voxel phantom of a cylindrical fiber with varying degrees of FW contamination was generated to asses if the spatial regularization of the RGD FWE-DTI improves the initialized parameters. Two phantoms were generated: single- (Figure 3) and multi-shell (Figure 4). The parameters estimated with RGD FWE were compared to the GT and those estimated with standard DTI and NLS FWE-DTI (multi-shell only).

Spherical lesions were simulated for the in vivo data from CENIR. Two types of lesions were simulated: increased FW (Figure 5) or increased mean diffusivity (MD) (Figure 6). The purpose of this test was to evaluate the specificity of different FWE routines, i.e. if the single-shell FWE routine is capable of decouplig alterations in the tissue diffusion tensor from changes in FW contamination. The medians and interquartile ranges of estimated parameters inside the lesion masks of both lesions types are compared to the GT in Figure 7.

## References
[1] Pasternak O, Maier-Hein K, Baumgartner C, Shenton ME, Rathi Y, Westin C-F. The Estimation of FreeWater Corrected Diffusion Tensors. In: Westin C-F, Vilanova A, Burgeth B, editors. Visualization and Processing of Tensors and Higher Order Descriptors for Multi-Valued Data. Mathematics and Visualization.
Berlin, Heidelberg: Springer; 2014. pp. 249–270. doi: 10.1007/978-3-642-54301-2_11.

[2] Ould Ismail AA, Parker D, Fernandez M, et al. Freewater EstimatoR using iNtErpolated iniTialization
(FERNET): Toward Accurate Estimation of Free Water in Peritumoral Region Using Single-Shell Diffusion
MRI Data.; 2019. doi: 10.1101/796615.

[3] Garyfallidis E, Brett M, Amirbekian B, et al. Dipy, a library for the analysis of diffusion MRI data. Front
Neuroinform 2014;8:8 doi: 10.3389/fninf.2014.00008.

[4] Henriques RN, Rokem A, Garyfallidis E, St-Jean S, Peterson ET, Correia MM. [Re] Optimization of a free
water elimination two-compartment model for diffusion tensor imaging. Neuroscience; 2017. doi:
10.1101/108795.

[5] Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar
images: application to diffusion tensor imaging. Neuroimage 2003;20:870–888 doi: 10.1016/S1053-
8119(03)00336-7.

[6] Smith SM, Jenkinson M, Woolrich MW, et al. Advances in functional and structural MR image analysis
and implementation as FSL. Neuroimage 2004;23 Suppl 1:S208-219 doi:
10.1016/j.neuroimage.2004.07.051.

[7] Zhang Y, Brady M, Smith S. Segmentation of brain MR images through a hidden Markov random field
model and the expectation-maximization algorithm. IEEE Trans Med Imaging 2001;20:45–57 doi:
10.1109/42.906424.
