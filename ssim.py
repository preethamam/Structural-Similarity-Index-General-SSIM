import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple

def ssim_general(A: np.ndarray, ref: np.ndarray, exponents:list=None, C:list=None, radius:float=None) -> Tuple[float, np.array]:
    """_summary_

    Args:
        A (np.ndarray): Image for quality measurement
        ref (np.ndarray): Reference image
        exponents (list, optional): Exponents for luminance, contrast, and structural terms. Defaults to None.
        C (list, optional): Regularization constants for luminance, contrast, and structural terms. Defaults to None.
        radius (float, optional): Standard deviation of isotropic Gaussian function. Defaults to None.

    Returns:
        Tuple[float, np.array]: [SSIM index, Local values of the SSIM index]
    """
    
    # Get the dynamic range of the image
    def get_dynamic_range(A):
        if np.issubdtype(A.dtype, np.integer):
            info = np.iinfo(A.dtype)
        else:
            info = np.finfo(A.dtype)
        return info.max - info.min

    def guardedDivideAndExponent(num, den, C, exponent):
        if C > 0:
            component = num / den
        else:
            component = np.ones_like(num)
            isDenNonZero = (den != 0)
            component[isDenNonZero] = num[isDenNonZero] / den[isDenNonZero]

        if exponent != int(exponent):
            # Clamp to 0 to avoid complex values
            component = np.maximum(component, 0)

        if exponent != 1:
            component = component ** exponent

        return component

    # If 'RegularizationConstants' is not specified, choose default C.
    if C is None:
        dynmRange = get_dynamic_range(ref)
        C = [(0.01 * dynmRange) ** 2, (0.03 * dynmRange) ** 2, ((0.03 * dynmRange) ** 2) / 2]

    # General form alpha, beta and gamma
    if exponents is None:
        exponents = [1, 1, 1]

    # Number of channels
    numSpatialDims = ref.ndim

    # Parameters for Gaussian filter
    if radius is None:
        radius = 1.5  # This value can be adjusted
    
    # Convert to single or double
    A = A.astype(np.float32)
    ref = ref.astype(np.float32)

    # Weighted-mean and weighted-variance computations
    mux2 = gaussian_filter(A, sigma=radius, mode='nearest')
    muy2 = gaussian_filter(ref, sigma=radius, mode='nearest')
    muxy = mux2 * muy2
    mux2 = mux2 ** 2
    muy2 = muy2 ** 2

    # Clamp to zero. Floating point math sometimes makes this negative
    sigmax2 = np.maximum(gaussian_filter(A ** 2, sigma=radius, mode='nearest') - mux2, 0)
    sigmay2 = np.maximum(gaussian_filter(ref ** 2, sigma=radius, mode='nearest') - muy2, 0)
    sigmaxy = gaussian_filter(A * ref, sigma=radius, mode='nearest') - muxy

    # Compute SSIM index
    if (C[2] == C[1] / 2) and np.allclose(exponents, np.ones(3)):
        # Special case: Equation 13 from [1]
        num = (2 * muxy + C[0]) * (2 * sigmaxy + C[1])
        den = (mux2 + muy2 + C[0]) * (sigmax2 + sigmay2 + C[1])
        if (C[0] > 0) and (C[1] > 0):
            ssimmap = num / den
        else:
            # Need to guard against divide-by-zero if either C(1) or C(2) is 0.
            isDenNonZero = (den != 0)
            ssimmap = np.ones_like(A)
            ssimmap[isDenNonZero] = num[isDenNonZero] / den[isDenNonZero]
    else:
        # General case: Equation 12 from [1]
        # Luminance term
        if exponents[0] > 0:
            num = 2 * muxy + C[0]
            den = mux2 + muy2 + C[0]
            ssimmap = guardedDivideAndExponent(num, den, C[0], exponents[0])
        else:
            ssimmap = np.ones_like(A)

        # Contrast term
        sigmaxsigmay = None
        if exponents[1] > 0:
            sigmaxsigmay = np.sqrt(sigmax2 * sigmay2)
            num = 2 * sigmaxsigmay + C[1]
            den = sigmax2 + sigmay2 + C[1]
            ssimmap *= guardedDivideAndExponent(num, den, C[1], exponents[1])

        # Structure term
        if exponents[2] > 0:
            num = sigmaxy + C[2]
            if sigmaxsigmay is None:
                sigmaxsigmay = np.sqrt(sigmax2 * sigmay2)
            den = sigmaxsigmay + C[2]
            ssimmap *= guardedDivideAndExponent(num, den, C[2], exponents[2])

    # SSIM value
    ssimval = np.mean(ssimmap, axis=tuple(range(numSpatialDims)))

    return ssimval, ssimmap


