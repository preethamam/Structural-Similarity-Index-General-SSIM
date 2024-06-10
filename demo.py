import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ssim import ssim_general

# Input images
img_ref = r"images/ref.png"
img_A = r"images/A.png"

# Read images
ref = np.array(Image.open(img_ref).convert('RGB'))
A = np.array(Image.open(img_A).convert("RGB"))

# Calculate the SSIM value    
ssimval, ssimmap = ssim_general(A, ref, exponents=[1, 1, 1])

# Print SSIM
print(ssimval)

# Show SSIM map image
plt.imshow(ssimmap, cmap='gray')
plt.colorbar()
plt.show()