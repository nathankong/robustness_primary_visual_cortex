import numpy as np

MAX_COLOUR = 255.0

IMG_SIZE = 224

# Gabor / grating parameters; orientation in radians, phase in radians, 
# spatial frequencies in cycles per image (assume 224 px <-> 6.4 deg)
PARAM_DICT = dict()
PARAM_DICT["orientations"] = np.linspace(0, 172.5, 10) * np.pi / 180
PARAM_DICT["spatial_frequencies"] = np.array([2.5, 3.5, 5, 7.1, 10, 14.1, 20, 28.3, 40, 56])
PARAM_DICT["phases"] = np.linspace(0, 2*np.pi, 10)

# Gabor patch parameters
GABOR_SIGMA = 35
GABOR_GAMMA = 1 # spatial aspect ratio (1 = isotropic)

