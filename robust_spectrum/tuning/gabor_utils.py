import numpy as np
import pandas as pd

from robust_spectrum.tuning.constants import \
    IMG_SIZE, MAX_COLOUR, GABOR_SIGMA, GABOR_GAMMA

def gabor(sigma, theta, Lambda, psi, gamma, size=40, x_shift=0, y_shift=0):
    # Gabor feature extraction.
    
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    ymin = -1.0 * (size/2.0) + 1
    ymax = -1.0 * ymin + 1
    xmin = ymin
    xmax = ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = (x-x_shift) * np.cos(theta) + (y-y_shift) * np.sin(theta)
    y_theta = -(x-x_shift) * np.sin(theta) + (y-y_shift) * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                    * np.cos(2 * np.pi / Lambda * x_theta + psi)
    
    # Convert to [0,255] range
    gb -= -1.0
    gb *= (MAX_COLOUR / 2.0 )
    
    return gb

def generate_gabors(angles, sfs, phases):
    """
    Function to generate a set of Gabors given a set of angles (orientation),
    spatial frequencies (sfs) and phases.

    Inputs:
        angles       : (numpy.ndarray) of orientations in radians
        sfs          : (numpy.ndarray) of spatial frequencies in cycles per image
        phases       : (numpy.ndarray) of phases for the Gabor wavelet

    Outputs:
        gbs          : (numpy.ndarray) of Gabors with shape (num_gabors, height,
                       width)
        gabor_params : (pandas.DataFrame) each column is a specific Gabor parameter
                       name and each entry is the specific parameter value.
    """
    sigma = GABOR_SIGMA
    gamma = GABOR_GAMMA
    img_size = IMG_SIZE # units of px. for size of each gabor

    # Currently we support three Gabor parameters
    num_gabors = len(angles) * len(sfs) * len(phases)
    gabor_params = list()
    gbs = list()
    for angle in angles:
        for sf in sfs:
            for phase in phases:
                wavelength = np.floor(img_size / sf) # px
                gb = gabor(
                    sigma,
                    angle,
                    wavelength,
                    phase,
                    gamma,
                    size=img_size,
                    x_shift=0,
                    y_shift=0
                )
                gbs.append(gb)

                # Ordering: orientation, spatial frequency, phase
                _gb_params = (angle, sf, phase)
                gabor_params.append(_gb_params)

    assert len(gbs) == num_gabors
    assert len(gabor_params) == num_gabors

    gabor_params = np.array(gabor_params)
    gbs = np.array(gbs)
    gabor_params = pd.DataFrame(
        gabor_params,
        columns=["orientations", "spatial_frequencies", "phases"]
    )
    return gbs, gabor_params


