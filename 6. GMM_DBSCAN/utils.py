import numpy as np
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

def confidence_ellipse(mu,sigma,ax,n_std=3.0,edgecolor='red'):
    cov = sigma
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse([0,0], width=ell_radius_x * 2.5, height=ell_radius_y * 2.5,
                        edgecolor=edgecolor,facecolor='None')

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return(ellipse)