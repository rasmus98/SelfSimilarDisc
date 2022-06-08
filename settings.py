import numpy as np
from spherical_fields import ScalarField
from dataclasses import dataclass

@dataclass
class Settings:
    cs : ScalarField # defines the sound speeds and thus implicitly the grid
    Lambda_a_0 : float
    Rm_0 : float
    lambda_ : float
    beta : float
    Sigma : float
        
def fiducial_simulation(theta_grid_inner):
    return settings(theta_grid_inner=theta_grid,
                    cs = scalar_field())