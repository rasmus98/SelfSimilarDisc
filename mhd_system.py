import numpy as np
from spherical_fields import *
import units as u

#defines a class "mhd_system" which is instantiated for a given set of assumed constants (ie. diffusions, magnetic field strength, column density and temperature profile), and provides functions to evaluate residual time differential for interior points (excluding 2-wide boundary on either side)
class mhd_system:
    def __init__(self, cs, Lambda_a_0, Rm_0, lambda_, ):
        # set prescibed fields eta_O, eta_H, eta_A, c_s
        # and initial conditions rho, v, B
        """Initialize simulation.

        Parameters:
        theta_grid -- the theta grid to simulate on
        h -- disc geometrical thickness
        imag -- the imaginary part (default 0.0)
        """
        centre = lambda arr: arr[len(arr)//2]

        self.cs = ScalarField(pad(cs.v, 2, False), -1/2, pad(cs.theta_grid, 2, False))
        theta_grid = self.cs.theta_grid
        z = 1/np.tan(theta_grid)
        v_k = ScalarField(np.sqrt(u.constants.GM), -1/2, theta_grid)
        omega_k = v_k
        h = centre(self.cs.v) / omega_k.v
        print(h)
        Lambda_A = ScalarField(Lambda_a_0*np.exp((z/(lambda_*h))**4), -1, theta_grid, prescription="cylindrical")
        self.Lambda_A = Lambda_A
        self.eta_O = lambda rho: omega_k*h*h/ \
            ScalarField(Rm_0*np.exp((z/(lambda_*h))**4)*(centre(rho.v)/rho.v), -2, theta_grid, prescription="cylindrical")
        self.eta_H = ScalarField(0, -3/2 + 2, theta_grid)
        self.eta_A = lambda B, rho: (B.theta**2+B.r**2+B.phi**2)/(rho * u.constants.mu0)/(omega_k*Lambda_A)
    
    #returns J*4pi/c
    def J_tilde(self, B):
        return B.curl()
    
    def epsilon(self, rho, v, B):
        b_unit = B.unit()[1:-1]
        J_tilde = self.J_tilde(B)
        J_tilde_cross_B = J_tilde.cross(b_unit)

        return -v.cross(B)[1:-1] + (
              self.eta_O(rho)[1:-1] * J_tilde \
            + self.eta_H[1:-1] * J_tilde_cross_B \
            - self.eta_A(B, rho)[1:-1] * J_tilde_cross_B.cross(b_unit)
        )
    
    def get_rho_dt(self, rho, v):
        return -(rho[1:-1] * v[1:-1]).div()
    
    def get_rho_v_dt(self, rho, v, B):
        grav_force_field = VectorField(np.ones_like(rho.v)*u.constants.GM/u.constants.AU**2, np.zeros_like(rho.v), np.zeros_like(rho.v), -2, rho.theta_grid)  
        
        return (-v.tensor_deriv(rho*v)[1:-1]
            - (rho[1:-1]*(self.cs[1:-1])**2).grad()
            + self.J_tilde(B[1:-1]).cross(B[2:-2])*(1/(4*np.pi))
            - (grav_force_field * rho)[2:-2]
        )
    
    def get_B_dt(self, rho, v, B):
        return -self.epsilon(rho, v, B).curl()
