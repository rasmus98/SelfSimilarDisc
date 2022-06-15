import numpy as np
from spherical_fields import *

#Defines the class "field_regularizer" which parses a "parameter representaion" (ie flattened rho, vr, vtheta, vphi, btheta, bphi) into the physical rho, B and v fields normalized correctly for density and magnetic field, and provides backpropagation for converting physical gradient to normalized parameter gradient
class field_regularizer:
    def __init__(self, parameter_rep, theta_grid, cs, beta, Sigma = None):
        self.rho, self.v, self.B, self.theta_grid = self.un_flatten(parameter_rep, theta_grid)
        
        centre = lambda arr: arr[len(arr)//2]

        #Normalize column density
        if Sigma != None:
            column_density = np.trapz(self.rho.as_cylendrical(), np.tan(self.theta_grid+np.pi/2))
            #print(column_density)
            self.rho_multiplier = Sigma / column_density
            self.rho *= self.rho_multiplier
        else:
            self.rho_multiplier = 1
        
        #Normalize magnetic field strength
        B_target = np.sqrt(8*np.pi*centre(self.rho.v)**2*centre(cs.v)**2/beta)
        self.B_multiplier = B_target / centre(-self.B.v_r)
        self.B *= self.B_multiplier

    # A divergence-free B-field uniquely sets B_r
    # here, we also expand with 3 on each side, such that Br is expanded with 2 on each side
    @staticmethod
    def infer_Br(B_theta, theta_grid):
        expanded_theta = pad(theta_grid, 3, False)
        expanded_B     = pad(B_theta, 3, False)
        expanded_Br = -diff(expanded_B*np.sin(expanded_theta), theta_grid[1]-theta_grid[0]) \
                /(np.sin(expanded_theta[1:-1]) * (-5/4 + 2))
        return expanded_Br

    @staticmethod
    def un_flatten(parameter_rep, theta_grid):
        theta_grid = pad(theta_grid, 2, False)
        rho_data, vr_data, vtheta_data, vphi_data, Btheta_data, Bphi_data = np.split(parameter_rep, 6)
        rho = ScalarField(pad(rho_data**2/np.sum(rho_data**2), 2), -3/2, theta_grid)
        v = VectorField(*[pad(v, 2) for v in [vr_data, vtheta_data, vphi_data]],
                  -1/2, theta_grid)
        B = VectorField(field_regularizer.infer_Br(Btheta_data, theta_grid[2:-2]), *[pad(B, 2, False) for B in [Btheta_data, Bphi_data]], -5/4, theta_grid)
        return rho, v, B, theta_grid
    
    def gradient_to_parameter_rep(self, rho_dt, rho_v_dt, B_dt):
        v_dt = (rho_v_dt-self.v[2:-2]*rho_dt)/self.rho[2:-2]
        return np.concatenate(
            [rho_res.v / self.rho_multiplier, 
             vres.v_rho, 
             vres.v_theta, 
             vres.v_phi, 
             B_res.v_theta / self.B_multiplier, 
             B_res.v_phi / self.B_multiplier])
