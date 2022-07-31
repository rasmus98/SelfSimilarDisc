import numba
import numpy as np

#performes central differences differentiation in theta direction
@numba.njit(fastmath=True)
def diff(array, dtheta):
    return (array[2:] - array[:-2])/(dtheta*2)

@numba.njit
def cylindrical_to_spherical(values, theta_grid, a):
    return values * np.sin(theta_grid)**a
        
@numba.njit
def pad(values, amount, constant=True):
    res = np.empty(len(values) + 2*amount)
    res[amount:-amount] = values
    if constant:
        res[:amount] = values[0]
        res[-amount:] = values[-1]
    else:
        for i in range(1,amount+1):
            res[amount -i]   = values[0] + i * (values[0] - values[1])
            res[-1-(amount - i)] = values[-1] + i * (values[-1] - values[-2])
    return res    

# represents a scale invariant vector field, and provides assosiated vector operations
# ie a field such that f(r,theta,phi)=val(theta)*(r/r_0)**a
# at the point r=r_0 in units of r_0=1
#@numba.experimental.jitclass([("v_r", numba.float64[:]), ("v_theta", numba.float64[:]), ("v_phi", numba.float64[:]), ("a", numba.float64), ("dtheta", numba.float64), ("theta_grid", numba.float64[:])])
class VectorField:
    def __init__(self, v_r, v_theta, v_phi, a, theta_grid, prescription="spherical"):
        #assert (prescription in ["spherical", "cylindrical"]), f"val must be prescibed as either Q(z/R) or Q(theta). was given {prescription}"
        theta_diff = np.diff(theta_grid)
        #assert (np.abs(np.max(theta_diff)/ np.min(theta_diff)) - 1) < 1e-3, "all theta values must be identical"
        if prescription == "spherical":
            self.v_r = v_r
            self.v_theta = v_theta
            self.v_phi = v_phi
        elif prescription == "cylindrical":
            self.v_r = cylindrical_to_spherical(v_r, theta_grid, a) 
            self.v_theta = cylindrical_to_spherical(v_theta, theta_grid, a) 
            self.v_phi = cylindrical_to_spherical(v_phi, theta_grid, a) 
        else:
            raise NotImplementedError(f"prescription not implemented")

        self.a = a
        self.theta_grid = theta_grid
        self.dtheta = theta_grid[1] - theta_grid[0]
            
    def __repr__ (f):
        return f"({f.v_r}\\hat{{r}}\n+{f.v_theta}\\hat{{theta}}\n+{f.v_phi}\\hat{{phi}})\n* (r/r_0)^{f.a}"
        
    # method for extracting a part of the field. Usually usefull for cutting off the edges
    def __getitem__(self, key):
        return VectorField(self.v_r[key], self.v_theta[key], self.v_phi[key], self.a, self.theta_grid[key])
    
    @property
    def r(self):
        return ScalarField(self.v_r, self.a, self.theta_grid)
    @property
    def theta(self):
        return ScalarField(self.v_theta, self.a, self.theta_grid)
    @property
    def phi(self):
        return ScalarField(self.v_phi, self.a, self.theta_grid)
                
    def div(f):
        f_tilde = (f.a+2)*f.v_r[1:-1] \
                + 1/np.sin(f.theta_grid[1:-1])*diff(np.sin(f.theta_grid) * f.v_theta, f.dtheta)
        return ScalarField(f_tilde, f.a-1, f.theta_grid[1:-1])
    
    #Implements tensor deriv \nabla\cdot(g \otimes f) as f.direction_deriv(g)
    def tensor_deriv(f, g): 
        cot = 1/np.tan(f.theta_grid)
        res_r     = (f*g.r).div().v   - (f.v_theta*g.v_theta + f.v_phi * g.v_phi)[1:-1]
        res_theta = (f*g.theta).div().v   + (f.v_theta*g.v_r - cot*f.v_phi * g.v_phi)[1:-1]
        res_phi   = (f*g.phi).div().v + (f.v_phi*g.v_r + cot*f.v_phi * g.v_theta)[1:-1]
        #print((f*g.r).div().v)
        #print((f.v_theta*g.v_theta + f.v_phi * g.v_phi)[1:-1])
        return VectorField(res_r, res_theta, res_phi, g.a+f.a-1, f.theta_grid[1:-1])
    
    def curl(f):
        f_new = f[1:-1]
        f_tilde = [1/np.sin(f_new.theta_grid) * diff(f.v_phi*np.sin(f.theta_grid), f.dtheta),
                   -(f.a+1)*f_new.v_phi,
                   +(f.a+1)*f_new.v_theta - diff(f.v_r, f.dtheta)]
        return VectorField(*f_tilde, f.a-1, f_new.theta_grid)
    
    # vector field * vector field = dot product
    # vector field * scalar field = scalar product
    # scalar field * scalar field = normal multi
    def __mul__(f, g):
        if type(g) is VectorField:
            #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
            return ScalarField(f.v_r*g.v_r + f.v_theta*g.v_theta + f.v_phi*g.v_phi, f.a + g.a, f.theta_grid)
        elif type(g) is ScalarField:
            return VectorField(f.v_r*g.v, f.v_theta*g.v, f.v_phi*g.v, f.a+g.a, f.theta_grid)
        else: #scalar
            return VectorField(f.v_r*g, f.v_theta*g, f.v_phi*g, f.a, f.theta_grid)

    # vector field / scalar field 
    # scalar field / float 
    def __truediv__(f, g):
        if type(g) is ScalarField:
            #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
            return VectorField(f.v_r/g.v, f.v_theta/g.v, f.v_phi/g.v,
                               f.a - g.a, f.theta_grid)
        else:
            raise NotImplementedError(f"type of g not implemented")
    
    def cross(f, g):
        #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
        return VectorField(f.v_theta*g.v_phi-f.v_phi*g.v_theta,
                           f.v_phi*g.v_r-f.v_r*g.v_phi,
                           f.v_r*g.v_theta-f.v_theta*g.v_r, 
                           f.a + g.a, f.theta_grid)
    
    def __add__(f, g):
        #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
        #assert f.a == g.a, f"need to have the same power index. has {f.a} and {g.a}"
        return VectorField(f.v_r+g.v_r, f.v_theta+g.v_theta, f.v_phi+g.v_phi, f.a, f.theta_grid)
    
    def __sub__(f, g):
        return VectorField(f.v_r-g.v_r, f.v_theta-g.v_theta, f.v_phi-g.v_phi, f.a, f.theta_grid)
    
    def __neg__(f): 
        return VectorField(-f.v_r, -f.v_theta, -f.v_phi, f.a, f.theta_grid)

    def __pow__(f, scalar): 
        return VectorField(f.v_r**scalar, f.v_theta**scalar, f.v_phi**scalar, f.a*scalar, f.theta_grid)
    
    def unit(f):
        norm = f.norm()
        return VectorField(f.v_r/norm, f.v_theta/norm, f.v_phi/norm, 0, f.theta_grid)
    
    def norm(f):
        return (f.v_r**2+f.v_theta**2+f.v_phi**2)**0.5
    
    
    def at_coord(f, r, theta):
        return np.array([f.r.at_coord(r, theta),
                         f.theta.at_coord(r, theta),
                         f.phi.at_coord(r, theta)])

# represents a scale invariant vector field, and provides assosiated vector operations
# ie a field such that f(r,theta,phi)=val(theta)*(r/r_0)**a
# at the point r=r_0 in units of r_0=1
#@numba.experimental.jitclass([("v", numba.float64[:]), ("a", numba.float64), ("dtheta", numba.float64), ("theta_grid", numba.float64[:])])
class ScalarField:
    def __init__(self, v, a, theta_grid, prescription="spherical"):
        #assert (prescription in ["spherical", "cylindrical"]), f"val must be prescibed as either Q(z/R) or Q(theta). was given Q({prescription})"
        theta_diff = np.diff(theta_grid)
        #assert (np.abs(np.max(theta_diff)/ np.min(theta_diff)) - 1) < 1e-3, "all theta values must be identical"
        v_ = np.zeros_like(theta_grid) + v
        if prescription == "spherical":
            self.v = v_
        elif prescription == "cylindrical":
            self.v = cylindrical_to_spherical(v_, theta_grid, a) 
        else:
            raise NotImplementedError(f"prescription not implemented")
        self.a = a
        self.theta_grid = theta_grid
        self.dtheta = theta_grid[1] - theta_grid[0]
            
    def __repr__ (f):
        return f"{f.v} * (r/r_0)^{f.a}"
        
    # method for extracting a part of the field. Usually usefull for cutting off the edges
    def __getitem__(self, key):
        return ScalarField(self.v[key], self.a, self.theta_grid[key])
        
    def grad(f):
        f_tilde = ([f.a*f.v[1:-1], diff(f.v, f.dtheta), np.zeros(len(f.v)-2)])
        return VectorField(*f_tilde, f.a-1, f.theta_grid[1:-1])
    
    # vector field * vector field = dot product
    # vector field * scalar field = scalar product
    # scalar field * scalar field = normal multi
    def __mul__(f, g):
        if type(g) is VectorField:
            return g*f
        elif type(g) is ScalarField:
            return ScalarField(f.v*g.v, f.a+g.a, f.theta_grid)
        else: #scalar
            return ScalarField(f.v*g, f.a, f.theta_grid)
        

    # vector field / scalar field 
    # scalar field / float 
    def __truediv__(f, g):
        if type(g) is ScalarField:
            #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
            return ScalarField(f.v/g.v, f.a - g.a, f.theta_grid)
        if isinstance(g, (int,float)):
            #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
            return ScalarField(f.v/g, f.a, f.theta_grid)
        else:
            raise NotImplementedError(f"type of g not implemented")

    def __add__(f, g):
        #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
        #assert f.a == g.a, f"need to have the same power index. has {f.a} and {g.a}"
        return ScalarField(f.v+g.v, f.a, f.theta_grid)
    
    def __sub__(f, g):
        return ScalarField(f.v-g.v, f.a, f.theta_grid)
    
    def __neg__(f): 
        return ScalarField(-f.v, f.a, f.theta_grid)

    def __pow__(f, scalar): 
        return ScalarField(f.v**scalar, f.a*scalar, f.theta_grid)
    
    def norm(f):
        return np.abs(f.v)
    
    def at_coord(f, r, theta):
        return r**f.a*np.interp(theta, f.theta_grid, f.v, left=np.nan, right=np.nan)