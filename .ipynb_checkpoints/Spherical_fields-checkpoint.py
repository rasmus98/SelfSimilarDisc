
#performes central differences differentiation in theta direction
@numba.njit
def diff(array, dtheta):
    return (array[2:] - array[:-2])/(dtheta*2)

@numba.njit
def cylindrical_to_spherical(values, theta_grid, a):
    return values / np.cos(theta_grid)**a

@numba.njit
def pad(values, amount, constant=True):
    res = np.empty(len(values) + 2*amount)
    res[amount:-amount] = values
    if constant:
        res[:amount] = values[0]
        res[-amount:] = values[-1]
    else:
        for i in range(amount):
            res[amount-i-1] = (i+1) * values[0] - values[1]
        res[:amount] = values[0]
        res[-amount:] = values[-1]
    return res
        

# represents a scale invariant vector field, and provides assosiated vector operations
# ie a field such that f(r,theta,phi)=val(theta)*(r/r_0)**a
# at the point r=r_0 in units of r_0=1
#@numba.experimental.jitclass([("v_r", numba.float64[:]), ("v_theta", numba.float64[:]), ("v_phi", numba.float64[:]), ("a", numba.float64), ("dtheta", numba.float64), ("theta_grid", numba.float64[:])])
class VectorField:
    def __init__(self, v_r, v_theta, v_phi, a, theta_grid, prescription="spherical"):
        #assert (prescription in ["theta", "z"]), f"val must be prescibed as either Q(z/R) or Q(theta). was given Q({prescription})"
        #theta_diff = np.diff(theta_grid)
        #assert (np.max(theta_diff) - np.min(theta_diff)) < 1e-6, "all theta values must be identical"
        #else:
        #    self.val = np.zeros_like(theta_grid) + val
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
            
    def div(f):
        f_tilde = (f.a+2)*f.v_r[1:-1] \
                + 1/np.sin(f.theta_grid[1:-1])*diff(np.sin(f.theta_grid) * f.v_theta, f.dtheta)
        return ScalarField(f_tilde, f.a-1, f.theta_grid[1:-1])
    
    #Implements directional deriv (g \cdot \nabla f) as f.direction_deriv(g)
    def direction_deriv(f, g): 
        cot = 1/np.tan(f.theta_grid[1:-1])
        g_new = g[1:-1]
        res_r     = (g_new * ScalarField(f.v_r, f.a, f.theta_grid).grad()).v     \
                    - g_new.v_theta * f.v_theta[1:-1] - g_new.v_phi*f.v_phi[1:-1]
        res_theta = (g_new * ScalarField(f.v_theta, f.a, f.theta_grid).grad()).v \
                    - g_new.v_theta * f.v_r[1:-1] - g_new.v_phi*f.v_phi[1:-1]*cot
        res_phi   = (g_new * ScalarField(f.v_phi, f.a, f.theta_grid).grad()).v   \
                    - g_new.v_phi  *  f.v_r[1:-1] + g_new.v_phi*f.v_theta[1:-1]*cot
        return VectorField(res_r, res_theta, res_phi, g.a+f.a-1, g_new.theta_grid)
    
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
        #assert f.val.shape == g.val.shape and f.val.ndim == 2, "need to both be vector fields"
        return VectorField(f.v_theta*g.v_phi-f.v_phi*f.v_theta,
                           f.v_phi*g.v_r-f.v_r*g.v_phi,
                           f.v_r*g.v_theta-f.v_theta*f.v_r, 
                           f.a + g.a, f.theta_grid)
    
    def __add__(f, g):
        #assert np.all(np.isclose(f.theta_grid, g.theta_grid)), "need to be defined on the same domain"
        #assert f.val.shape == g.val.shape,  f"need to have the same shape. has {f.val.shape} and {g.val.shape}"
        #assert f.a == g.a, f"need to have the same power index. has {f.a} and {g.a}"
        return VectorField(f.v_r+g.v_r, f.v_theta+g.v_theta, f.v_phi+g.v_phi, f.a, f.theta_grid)
    
    def __sub__(f, g):
        return VectorField(f.v_r-g.v_r, f.v_theta-g.v_theta, f.v_phi-g.v_phi, f.a, f.theta_grid)
    
    def __neg__(f): 
        return VectorField(-f.v_r, -f.v_theta, -f.v_phi, f.a, f.theta_grid)

    def __pow__(f, scalar): 
        return VectorField(f.v_r**scalar, f.v_theta**scalar, f.v_phi**scalar, f.a*scalar, f.theta_grid)
    
    def unit(f):
        #assert f.val.ndim == 2, "need to be vector field"
        norm = f.norm()
        return VectorField(f.v_r/norm, f.v_theta/norm, f.v_phi/norm, 0, f.theta_grid)
    
    def norm(f):
        return (f.v_r**2+f.v_theta**2+f.v_phi**2)**0.5

# represents a scale invariant vector field, and provides assosiated vector operations
# ie a field such that f(r,theta,phi)=val(theta)*(r/r_0)**a
# at the point r=r_0 in units of r_0=1
#@numba.experimental.jitclass([("v", numba.float64[:]), ("a", numba.float64), ("dtheta", numba.float64), ("theta_grid", numba.float64[:])])
class ScalarField:
    def __init__(self, v, a, theta_grid, prescription="spherical"):
        #assert (prescription in ["theta", "z"]), f"val must be prescibed as either Q(z/R) or Q(theta). was given Q({prescription})"
        #theta_diff = np.diff(theta_grid)
        #assert (np.max(theta_diff) - np.min(theta_diff)) < 1e-6, "all theta values must be identical"
        #else:
        #    self.val = np.zeros_like(theta_grid) + val
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
            return VectorField(f.v*g, f.a, f.theta_grid)

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
        #assert f.val.shape == g.val.shape,  f"need to have the same shape. has {f.val.shape} and {g.val.shape}"
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
    