import astropy.units
import numpy as np

#units: 
# dist = 1 au
# time = 1 yr
# mass = 1/(2*pi)^2 M_sun
# charge = such that mu_0=B^2/2P_B=1/2
# This unit system ensures that r_0=1, v_0=1 and GM=1
class units:
    m  = astropy.units.m.to("au")                  # AU per meter
    s  = astropy.units.s.to("yr")                  # yr per s
    kg = (2*np.pi)**2*astropy.units.kg.to("M_sun") # msun per kg
    
class constants:
    AU = 1#astropy.constants.au.value * units.m
    yr = astropy.units.yr.to("s") * units.s
    Msun = astropy.constants.M_sun.value * units.kg
    G = astropy.constants.G.value * units.m**3 * units.kg**-1 *  units.s**-2
    M = 1
    GM = G*M
    c = astropy.constants.c.value * units.m / units.s
    mu0 = 1#0.01

"""
class constants:
    AU = astropy.constants.au.si.value
    GM = (astropy.constants.M_sun * astropy.constants.G).si.value
    mu0 = astropy.constants.mu0.si.value
"""