import astropy.units
import numpy as np

#units: 
# dist = 1 AU
# time = 1 yr
# mass = 1 M_sun
# charge = such that mu_0=B^2/2P_B=1/2
class units:
    m  = astropy.units.m.to("au")     # AU per meter
    s  = astropy.units.s.to("yr")     # yr per s
    kg = astropy.units.kg.to("M_sun") # msun per kg
    
    
class constants:
    AU = astropy.constants.au.value * units.m
    yr = astropy.units.yr.to("s") * units.s
    Msun = astropy.constants.M_sun.value * units.kg
    G = astropy.constants.G.value * units.m**3 * units.kg**-1 *  units.s**-2
    GM = G*Msun
    c = astropy.constants.c.value * units.m / units.s
    mu0 = 1/2