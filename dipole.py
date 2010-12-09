import matplotlib
matplotlib.use('Agg')
import numpy as np
import logging as l
import math
import physcon
from exceptions import IOError

import healpy

import quaternionarray as qarray

T_CMB = 2.725

#from hidra
#QECL2GAL = np.array((-0.37382079227204573, 0.33419217216073838, 0.64478939348298625, 0.57690575088960561))
#from healpix
l.debug('Using healpix ecl2gal')
ecl2gal = np.array([[ -5.48824860e-02,  -9.93821033e-01,  -9.64762490e-02],
                    [  4.94116468e-01,  -1.10993846e-01,   8.62281440e-01],
                    [ -8.67661702e-01,  -3.46354000e-04,   4.97154957e-01]])
import Quaternion
QECL2GAL = Quaternion.Quat(ecl2gal).q
#              array([-0.37381694,  0.3341907 ,  0.64479285,  0.57690524])

def ecl2gal(vec):
    return qarray.rotate(QECL2GAL , vec)

def gal2ecl(vec):
    return qarray.rotate(qarray.inv(QECL2GAL) , vec)

def relativistic_add(v,u):
    #http://en.wikipedia.org/wiki/Velocity-addition_formula
    v2 = qarray.arraylist_dot(v,v) 
    c2 = physcon.c ** 2
    v_dot_u = qarray.arraylist_dot(v,u) 
    #turn into column vector
    #if len(v_dot_u) > 1:
    #    v_dot_u = v_dot_u[:,np.newaxis]
    u_II = v_dot_u / v2 * v
    u_I_ = u - u_II
    return (v + u_II + np.sqrt(1 - v2/c2) * u_I_) / (1 + v_dot_u/c2)

#solar system speed vector

########## WMAP5  from: http://arxiv.org/abs/0803.0732
# 369.0 +- .9 Km/s
SOLSYSSPEED = 369e3
## direction in galactic coordinates
##(d, l, b) = (3.355 +- 0.008 mK,263.99 +- 0.14,48.26deg +- 0.03)
SOLSYSDIR_GAL_THETA = np.deg2rad( 90 - 48.26 )
SOLSYSDIR_GAL_PHI = np.deg2rad( 263.99 )
SOLSYSSPEED_GAL_U = healpy.ang2vec(SOLSYSDIR_GAL_THETA,SOLSYSDIR_GAL_PHI)
SOLSYSSPEED_GAL_V = SOLSYSSPEED * SOLSYSSPEED_GAL_U
SOLSYSSPEED_ECL_U = gal2ecl(SOLSYSSPEED_GAL_U)
SOLSYSDIR_ECL_THETA, SOLSYSDIR_ECL_PHI = healpy.vec2ang(SOLSYSSPEED_ECL_U)
########## /WMAP5

def SOLSYSSPEED_V():
    return SOLSYSSPEED * healpy.ang2vec(SOLSYSDIR_ECL_THETA,SOLSYSDIR_ECL_PHI)

def jd2obt(jd):
     #ephem.Date('1958/1/1 00:00')-ephem.Date('-4713/1/1 12:00:0')
     daydiff = 2436204.5
     return 3600 * 24 * (float(jd) - daydiff)

def doppler_factor(v):
    beta=v/physcon.c
    return np.sqrt((1+beta)/(1-beta))

def load_ephemerides(file='/home/zonca/p/testenv/eph/eph.txt'):
    '''Loads horizon ephemerides from CSV file, converts Julian Date to OBT, converts Km to m,
    saves to npy file'''
    l.debug('Loading ephemerides from %s' % file)
    npyfile = file.replace('txt','npy')
    try:
        eph = np.load(npyfile)
    except IOError:
        eph = np.loadtxt(file, delimiter=',',usecols = (0,1,2,3),converters={0:jd2obt})
        eph[:,1:] *= 1e3
        npyfile = file.replace('txt','npy')
        np.save(npyfile, eph)
    return eph
    
def Planck_to_RJ(T,nu):
    h_nu_over_k = physcon.h * nu / physcon.k_B
    return h_nu_over_k / ( np.exp(h_nu_over_k / T)-1)

class SatelliteVelocity(object):
    """Satellite speed from Horizon"""

    def __init__(self, coord='G'):
        self.eph = load_ephemerides()
        self.coord = coord
        if self.coord == 'G':
            self.convert_coord = ecl2gal
        else:
            # no conversion
            self.convert_coord = lambda x:x

    def orbital_v(self, obt):
        '''satellite velocity from Horizon Km/s sol sys bar mean ecliptic ref
        
        nearest value from 1 minute sampled Horizon data'''
        l.debug('Computing satellite speed')

        # TODO linear interpolation
        i_interp = np.interp(obt,self.eph[:,0],np.arange(len(self.eph[:,0])))
        i_interp = i_interp.round().astype(np.int)
        vsat = self.eph[i_interp,1:]

        l.debug('Computing satellite speed:done')
        return self.convert_coord(vsat)

    def solar_system_v(self):
        return self.convert_coord(SOLSYSSPEED_V())

    def total_v(self, obt, relativistic=True):
        if relativistic:
            l.info('Relativistic velocity addition')
            return relativistic_add(self.solar_system_v(), self.orbital_v(obt))
        else:
            l.info('Classical velocity addition')
            return self.solar_system_v() + self.orbital_v(obt)

class Dipole(object):
    """Dipole prediction:

        type: 'total', 'total_classic', 'solar_system', 'orbital'
    """

    def __init__(self, obt=None, type='total', K_CMB=True, coord='G', lowmem=True):

        self.satellite_velocity = SatelliteVelocity(coord=coord)
        if type == 'total':
            self.satellite_v = self.satellite_velocity.total_v(obt, relativistic = True)
        elif type == 'total_classic':
            self.satellite_v = self.satellite_velocity.total_v(obt, relativistic = False)
        elif type == 'solar_system':
            self.satellite_v = self.satellite_velocity.solar_system_v()
        elif type == 'orbital':
            self.satellite_v = self.satellite_velocity.orbital_v(obt)
            
        self.K_CMB = K_CMB

        if lowmem:
            del self.satellite_velocity

    def get(self, ch, vec):
        l.info('Computing dipole temperature')
        #T_dipole_CMB = doppler_factor(qarray.arraylist_dot(self.satellite_v,vec).flatten()) * T_CMB
        vel = qarray.amplitude(self.satellite_v).flatten()
        beta = vel / physcon.c
        gamma=1/np.sqrt(1-beta**2)
        cosdir = qarray.arraylist_dot(qarray.norm(self.satellite_v), vec).flatten()
        T_dipole_CMB = T_CMB / (gamma * ( 1 - beta * cosdir ))
        if self.K_CMB:
            return T_dipole_CMB - T_CMB
        else:
            T_dipole_RJ = ch.Planck_to_RJ( T_dipole_CMB ) - ch.Planck_to_RJ(T_CMB)
            return T_dipole_RJ


def solar_system_dipole_map(nside=16):
    pix = np.arange(healpy.nside2npix(nside),dtype=np.int)
    vec = np.zeros([len(pix),3])
    vec[:,0], vec[:,1], vec[:,2] = healpy.pix2vec(nside, pix)
    dip = Dipole(type='solar_system', coord='G', lowmem=False)
    dipole_tod = dip.get(None, vec)
    m = np.zeros(len(pix))
    m[pix] = dipole_tod
    return m

if __name__ == '__main__':
    obt = np.arange(1631280082, 1631280082 + 3600, 1/30.)
    from planck import pointing, Planck

    #channel
    ch = Planck.Planck()['LFI28M']

    #pointing
    pnt = pointing.Pointing(obt)
    vec = pnt.get(ch)

    #dipole
    dip = Dipole(obt, type='total', K_CMB=True)
    d = dip.get(ch, vec)

    #plot
    import matplotlib.pyplot as plt
    plt.plot(obt, d, label='channel %s' % ch.tag)
    plt.xlabel('OBT[s]'); plt.ylabel('Dipole K_CMB'); plt.grid()
