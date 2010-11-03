import matplotlib
matplotlib.use('Agg')
import numpy as np
import logging as l
import math
import physcon
from exceptions import IOError

from healpy.pixelfunc import ang2vec

import quaternionarray as qarray

T_CMB = 2.725
QECL2GAL = np.array((-0.37382079227204573, 0.33419217216073838, 0.64478939348298625, 0.57690575088960561))

#solar system speed vector
# ONE
SOLSYSDIR_ECL_THETA = 1.7678013480275747
SOLSYSDIR_ECL_PHI = 3.0039153062803194
# TWO
#SOLSYSDIR_ECL_THETA = 1.765248346
#SOLSYSDIR_ECL_PHI = 2.995840906
SOLSYSSPEED = 371000.0 #todo CHECK
SOLSYSSPEED_V = SOLSYSSPEED * ang2vec(SOLSYSDIR_ECL_THETA,SOLSYSDIR_ECL_PHI)

def jd2obt(jd):
     #ephem.Date('1958/1/1 00:00')-ephem.Date('-4713/1/1 12:00:0')
     daydiff = 2436204.5
     return 3600 * 24 * (float(jd) - daydiff)

def doppler_factor(v):
    beta=v/physcon.c
    return np.sqrt((1+beta)/(1-beta))
    
def Planck_to_RJ(T,nu):
    h_nu_over_k = physcon.h * nu / physcon.k_B
    return h_nu_over_k / ( np.exp(h_nu_over_k / T)-1)

def ecl2gal(vec):
    return qarray.rotate(QECL2GAL , vec)

class SatelliteVelocity(object):
    """Satellite speed from Horizon"""

    def __init__(self, coord='G'):
        self.load_ephemerides()
        self.coord = coord
        if self.coord == 'G':
            self.convert_coord = ecl2gal
        else:
            # no conversion
            self.convert_coord = lambda x:x

    def load_ephemerides(self, file='/home/zonca/p/testenv/eph/eph.txt'):
        l.debug('Loading ephemerides from %s' % file)
        npyfile = file.replace('txt','npy')
        try:
            self.eph = np.load(npyfile)
        except IOError:
            eph = np.loadtxt(file, delimiter=',',usecols = (0,1,2,3),converters={0:jd2obt})
            eph[:,1:] *= 1e3
            self.eph = eph
            np.save(npyfile, eph)

    def satellite_v(self, obt):
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
        return self.convert_coord(SOLSYSSPEED_V)

    def total_v(self, obt):
        #TODO relativistic sum
        return self.satellite_v(obt) + self.solar_system_v()

class Dipole(object):

    def __init__(self, obt, type='total'):
        if type == 'total':
            self.satellite_v = SatelliteVelocity().total_v(obt)
        elif type == 'cmb':
            self.satellite_v = 0

    def get(self, ch, vec, K_CMB=True):
        l.info('Computing dipole temperature')
        T_dipole_CMB = doppler_factor(np.sum(self.satellite_v*vec,axis=1)) * T_CMB
        if K_CMB:
            return T_dipole_CMB - T_CMB
        else:
            T_dipole_RJ = ch.Planck_to_RJ( T_dipole_CMB ) - ch.Planck_to_RJ(T_CMB)
            return T_dipole_RJ

if __name__ == '__main__':
    obt = np.arange(1631280082, 1631280082 + 3600, 1/30.)
    from planck import pointing, planck

    #channel
    ch = planck.Planck()['LFI28M']

    #pointing
    pnt = pointing.Pointing(obt)
    vec = pnt.get(ch)

    #dipole
    dip = Dipole(obt, type='total')
    d = dip.get(ch, vec, T_CMB=True)

    #plot
    import matplotlib.pyplot as plt
    plt.plot(obt, d, label='channel %s' % ch.tag)
    plt.xlabel('OBT[s]'); plt.ylabel('Dipole K_CMB'); plt.grid()
