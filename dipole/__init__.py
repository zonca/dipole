import numpy as np
import pandas as pd
import logging as l
import os.path
try:
    from exceptions import IOError
except:
    pass

import healpy

import quaternionarray as qarray

class DummyClass:
    pass
physcon = DummyClass()
physcon.k = 1.380650400000e-23
physcon.c = 2.997924580000e+08
physcon.h = 6.626068960000e-34

#T_CMB = 2.725
T_CMB = 2.72548

#from hidra
#QECL2GAL = np.array((-0.37382079227204573, 0.33419217216073838, 0.64478939348298625, 0.57690575088960561))
#from healpix
l.debug('Using healpix ecl2gal')
ecl2gal = np.array([[ -5.48824860e-02,  -9.93821033e-01,  -9.64762490e-02],
                    [  4.94116468e-01,  -1.10993846e-01,   8.62281440e-01],
                    [ -8.67661702e-01,  -3.46354000e-04,   4.97154957e-01]])
eps = np.radians(23.452294 - 0.0130125 - 1.63889E-6 + 5.02778E-7)
e2q =      [[1.,     0.    ,      0.         ],
            [0., np.cos( eps ), -1. * np.sin( eps )], 
            [0., np.sin( eps ),    np.cos( eps )   ]]

QECL2GAL = qarray.from_rotmat(ecl2gal)
QECL2EQ = qarray.from_rotmat(e2q)
#              array([-0.37381694,  0.3341907 ,  0.64479285,  0.57690524])

#ephem.Date('1958/1/1 00:00')-ephem.Date('-4713/1/1 12:00:0')
JD_OBT_DAYDIFF = 2436204.5

def compute_dipole_from_vel(vel):
    beta = vel / physcon.c
    gamma=1/np.sqrt(1-beta**2)
    cosdir = 1
    T_dipole_CMB = T_CMB / (gamma * ( 1 - beta * cosdir ))
    return T_dipole_CMB - T_CMB

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

def wmap5_parameters():
    """WMAP5 solar system dipole parameters, 
    from: http://arxiv.org/abs/0803.0732"""
    # 369.0 +- .9 Km/s
    SOLSYSSPEED = 369e3
    #l.critical('USING 371')
    #SOLSYSSPEED = 371e3
    ## direction in galactic coordinates
    ##(d, l, b) = (3.355 +- 0.008 mK,263.99 +- 0.14,48.26deg +- 0.03)
    SOLSYSDIR_GAL_THETA = np.deg2rad( 90 - 48.26 )
    SOLSYSDIR_GAL_PHI = np.deg2rad( 263.99 )
    SOLSYSSPEED_GAL_U = healpy.ang2vec(SOLSYSDIR_GAL_THETA,SOLSYSDIR_GAL_PHI)
    #SOLSYSSPEED_GAL_V = SOLSYSSPEED * SOLSYSSPEED_GAL_U
    SOLSYSSPEED_ECL_U = gal2ecl(SOLSYSSPEED_GAL_U)
    SOLSYSDIR_ECL_THETA, SOLSYSDIR_ECL_PHI = healpy.vec2ang(SOLSYSSPEED_ECL_U)
    ########## /WMAP5
    return SOLSYSSPEED, SOLSYSDIR_ECL_THETA, SOLSYSDIR_ECL_PHI

def lfi_parameters():
    SOLSYSSPEED = 370374.00
    SOLSYSDIR_ECL_THETA, SOLSYSDIR_ECL_PHI = 1.7652343830474744, 2.9965189663018474
    return SOLSYSSPEED, SOLSYSDIR_ECL_THETA, SOLSYSDIR_ECL_PHI

def dipole_parameters_galactic_to_ecl_amp_ang(dipole_vector_galactic):
    dipole_vector_ecliptic = gal2ecl(dipole_vector_galactic)
    SOLSYSSPEED = np.linalg.norm(dipole_vector_ecliptic)
    SOLSYSDIR_ECL_THETA, SOLSYSDIR_ECL_PHI = healpy.vec2ang(dipole_vector_ecliptic/SOLSYSSPEED)
    return SOLSYSSPEED, SOLSYSDIR_ECL_THETA, SOLSYSDIR_ECL_PHI

def compute_SOLSYSSPEED_V(norm, theta, phi):
    return norm * healpy.ang2vec(theta,phi)

def jd2obt(jd):
    return 3600 * 24 * (float(jd) - JD_OBT_DAYDIFF)

def obt2jd(obt):
    return obt / 3600 / 24 + JD_OBT_DAYDIFF

def doppler_factor(v):
    beta=v/physcon.c
    return np.sqrt((1+beta)/(1-beta))

def load_ephemerides(file=None):
    '''Loads horizon ephemerides from CSV file, converts Julian Date to OBT, converts Km to m,
    saves to npy file'''
    if file is None:
        from planck import private
        file = private.ephem_file
    l.debug('Loading ephemerides from %s' % file)

    npyfile = file.replace('txt','npy')
    try:
        eph = np.load(npyfile)
    except IOError:
        eph = np.loadtxt(file, delimiter=',',usecols = (0,2,3,4),converters={0:jd2obt})
        eph[:,1:] *= 1e3
        npyfile = file.replace('txt','npy')
        np.save(npyfile, eph)
    return eph
    
def Planck_to_RJ(T,nu):
    h_nu_over_k = physcon.h * nu / physcon.k
    return h_nu_over_k / ( np.exp(h_nu_over_k / T)-1)

class SatelliteVelocity(object):
    """Satellite speed from Horizon"""


    def __init__(self, coord='G', interp='linear', dipole_vector_galactic=None):
        self.eph = load_ephemerides()
        self.coord = coord
        self.interp = interp
        if dipole_vector_galactic:
            self.solar_system_v_ecl = gal2ecl(dipole_vector_galactic)
        else:
            self.solar_system_v_ecl = compute_SOLSYSSPEED_V(*lfi_parameters())
        l.info('Satellite Velocity: coord=%s' % coord)
        l.debug('Dipole solar system speed: %.2f' % np.linalg.norm(self.solar_system_v_ecl))
        if self.coord == 'G':
            self.convert_coord = ecl2gal
        else:
            # no conversion
            self.convert_coord = lambda x:x

    def orbital_v(self, obt):
        '''satellite velocity from Horizon Km/s sol sys bar mean ecliptic ref
        
        nearest value from 1 minute sampled Horizon data'''
        l.debug('Computing satellite speed')

        if self.interp == 'linear':
            vsat = np.zeros([len(obt),3])
            for col in range(3):
                vsat[:,col] = np.interp(obt,self.eph[:,0],self.eph[:,col + 1])
        else:
            i_interp = np.interp(obt,self.eph[:,0],np.arange(len(self.eph[:,0])))
            i_interp = i_interp.round().astype(np.int)
            vsat = self.eph[i_interp,1:]

        l.debug('Computing satellite speed:done')
        return self.convert_coord(vsat)

    def solar_system_v(self):
        return self.convert_coord(self.solar_system_v_ecl)

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

    def __init__(self, obt=None, type='total', K_CMB=True, satellite_velocity = None, lowmem=True, coord='G', dipole_vector_galactic=None):
        if not satellite_velocity:
            satellite_velocity = SatelliteVelocity(coord=coord, dipole_vector_galactic=dipole_vector_galactic)

        l.info('Dipole: type=%s' % type)

        if type == 'total':
            self.satellite_v = satellite_velocity.total_v(obt, relativistic = True)
        elif type == 'total_classic':
            self.satellite_v = satellite_velocity.total_v(obt, relativistic = False)
        elif type == 'solar_system':
            self.satellite_v = satellite_velocity.solar_system_v()
        elif type == 'orbital':
            self.satellite_v = satellite_velocity.orbital_v(obt)
            
        self.K_CMB = K_CMB
        
        self.beam_sum = {}
        for row in np.loadtxt(os.path.join(os.path.dirname(__file__), "4pidipbeam.csv"),delimiter=',',dtype=[('tag','S10'),('s',np.double,(1,3)),('S',np.double)]):
            self.beam_sum[row['tag']] = row['s'].flatten() # * row['S'] # fix for delta dx9

        fourpi = pd.read_csv("/oasis/projects/nsf/csb136/zonca/p/software/dipole/dipole/e4pi_coeff_ba_qucs_by_diode_dx11_power=00_noalpha_v4.csv")
        fourpi["chtag"] = ["LFI%d%s" % (fh, arm) for fh, arm in zip(fourpi.fh, fourpi.polarization)]
        self.fourpi=fourpi.set_index("chtag")

    def get(self, ch, vec, maximum=False):
        l.info('Computing dipole temperature')
        #T_dipole_CMB = doppler_factor(qarray.arraylist_dot(self.satellite_v,vec).flatten()) * T_CMB
        vel = qarray.amplitude(self.satellite_v).flatten()
        beta = vel / physcon.c
        gamma=1/np.sqrt(1-beta**2)
        if maximum:
            cosdir = 1
        else:
            cosdir = qarray.arraylist_dot(qarray.norm(self.satellite_v), vec).flatten()
        T_dipole_CMB = T_CMB / (gamma * ( 1 - beta * cosdir ))
        #T_dipole_CMB = T_CMB * (1 - beta * cosdir )
        if self.K_CMB:
            return T_dipole_CMB - T_CMB
        else:
            T_dipole_RJ = ch.Planck_to_RJ( T_dipole_CMB ) - ch.Planck_to_RJ(T_CMB)
            return T_dipole_RJ

    def get_4piconv(self, ch, theta, phi, psi, horn_pointing=False):
        l.info('Computing dipole temperature with 4pi convolver')
        vel = qarray.amplitude(self.satellite_v).flatten()
        beta = vel / physcon.c
        gamma = 1./np.sqrt(1-beta**2)
        unit_vel = self.satellite_v/vel[:,None]
        if horn_pointing: # psi comes from the S channel, so there is no need 
        # to remove psi_pol
            psi_nopol = psi
        else:
            # remove psi_pol
            psi_nopol = psi - np.radians(ch.get_instrument_db_field("psi_pol"))
        # rotate vel to ecliptic
        # phi around z
        #ecl_rotation = qarray.rotation([0,0,1], -phi)
        # theta around y
        ecl_rotation = qarray.norm( qarray.mult(
            qarray.rotation([0,0,1], -psi_nopol) ,
            qarray.mult(
                        qarray.rotation([0,1,0], -theta) , 
                        qarray.rotation([0,0,1], -phi)
                       )
            ))
        # psi around z
        #ecl_rotation = qarray.mult(qarray.rotation([0,0,1], -psi) , ecl_rotation)
        # vel in beam ref frame
        vel_rad = qarray.rotate(ecl_rotation, unit_vel)

        cosdir = qarray.arraylist_dot(vel_rad, self.beam_sum[ch.tag]).flatten()
        #return beta * cosdir * T_CMB
        return (1. / ( gamma * (1 - beta * cosdir ) ) - 1) * T_CMB

    def get_4piconv_dx10(self, ch, theta, phi, psi):
        l.info('Computing dipole temperature with 4pi convolver')
        rel_vel = self.satellite_v/physcon.c
        # remove psi_pol
        psi_nopol = psi - np.radians(ch.get_instrument_db_field("psi_pol"))
        # rotate vel to horn reference frame
        tohorn_rotation = qarray.norm( qarray.mult(
            qarray.rotation([0,0,1], -psi_nopol) ,
            qarray.mult(
                        qarray.rotation([0,1,0], -theta) , 
                        qarray.rotation([0,0,1], -phi)
                       )
            ))
        # vel in beam ref frame
        vel_rad = qarray.rotate(tohorn_rotation, rel_vel)

        dipole_amplitude = self.get_fourpi_prod(vel_rad, ["S100", "S010", "S001"], ch)

        # relative corrections
        dipole_amplitude += vel_rad[:,0] * self.get_fourpi_prod(vel_rad, ["S200", "S110", "S101"], ch)/2
        dipole_amplitude += vel_rad[:,1] * self.get_fourpi_prod(vel_rad, ["S110", "S020", "S011"], ch)/2
        dipole_amplitude += vel_rad[:,2] * self.get_fourpi_prod(vel_rad, ["S101", "S011", "S002"], ch)/2

        return dipole_amplitude * T_CMB

    def get_fourpi_prod(self, vel_rad, comps, ch):
        return qarray.arraylist_dot(vel_rad, [self.fourpi[comp][ch.tag] for comp in comps]).flatten()

def solar_system_dipole_map(nside=16, nest=False, dipole_vector_galactic=None):
    pix = np.arange(healpy.nside2npix(nside),dtype=np.int)
    vec = np.zeros([len(pix),3])
    vec[:,0], vec[:,1], vec[:,2] = healpy.pix2vec(nside, pix, nest=nest)
    dip = Dipole(type='solar_system', coord='G', lowmem=False, dipole_vector_galactic=dipole_vector_galactic)
    dipole_tod = dip.get(None, vec)
    m = np.zeros(len(pix))
    m[pix] = dipole_tod
    return m

def solar_dipole_fit(s, gal_cut=30):
    import healpy as hp
    dipmap = solar_system_dipole_map(nside=hp.npix2nside(len(s)))
    return np.sum(hp.remove_monopole(s.filled(), gal_cut=gal_cut) * np.logical_not(s.mask) * dipmap)/np.sum(dipmap ** 2 * np.logical_not(s.mask))
