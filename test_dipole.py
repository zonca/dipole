from __future__ import division
import numpy as np
import healpy
import unittest
import physcon

import sys
import os
sys.path.append(os.getcwd())
from dipole import *

class TestJDRelAdd(unittest.TestCase):
    def setUp(self):
        #data
        self.v1 = np.array([ 6e7,  0,  0])
        self.u = qarray.norm(np.array([ 4e4,  8e5,  9e6])) * physcon.c / 30
        self.v = qarray.norm(np.array([ 5e3,  8e7,  9e6])) * physcon.c / 40

    def test_jd2obt(self):
        # from horizon:
        #2455117.500000000, A.D. 2009-Oct-13 00:00:00.0000
        jd = 2455117.5
        # from HFI:
        # 1634083200997595094     13/10/2009 00:00:00    DOY      286
        obt = 1634083200997595094 / 1e9
        obt -= 1 #error?
        self.assertAlmostEqual(jd2obt(jd), obt, 0)

    def test_relativistic_add_norm(self):
        self.assertAlmostEqual(np.linalg.norm(relativistic_add(self.v1, self.v1*3)), (self.v1[0]+self.v1[0]*3)/(1+self.v1[0]*self.v1[0]*3/physcon.c**2))

    def test_relativistic_add_vec_components(self):
        result = relativistic_add(self.v, self.u)
        u1,u2,u3 = self.u
        v1,v2,v3 = self.v
        gamma = 1/np.sqrt( 1 - (v1**2 + v2**2 + v3**2)/physcon.c**2)
        pre = 1 / ( 1 + (v1*u1 + v2*u2 + v3*u3)/physcon.c**2)
        inside = 1 + 1/physcon.c**2 * (gamma/(1+gamma)) * ( v1*u1+v2*u2+v3*u3)
        np.testing.assert_array_almost_equal(result, (pre * ( inside * self.v + self.u/gamma))[np.newaxis,:])

    def test_relativistic_add_vec(self):
        result = relativistic_add(self.v, self.u)
        u = self.u
        v = self.v
        c2=physcon.c**2  
        g=1/np.sqrt(1-np.dot(v,v)/c2)
        res = 1/(1+np.dot(v,u)/c2)*(v+u/g+1/c2*(g/(1+g))*np.dot(v,u)*v)
        np.testing.assert_array_almost_equal(result, res[np.newaxis,:])

    def test_relativistic_add_vec_duncan_hanson(self):
        vs = np.array([-360148.44820513,   52867.15941346,  -71687.92583822]) # solsys velocity m/s
        vo = np.array([ -3.05003755e+04,  -1.55395210e+01,  -7.79262783e+01])  # orbital velocity m/s
        vpar = np.dot(vs, vo) / np.dot(vs,vs) * vs     #component of vo parallel to vs
        vper = vo - vpar                                           #component of vo perpendicular to vs
        np.testing.assert_array_almost_equal( (vs + vpar + np.sqrt(1.-np.dot(vs,vs)/(physcon.c*physcon.c))*vper)/(1.+np.dot(vs,vo)/(physcon.c*physcon.c)), relativistic_add(vs, vo).flatten())

class TestDipole(unittest.TestCase):
    
    def setUp(self):
        #first sample of ring 4303
        theta = 33.736313919148174 #deg
        phi = 180.67100354137608 #deg
        self.vec = healpy.ang2vec(np.radians(theta), np.radians(phi))

        jd = 2455187.202083333 #see previous test
        self.obt = [jd2obt(jd)]

    def test_satellite_velocity(self):
        # from Horizon
        # 2455187.202083333, A.D. 2009-Dec-21 16:51:00.0000, -3.050037254295194E+01, -1.516941595027445E-02, -7.792438654080185E-02,
        horiz_vec = np.array([-3.050037254295194E+01, -1.516941595027445E-02, -7.792438654080185E-02]) * 1e3
        satvel = SatelliteVelocity(coord='E')
        orbital_vec = satvel.orbital_v(self.obt)
        np.testing.assert_array_almost_equal(horiz_vec, orbital_vec.flatten(), decimal = 4)

    def test_orbital_dipole(self):
        #manually computed (see Dipole on joint wiki)
        expected_dip = 153.365312527 / 1.e6
        dip = Dipole(self.obt, type='orbital', satellite_velocity=SatelliteVelocity(coord='E'))
        dip_val = dip.get(None, self.vec)

        self.assertAlmostEqual(expected_dip, dip_val)


    def test_solsys_velocity(self):
        np.testing.assert_array_almost_equal(SatelliteVelocity.solar_system_v_ecl, SatelliteVelocity(coord = 'E').solar_system_v())
        
class TestSolSysDipole(unittest.TestCase):
    """Requires the planck module and access to PLANCK RIMO"""

    def setUp(self):
        #first sample of ring 4303
        theta = 33.736313919148174 #deg
        phi = 180.67100354137608 #deg
        self.vec = healpy.ang2vec(np.radians(theta), np.radians(phi))

        jd = 2455187.202083333 #see previous test
        self.obt = [jd2obt(jd)]

        from planck import Planck
        from testenv import hfidipole
        self.Dipole = hfidipole.Dipole
        self.ch = Planck.Planck()['100-4a']

    def test_solar_system_dipole(self):
        #manually computed (see Dipole on joint wiki)
        expected_dip = 0.0012714435357161769
        dip = self.Dipole(self.obt, type='solar_system', satellite_velocity=SatelliteVelocity(coord='E'))
        dip_val = dip.get(self.ch, self.vec)[0]

        self.assertAlmostEqual(expected_dip, dip_val)

    def test_total_dipole(self):
        #manually computed (see Dipole on joint wiki)
        expected_dip = 0.0014246200343004389
        dip = self.Dipole(self.obt, type='total', satellite_velocity=SatelliteVelocity(coord='E'))
        dip_val = dip.get(self.ch, self.vec)[0]

        self.assertAlmostEqual(expected_dip, dip_val)

if __name__ == '__main__':
    # better to use nose
    unittest.main()
