from dipole import *
from planck import pointing

from planck import Planck

##### CHANNEL AND OBT
ch = Planck.Planck()['LFI28M']
#first sample of ring 4303
jd = 2455187.202083333
obt = np.array([jd2obt(jd)])

##### POINTING
pnt = pointing.Pointing(obt, coord='E')
theta,phi,psi=pnt.get_3ang(ch) # 1.88146617 0.0197575 -2.61200353

vec = pnt.get(ch)

##### SATELLITE VEL
sv = SatelliteVelocity(coord='E')
satellite_v = sv.total_v(obt)[0] # [390881.43499062,   49917.03170657,  -72694.92428924]

##### DIPOLE WITH NO BEAM
dip = Dipole(obt, type='total', satellite_velocity=sv)
# dipole amplitude with no beam

##### BEAMCONV DIPOLE

# dipole direction
theta_dip, phi_dip = dip.get_theta_phi_dip(satellite_v) #  1.75322403  3.01457638
dip_val = dip.get(ch, vec)[0] # -0.0031704345024921032
#theta = theta_dip
#phi = phi_dip
#psi = 0
#
#vec[0] = healpy.ang2vec(theta, phi)

# max dipole amplitude
Dmax = np.abs(dip.get(ch, None, maximum=True)[0]) # 0.0036446949082749036

#theta_bar, psi_bar = dip.get_psi_theta_bar(theta_dip, phi_dip, theta, phi) # 0.19149457  0.85664949  using (15) and (16)

delta_phi = phi_dip - phi
#psi_bar = np.arctan2( 
#                np.sin(delta_phi) * np.sin(theta_dip),
#                np.sin(theta_dip) * np.cos(theta) * np.cos(delta_phi) + np.cos(theta_dip) * np.sin(theta)
#                      )
psi_bar = 0
theta_bar = np.arccos(
            np.cos(theta) * np.cos(theta_dip) + np.sin(theta) * np.sin(theta_dip) * np.cos(delta_phi)
                     ) # (16)
#only d_x0 for x in -1,0,1
#d = d_matrix(theta_bar) # {-1: array([ 0.13458106]), 0: array([ 0.98172088]), 1: array([-0.13458106])}

d = {
0 : np.cos(theta_bar)
}

# equation 18
dip = np.zeros(len(vec)) 
m_b = 0
dip += d[m_b] * (
                np.cos(m_b * (psi_bar - psi)) *  ch.get_beam_real(m_b) -
                np.sin(m_b * (psi_bar - psi)) *  ch.get_beam_imag(m_b)
                )
#dip *= np.sqrt(4*np.pi/3) * Dmax # (18) # 0.00178896 WRONG
dip *= np.sqrt(4*np.pi/3) * Dmax # (18) # 0.00178896 WRONG

print
print('NOBEAM dipole %.3f mK' % (dip_val*1e3))
print('BEAM dipole %.3f mK' % (dip[0]*1e3))
print('Decrease %.3f%%' % (100 - dip[0]/dip_val * 100))
