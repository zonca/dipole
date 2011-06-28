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
phi += np.pi
vec = pnt.get(ch)
vec[0] = healpy.ang2vec(theta, phi)
#theta = 0
#phi = 0
#psi = 0

##### SATELLITE VEL
sv = SatelliteVelocity(coord='E')
satellite_v = sv.total_v(obt)[0] # [390881.43499062,   49917.03170657,  -72694.92428924]

##### DIPOLE WITH NO BEAM
dip = Dipole(obt, type='total', satellite_velocity=sv)
# dipole amplitude with no beam
dip_val = dip.get(ch, vec)[0] # -0.0031704345024921032

##### BEAMCONV DIPOLE

# dipole direction
theta_dip, phi_dip = dip.get_theta_phi_dip(satellite_v) #  1.75322403  3.01457638

# max dipole amplitude
Dmax = np.abs(dip.get(ch, None, maximum=True)[0]) # 0.0036446949082749036

theta_bar, psi_bar = dip.get_psi_theta_bar(theta_dip, phi_dip, theta, phi) # 0.19149457  0.85664949  using (15) and (16)

#theta_bar += np.pi

#only d_x0 for x in -1,0,1
d = d_matrix(theta_bar) # {-1: array([ 0.13458106]), 0: array([ 0.98172088]), 1: array([-0.13458106])}

# equation 18
dip_beam = np.zeros(len(vec)) 
for m_b in [-1, 0, 1]:
    dip_beam += d[m_b] * (
            np.cos(m_b * (psi_bar - psi)) * (ch.get_beam_real(m_b) + ch.get_beam_real(m_b, 'farsidelobe')) -
            np.sin(m_b * (psi_bar - psi)) * (ch.get_beam_imag(m_b) + ch.get_beam_imag(m_b, 'farsidelobe'))
            )
dip_beam *= np.sqrt(4*np.pi/3) * Dmax # (18) # 0.00178896 WRONG

print
print('NOBEAM dipole %.3f mK' % (dip_val*1e3))
print('BEAM dipole %.3f mK' % (dip_beam[0]*1e3))
print('Decrease %.3f%%' % (100 - dip_beam[0]/dip_val * 100))
