from planck import pointing, Planck
import numpy as np
from dipole import Dipole

obt = np.arange(1631280082, 1631280082 + 3600, 1/30.)

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
