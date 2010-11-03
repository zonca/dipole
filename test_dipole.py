#TO BE RUN ON THE PLANCK CLUSTER
from dipole import *
from testenv import M3dipole
from testenv.remix import read_exchange
import matplotlib.pyplot as plt
from planck.LFI import LFI
from planck.pointing import Pointing

l.basicConfig(level=l.DEBUG,format='%(asctime)s %(levelname)-8s %(message)s')
lfi = LFI()
ch = lfi['LFI28M']
TOLERANCE = 3e-5
'''Test vs LFI DPC dipole OD100 LFI28M'''
obtx = np.load('/u/zonca/p/testenv/dipdpc/obt.npy') / 2**16
dipx = np.load('/u/zonca/p/testenv/dipdpc/dip.npy')
read_exchange([ch], ods = [100], discard_flag = True)
span = 32.5 * 60 * 10
obt = ch.f.obtx[:span]
pnt = Pointing(obt,coord='G')
det_dir = pnt.get(ch)
np.save('obt_LFI28M_OD100',obt)
dip = Dipole(obt)
vec = det_dir[:span]
dipole = dip.get(ch, det_dir[:span],K_CMB=True)
i = obtx.searchsorted(ch.f.obtx[0])
m3dip = M3dipole.Dipole().get(ch, obt)
plt.figure()
plt.plot(obtx[i:i+span],dipx[i:i+span],label='dpc')
plt.plot(obt,dipole,'r',label='te')
plt.plot(obt,m3dip[:len(obt)],'k',label='m3')
plt.ylabel('K')
plt.legend()
plt.grid()
plt.savefig('comparedip.png')
plt.figure()
plt.title('DIPOLE LFI28M OD100')
plt.plot(obt,(dipole-dipx[i:i+span])*1e3,'r',label='te-dpc')
plt.plot(obt,(m3dip-dipx[i:i+span])*1e3,'k',label='m3-dpc')
plt.ylabel('mK')
plt.xlabel('OBT[s]')
plt.legend()
plt.grid()
plt.savefig('diffdip.png')
std = (dipole - dipx[i:i+span]).std()
print('Standard deviation of difference of DPC and TE dipole')
print(std)
assert (dipole - dipx[i:i+span]).std() < TOLERANCE
