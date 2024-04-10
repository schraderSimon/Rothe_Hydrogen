import numpy as np
from numba import jit
from scipy.integrate import simps
import scipy.fftpack
import scipy.signal
from matplotlib import pyplot as plt
from utils import compute_hhg_spectrum
from grid_methods.spherical_coordinates.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
    Linear_map
)

N = 1200
r_max = 600

# setup Legendre-Lobatto grid
gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
weights = gll.weights
PN_x = gll.PN_x
r_dot = gll.r_dot
r = gll.r



### INPUTS #######################

# pulse inputs
E0 = 0.12
omega = 0.057
dt = 0.2



l_max_list = [30, 50, 60, 70]

for l_max in l_max_list:

	dat = np.load(f'length_E0=0.12_omega=0.057_rmax=600_N=1200_lmax={l_max}_dt=0.2.npz')
	time_points = dat["time_points"]
	expec_z = dat["expec_z"]
	expec_pz = dat["expec_pz"]
	angular_dist = dat["angular_distribution_final"]
	psi_final = dat["psi_tfinal"]
	print(psi_final.shape)
	
	dt = time_points[1]-time_points[0]
	
	freq, z_omega = compute_hhg_spectrum(time_points, expec_z.real, hann_window=True)
	freq, pz_omega = compute_hhg_spectrum(time_points, expec_pz.real, hann_window=True)
	
	plt.figure(1)
	plt.plot(time_points, expec_z.real)
	
	Up = E0**2 / (4 * omega**2)
	Ip = 0.5
	Ecutoff = Ip + 3.17 * Up
    

	plt.figure(2)
	plt.subplot(211)
	plt.semilogy(freq / omega, z_omega, label=r"$l_{max}={%d}$" % l_max)
	plt.xlim(0,100)
	plt.xticks(np.arange(1, 100, step=5))
	plt.axvline(
        Ecutoff / omega, linestyle="dotted", color="red"
    )
	plt.legend()
	plt.subplot(212)
	plt.semilogy(freq / omega, pz_omega, label=r"$l_{max}={%d}$" % l_max)
	plt.xlim(0,100)
	plt.xticks(np.arange(1, 100, step=5))
	plt.axvline(
	    Ecutoff / omega, linestyle="dotted", color="red"
	)

	print(np.sum(angular_dist))
	plt.figure(3)
	plt.semilogy(angular_dist, '-o', label=r"$l_{max}={%d}$" % l_max)
	plt.legend()
	plt.xlabel('l')
	plt.ylabel(r"$P_l(\Psi)$")

	ur_tot = np.zeros(len(r[1:-1]))
	for l in range(l_max):
		ur_tot += np.abs(psi_final[l]/np.sqrt(r_dot[1:-1])*PN_x)**2

	plt.figure(4)
	plt.subplot(211)
	plt.plot(r[1:-1], np.abs(psi_final[0]/np.sqrt(r_dot[1:-1])*PN_x)**2, label=r"$l_{max}={%d}$" % l_max)
	plt.subplot(212)
	plt.plot(r[1:-1], ur_tot, label=r"$l_{max}={%d}$" % l_max)
	plt.legend()
plt.show()