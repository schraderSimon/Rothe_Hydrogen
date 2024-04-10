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
from grid_methods.spherical_coordinates.lasers import (
    square_length_dipole,
    square_velocity_dipole,
    sine_square_laser
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
E0 = 0.06
omega = 0.057
dt = 0.2

l_max_list = [10, 15, 20, 25, 30]

for l_max in l_max_list:

	dat = np.load(f'length_E0={E0}_omega=0.057_rmax={r_max}_N={N}_lmax={l_max}_dt=0.2.npz')
	time_points = dat["time_points"]
	expec_z = dat["expec_z"]
	expec_pz = dat["expec_pz"]
	
	freq, z_omega = compute_hhg_spectrum(time_points, expec_z.real, hann_window=True)
	freq, pz_omega = compute_hhg_spectrum(time_points, expec_pz.real, hann_window=True)
	
	plt.figure(1)
	plt.plot(time_points, expec_z.real, label=r'$<z(t)>, l_{max}={%d}$' % l_max)
	plt.legend()
	
	Up = E0**2 / (4 * omega**2)
	Ip = 0.5
	Ecutoff = Ip + 3.17 * Up
    

	plt.figure(2)
	plt.subplot(211)
	plt.semilogy(freq / omega, z_omega, label=r"$|z(\omega)|^2, l_{max}={%d}$" % l_max)
	plt.xlim(0,71)
	plt.xticks(np.arange(1, 71, step=2))
	plt.axvline(
        Ecutoff / omega, linestyle="dotted", color="red"
    )
	plt.legend()
	plt.subplot(212)
	plt.semilogy(freq / omega, pz_omega, label=r"$|p_z(\omega)|^2, l_{max}={%d}$" % l_max)
	plt.xlim(0,71)
	plt.xticks(np.arange(1, 71, step=2))
	plt.legend()
	plt.xlabel(r'$\omega/\omega_L$')
	plt.axvline(
	    Ecutoff / omega, linestyle="dotted", color="red"
	)

# ncycles = 3
# t_cycle = 2 * np.pi / omega
# tfinal = ncycles * t_cycle

# e_field_z = sine_square_laser(
#     E0=E0, omega=omega, td=tfinal
# )

# plt.figure()
# plt.plot(time_points, e_field_z(time_points), label=r'E(t)', color='red')
# plt.legend()
# plt.xlabel('Time [a.u.]')
	
plt.show()