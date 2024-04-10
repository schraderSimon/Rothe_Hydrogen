import numpy as np
from grid_methods.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)
from grid_methods.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_l,
    AngularMatrixElements_lm,
)
from grid_methods.spherical_coordinates.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
    Linear_map
)
import time
from matplotlib import pyplot as plt
from grid_methods.spherical_coordinates.lasers import (
    square_length_dipole,
    square_velocity_dipole,
    sine_square_laser
)
import tqdm
from opt_einsum import contract
from scipy.sparse.linalg import LinearOperator, eigsh, eigs, cg, gmres, bicgstab
from grid_methods.spherical_coordinates.utils import mask_function
from grid_methods.spherical_coordinates.preconditioners import M2Psi
from grid_methods.spherical_coordinates.rhs import (
    H0Psi,
    HtPsi,
)
from grid_methods.spherical_coordinates.time_dependent_field_interaction import (
    V_psi_length_z,
    V_psi_length,
    V_psi_velocity,
)
from opt_einsum import contract
from grid_methods.spherical_coordinates.utils import (
    Counter,
    quadrature,
)
from grid_methods.spherical_coordinates.properties import (
    expec_x_i,
    expec_p_i,
    compute_norm
)
from grid_methods.spherical_coordinates.ground_state import compute_ground_state
import sys

### INPUTS #######################

# pulse inputs
E0 = float(sys.argv[1]) #0.06
omega = 0.057
ncycles = 3
dt = 0.2

# grid inputs
N = 640
nr = N - 1
r_max = 320
l_max = int(sys.argv[2])
alpha = 0.4


### SETUP ########################

# setup Legendre-Lobatto grid
gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
weights = gll.weights

# setup radial matrix elements
radial_matrix_elements = RadialMatrixElements(gll)
potential = -radial_matrix_elements.r_inv
r = radial_matrix_elements.r
D1 = radial_matrix_elements.D1
T_D2 = -(1 / 2) * radial_matrix_elements.D2

# setup angular matrix elements
angular_matrix_elements = AngularMatrixElements_l(
    arr_to_calc=["z_Omega", "H_z_beta"], l_max=l_max
)
# angular_matrix_elements = AngularMatrixElements_lm(arr_to_calc=["z_Omega"], l_max=l_max)
n_lm = angular_matrix_elements.n_lm

# setup mask function
mask_r = mask_function(r, r[-1], r[-1] - 30)

# Compute ground/intial state
tic = time.time()
eps, phi_n = compute_ground_state(
    angular_matrix_elements, radial_matrix_elements, potential
)

# setup initial state
psi_t = np.zeros((n_lm, nr), dtype=np.complex128)
psi_t[0] = np.complex128(phi_n[:, 0])
psi_t[0] /= np.sqrt(quadrature(weights, np.abs(psi_t[0]) ** 2))
psi0 = psi_t[0].copy()
toc = time.time()
print(f"Time computing initial state: {toc-tic}")

# setup pulses
t_cycle = 2 * np.pi / omega
tfinal = ncycles * t_cycle

e_field_z = sine_square_laser(
    E0=E0, omega=omega, td=tfinal
)


# sampling arrays
num_steps = int(tfinal / dt) + 1
time_points = np.zeros(num_steps)


expec_z = np.zeros(num_steps, dtype=np.complex128)
expec_pz = np.zeros(num_steps, dtype=np.complex128)
norm_t = np.zeros(num_steps, dtype=np.complex128)
norm_t[0] = compute_norm(psi_t, weights)

nr_its_conv = np.zeros(num_steps - 1)


# right-hand side
H0_psi = H0Psi(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
)

Vt_psi = V_psi_length_z(
    angular_matrix_elements, radial_matrix_elements, e_field_z
)
# Vt_psi = V_psi_length(angular_matrix_elements, radial_matrix_elements, e_field_z=e_field_z)

rhs = HtPsi(angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi])


# preconditioner
preconditioner = M2Psi(angular_matrix_elements, radial_matrix_elements, dt)
M_linear = LinearOperator((nr * (n_lm), nr * (n_lm)), matvec=preconditioner)


# arrays needed for sampling
z_Omega = angular_matrix_elements("z_Omega")
H_z_beta = angular_matrix_elements("H_z_beta")


### RUN ##########################

for i in tqdm.tqdm(range(num_steps - 1)):
    time_points[i + 1] = (i+1) * dt
    ti = time_points[i] + dt / 2

    Ap_lambda = lambda psi, ti=ti: psi.ravel() + 1j * dt / 2 * rhs(psi, ti)
    Ap_linear = LinearOperator((nr * (n_lm), nr * (n_lm)), matvec=Ap_lambda)
    z = psi_t.ravel() - 1j * dt / 2 * rhs(psi_t, ti)

    local_counter = Counter()
    psi_t, info = bicgstab(
        Ap_linear,
        z,
        M=M_linear,
        x0=psi_t.ravel(),
        tol=1e-10,
        callback=local_counter,
    )
    nr_its_conv[i] = local_counter.counter
    psi_t = psi_t.reshape((n_lm, nr))

    #psi_t = contract("Ik, k->Ik", psi_t, mask_r)
    dpsi_t_dr = contract("ij, Ij->Ii", D1, psi_t)

    expec_z[i + 1] = expec_x_i(psi_t, weights, r, z_Omega)
    expec_pz[i + 1] = expec_p_i(psi_t, dpsi_t_dr, weights, r, z_Omega, H_z_beta)
    norm_t[i + 1] = compute_norm(psi_t, weights)


angular_distribution_psit = np.zeros(n_lm)
for I in range(n_lm):
    angular_distribution_psit[I] = quadrature(weights, np.abs(psi_t[I]) ** 2)


samples = {
    "time_points": time_points,
    "expec_z": expec_z,
    "expec_pz": expec_pz,
    "norm_t": norm_t,
    "angular_distribution_final": angular_distribution_psit,
    "psi_tfinal" : psi_t,
    "r" : r,
    "weights" : weights
}

np.savez(f"length_E0={E0}_omega={omega}_rmax={r_max}_N={N}_lmax={l_max}_dt={dt}", **samples)
