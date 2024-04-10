import numpy as np
from scipy.linalg import eigh
from matplotlib import pyplot as plt
from lasers import sine_square_laser, linear_laser
import pyscf
import tqdm
import os
import sys
def search_ao_nr(mol, l, m, atmshell=0):
    ibf = 0
    for ib in range(len(mol._bas)):
        l1 = mol.bas_angular(ib)
        #print(ib,l1)
        degen = l1 * 2 + 1
        if l1 == l:
            print("right l, shell: %d"%atmshell)
            if atmshell > 1+l1:
                atmshell = atmshell - 1
            else:
                return ibf + (atmshell-l1-1)*degen + (l1+m)
        ibf += degen
        print(ibf)
    raise RuntimeError('Required AO not found')

molecule = "h 0.0 0.0 0.0"

#basis = "aug-cc-pvtz"
basis = "6-aug-cc-pvtz_8K"
absorber=False
basis_path = f"basis_sets/{basis}.dat"
#basis="cc-pVTZ"
mol = pyscf.gto.Mole()
mol.unit = "bohr"
mol.build(atom=molecule, basis=basis_path, spin=1, charge=0)
#mol.build(atom=molecule,basis=basis,spin=1,charge=0)
nuclear_repulsion_energy = mol.energy_nuc()

n = mol.nelectron
l = mol.nao
h_ao = pyscf.scf.hf.get_hcore(mol)
s_ao = mol.intor_symmetric("int1e_ovlp")

r_ao = mol.intor("int1e_r").reshape(3, l, l) #<x> = <p|x|q>, <y> = <p|y|q>, <z> = <p|z|q>
counter=0
mZero_basis_elements=[]
for ib in range(len(mol._bas)):
        ia = mol.bas_atom(ib)
        l1 = mol.bas_angular(ib)
        degen = l1 * 2 + 1
        nc = mol.bas_nctr(ib)
        for m in range(-l1,l1+1):
            if m==0:
                mZero_basis_elements.append(counter)
            counter+=1
print(mZero_basis_elements)
z_ao=r_ao[1,:,:]
eps, C = eigh(h_ao, s_ao+np.eye(len(s_ao))*1e-14)
c=np.zeros(len(z_ao))
z=z_ao
c=C[:,0]
new_c=z@c
x=1e-5
z1=np.argwhere(abs(new_c)>x)
z2=np.argwhere(abs(z@new_c)>x)
z3=np.argwhere(abs(z@z@new_c)>x)
z4=np.argwhere(abs(z@z@z@new_c)>x)
z5=np.argwhere(abs(z@z@z@z@new_c)>x)
z6=np.argwhere(abs(z@z@z@z@z@new_c)>x)
z7=np.argwhere(abs(z@z@z@z@z@z@new_c)>x)
z8=np.argwhere(abs(z@z@z@z@z@z@z@new_c)>x)

from functools import reduce
mZero_basis_elements=reduce(np.union1d, (z1,z2,z3,z4,z5,z6,z7))
print(mZero_basis_elements)
print(mol.search_ao_nr(atm_id=0,l=2,m=2,atmshell=3))
print(len(mZero_basis_elements))
h_ao_new=h_ao[np.ix_(mZero_basis_elements,mZero_basis_elements)]
s_ao_new=s_ao[np.ix_(mZero_basis_elements,mZero_basis_elements)]
z_ao_new=z_ao[np.ix_(mZero_basis_elements,mZero_basis_elements)]
#h_ao=h_ao_new;s_ao=s_ao_new;z_ao=z_ao_new

l=len(h_ao)
eps, C = eigh(h_ao, s_ao+np.eye(len(s_ao))*1e-14)
#Transform to MO-basis
H = C.T @ h_ao @ C
z = C.T @ z_ao @ C #For reasons that are NOT obvious, r_ao[1] is z
try:
    E0=float(sys.argv[1])
except:
    E0 = 0.03
print(E0)
omega = 0.057

if absorber is True:
    Ecutoff=3.17*E0**2 / (4 * omega**2)
    d0=0.1
    d1=50
    gamma_val=np.diag(H).copy()
    gamma_val[gamma_val<Ecutoff]=d1
    gamma_val[abs(gamma_val-d1)>x]=d0
    print(gamma_val)
    Gamma_k=np.sqrt(2*np.diag(H)*(np.diag(H)>0))/gamma_val
    H=H-np.diag(1j*Gamma_k/2)
else:
    Gamma_k=0


t_cycle = 2 * np.pi / omega
td = 3 * t_cycle

laser = sine_square_laser(
    E0=E0, omega=omega, td=td
)


#Set the initial state to the field-free ground state.
C_t = np.zeros(l, dtype=np.complex128)
C_t[0] = 1+0j

I = np.complex128(np.eye(l))

tfinal = td
dt = 0.2

time_points = np.linspace(0, 330.8, int(330.8/dt)+1)
print(time_points)
num_steps=len(time_points)
z_t = np.zeros(num_steps, dtype=np.complex128)
z_t[0] = np.vdot(C_t, z @ C_t)
norm = np.zeros(num_steps)
norm[0] = np.linalg.norm(C_t)

for i in tqdm.tqdm(range(num_steps - 1)):

    #Do Crank-Nicholson step
    ti = time_points[i]
    Ht = H + laser(ti+dt/2) * z
    A_p = I + 1j*dt/2*Ht
    A_m = I - 1j*dt/2*Ht
    C_tmp = np.dot(A_m, C_t)
    C_t = np.linalg.solve(A_p, C_tmp)

    #Compute expectation values
    z_t[i + 1] = np.vdot(C_t, np.dot(z, C_t))
    norm[i + 1] = np.linalg.norm(C_t)
np.save("time_points_E=%.2f_absorber=%d.txt"%(E0,absorber),time_points)
np.save("time_points_E=%.2f.txt"%(E0),time_points)

np.save("z_E=%.2f_absorber=%d.txt"%(E0,absorber),z_t.real)
print(list(z_t.real))
#print(len(z_t.real))
#print(len(time_points))
#print(C.shape)
#print(C_t.shape)
#print((abs(C_t)))
