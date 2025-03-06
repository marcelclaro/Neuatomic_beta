from NNTB.util import *
from cmath import exp
import scipy as sci
import torch
import numpy as np
from ase.dft.kpoints import monkhorst_pack
import math
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh, inv
from scipy.special import jv  # Bessel function of the first kind
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import hilbert


# Function to process a batch of edges
def process_edge_batch_sparse(batch, R_vec_lst, system):
    indices_H = []
    values_H = []
    indices_S = []
    values_S = []

    for index, reversed, Rvec, ham, overlap in batch:
        if not reversed:
            Ridx = next((i for i, vec in enumerate(R_vec_lst) if torch.equal(vec, Rvec)), None)

            hoppingindices_row = torch.arange(index[1]*system.num_orbitals,(index[1]+1)*system.num_orbitals)
            hoppingindices_col = torch.arange(index[0]*system.num_orbitals,(index[0]+1)*system.num_orbitals)
            row_ix, col_ix = torch.meshgrid(
                torch.tensor(hoppingindices_row, dtype=torch.int32),
                torch.tensor(hoppingindices_col, dtype=torch.int32),
                indexing='ij'
            )

            indices_H.append(torch.stack([row_ix.flatten(), col_ix.flatten()]))
            values_H.append(ham.flatten())
            indices_S.append(torch.stack([row_ix.flatten(), col_ix.flatten()]))
            values_S.append(overlap.flatten())

            if torch.all(Rvec == 0) and index[1] != index[0]:
                indices_H.append(torch.stack([col_ix.flatten(), row_ix.flatten()]))
                values_H.append(ham.flatten())
                indices_S.append(torch.stack([col_ix.flatten(), row_ix.flatten()]))
                values_S.append(overlap.flatten())

    return indices_H, values_H, indices_S, values_S

# Function to create and distribute batches
def process_edges_parallel_sparse(graph, R_vec_lst, system, n_process=4):
    # Prepare data batches
    edge_data = list(zip(
        graph.edge_index.t(), graph.reversed, graph.edge_shift.to(torch.float32), graph.hopping, graph.overlap
    ))
    batch_size = math.ceil(len(edge_data) / n_process)
    num_batches = math.ceil(len(edge_data) / batch_size)
    batches = [edge_data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    # Start threading
    indices_H_all = []
    values_H_all = []
    indices_S_all = []
    values_S_all = []

    with ProcessPoolExecutor(max_workers=n_process) as executor:
        future_to_batch = {executor.submit(process_edge_batch_sparse, batch,R_vec_lst, system): batch for batch in batches}

        for future in as_completed(future_to_batch):
            indices_H, values_H, indices_S, values_S = future.result()
            indices_H_all.extend(indices_H)
            values_H_all.extend(values_H)
            indices_S_all.extend(indices_S)
            values_S_all.extend(values_S)


    # Concatenate results to form final sparse matrices
    indices_H_all = torch.cat(indices_H_all, dim=1)
    values_H_all = torch.cat(values_H_all)
    indices_S_all = torch.cat(indices_S_all, dim=1)
    values_S_all = torch.cat(values_S_all)

    return indices_H_all, values_H_all, indices_S_all, values_S_all


class TBHamiltonian_scipy:
    
    def __init__(self, graph, system, periodic = False, orthogonalization = False ,sparse = True, n_process=4):
        """
        Compute the Hamiltonian from a graph representation using PyTorch.
        Args:
            graph (Graph): The graph representation of the system, containing attributes such as spin, elementonehot, edge_shift, edge_index, reversed, hopping, and overlap.
            system (System): The system object containing attributes such as element_decoding, type_onehot, orbitalstrings, spin, and band_kpoints.
            kpoints (list, optional): List of k-points for which to compute the band structure. If None, system.band_kpoints will be used.
        Returns:
            torch.Tensor: A tensor containing the Hamiltonian for each k-point.
        """
        if graph.spin:
            hamtype = torch.complex64
            self.spin = True
        else:
            hamtype = torch.float32
            self.spin = False

        self.periodic = periodic
        self.orthogonal = False
        self.sparse = sparse
        self.basechange = None

        assert sparse, "Dense Hamiltonian not implemented for scipy, torch preferred."
        assert not periodic and sparse, "Periodic Hamiltonian not implemented for sparse matrices."

        #Get the element from the one-hot attribute
        elemlst=[system.element_decoding[i] for x in graph.elementonehot.to('cpu') for i, value in enumerate(system.type_onehot) if torch.equal(value, x)]
        ham_template = [(idx, orb) for idx, atom in enumerate(elemlst) for orb in system.orbitals]
        self.basis_len= len(ham_template)
        self.num_orbitals = len(system.orbitals)

        self.cell_volume = torch.linalg.det(torch.tensor(graph.lattice_vectors))
        self.orbital_pos = graph.pos
        self.R_vec = [torch.tensor([0, 0, 0], dtype=torch.float32)]   
        indices_H_all, values_H_all, indices_S_all, values_S_all = process_edges_parallel_sparse(graph, self.R_vec, system, n_process=n_process)
        self.HR = sp.csr_matrix((values_H_all.cpu().numpy(), (indices_H_all[0].cpu().numpy(), indices_H_all[1].cpu().numpy())), shape=(self.basis_len, self.basis_len))
        self.SR = sp.csr_matrix((values_S_all.cpu().numpy(), (indices_S_all[0].cpu().numpy(), indices_S_all[1].cpu().numpy())), shape=(self.basis_len, self.basis_len))            

        if orthogonalization:
            self.orthogonalize()

    def save_hamiltonian(self, filename):
        """
        Save the Hamiltonian (HR, SR, R_vec, periodic, orthogonal, sparse, basechange) to a file.
        Args:
            filename (str): The file path to save the Hamiltonian.
        """
        data = {
            'HR_data': self.HR.data,
            'HR_indices': np.vstack((self.HR.indices, self.HR.indptr)),
            'SR_data': self.SR.data,
            'SR_indices': np.vstack((self.SR.indices, self.SR.indptr)),
            'R_vec': np.array([vec.cpu().numpy() for vec in self.R_vec]),
            'periodic': self.periodic,
            'orthogonal': self.orthogonal,
            'sparse': self.sparse,
            'basechange': self.basechange.cpu().numpy() if self.basechange is not None else None,
            'spin': self.spin,
            'type': 'scipy'
        }
        np.savez_compressed(filename, **data)

    def load_hamiltonian(self, filename):
        """
        Load the Hamiltonian (HR, SR, R_vec, periodic, orthogonal, sparse, basechange) from a file.
        Args:
            filename (str): The file path to load the Hamiltonian from.
        """
        data = np.load(filename)
        assert data['type'] == 'scipy', "File type mismatch. Expected 'scipy' but got {data['type']}."


        self.HR = sp.csr_matrix((data['HR_data'], data['HR_indices']))
        self.SR = sp.csr_matrix((data['SR_data'], data['SR_indices']))

        self.R_vec = torch.tensor(data['R_vec'])
        self.periodic = bool(data['periodic'])
        self.orthogonal = bool(data['orthogonal'])
        self.sparse = bool(data['sparse'])
        self.spin = bool(data['spin'])
        self.basechange = torch.tensor(data['basechange']) if data['basechange'] is not None else None

    def orthogonalize(self):
        dS = self.SR - sp.eye(self.SR.shape[0])
        S_1_2 = (sp.eye(dS.shape[0])  # Identity matrix
                + (1 / 2.0) * dS
                - (1 / 8.0) * dS @ dS
                + (1 / 16.0) * dS @ dS @ dS
                - (5 / 128.0) * dS @ dS @ dS @ dS)
        # + (7/256.0) * dS @ dS @ dS @ dS @ dS
        # - (21/1024.0) * dS @ dS @ dS @ dS @ dS @ dS
        # + (33/2048.0) * dS @ dS @ dS @ dS @ dS @ dS @ dS)

        S_1_2_inv = sp.linalg.inv(S_1_2)

        self.HR = S_1_2_inv @ self.HR @ S_1_2_inv
        self.SR = torch.eye(self.SR.size(0))
        self.basechange = S_1_2_inv
        self.orthogonal = True

def rescale_hamiltonian(H, symmetrization = True):
    """
    Rescale the Hamiltonian such that its eigenvalues are in the range [-1, 1].
    """
    if symmetrization:
        H_symm = 0.5 * (H + H.T)
    H_max = sp.linalg.norm(H, ord=2)  # Largest eigenvalue approximation for sparse matrix
    print(f"Rescaling factor (Maximum Eigenvalue): {H_max}")
    return  H_symm / H_max, H_max

def get_bessel_series(time_step,scale, bessel_max = 250, bessel_precision = 1.0e-14):
    """
    Get the values of Bessel functions up to given order.

    :param time_step: time step
    :return: values of Bessel functions
    :raises ValueError: if self.config.generic["Bessel_max"] is too low
    """
    time_scaled = time_step * scale

    # Collect bessel function values
    bessel_series = []
    converged = False
    for i in range(bessel_max):
        bes_val = jv(i, time_scaled)
        if np.abs(bes_val) > bessel_precision:
            bessel_series.append(bes_val)
        else:
            bes_val_up = jv(i + 1, time_scaled)
            if np.abs(bes_val_up) > bessel_precision:
                bessel_series.append(bes_val)
            else:
                converged = True
                break

    # Check and return results
    if not converged:
        raise ValueError("Bessel_max too low")

    return bessel_series

def chebyshev_time_evolution_H(H_rescaled, psi_0, bessel_series):
    """
    Evolve the wave function using the Chebyshev expansion.
    
    H: Rescaled Hamiltonian matrix (numpy array)
    psi_0: Initial wave function (numpy array)
    t_max: Total time for the evolution
    num_steps: Number of time steps
    """

    bessel = bessel_series
    
    # Prepare arrays to store the Chebyshev polynomials and wavefunctions
    psi_t = psi_0.copy()

    # Chebyshev polynomial terms
    T0 = psi_t  # T_0(H) |psi_0> = |psi_0>
    T1 = -1j * H_rescaled @ psi_t  # T_1(H) |psi_0> = -i H |psi_0>
    
    # Initial wavefunction time evolution using J0
    psi_t = bessel[0] * T0 + 2 * bessel[1] * T1
    
    num_factors = len(bessel)

    # Start the Chebyshev recurrence for T_m
    for n in range(2, num_factors):
        T2 = -2j * H_rescaled @ T1 + T0  # Chebyshev recurrence: T_{n+1} = -2i H T_n + T_{n-1}
        
        # Update the wave function at each step using the Bessel function J_n
        psi_t += 2 * bessel[n] * T2
        
        # Update T0, T1 for the next iteration
        T0 = T1
        T1 = T2
    
    return psi_t

def chebyshev_Fermi(H_rescaled, psi_0, factor_series):
    """
    Evolve the wave function using the Chebyshev expansion.
    
    H: Rescaled Hamiltonian matrix (numpy array)
    psi_0: Initial wave function (numpy array)
    t_max: Total time for the evolution
    num_steps: Number of time steps
    """
   
    # Prepare arrays to store the Chebyshev polynomials and wavefunctions
    psi_t = psi_0.copy()

    # Chebyshev polynomial terms
    T0 = psi_t  # T_0(H) |psi_0> = |psi_0>
    T1 = H_rescaled @ psi_t  # T_1(H) |psi_0> =  H |psi_0>
    
    # Initial wavefunction time evolution using J0
    psi_t = factor_series[0] * T0 + 2 * factor_series[1] * T1
    
    num_factors = len(factor_series)

    # Start the Chebyshev recurrence for T_m
    for n in range(2, num_factors):
        T2 = 2* H_rescaled @ T1 - T0  # Chebyshev recurrence: T_{n+1} = -2i H T_n + T_{n-1}
        
        # Update the wave function at each step using the Bessel function J_n
        psi_t += factor_series[n] * T2
        
        # Update T0, T1 for the next iteration
        T0 = T1
        T1 = T2
    
    return psi_t


def get_Fermi_cheb_coef(beta, mu, one_minus_Fermi, nr_Fermi =  2**15, eps = 1.0e-12):
    # Placeholder for required external functions
    def Fermi_dist(beta, mu, energy):
        # Define the Fermi distribution function
        return 1 / (np.exp(beta * (energy - mu)) + 1)

    def jackson_kernel(n_cheb):
        # Define the Jackson kernel for damping
        return np.array([((n_cheb - i + 1) * np.cos(np.pi * i / n_cheb) +
                          np.sin(np.pi * i / n_cheb) / np.tan(np.pi / n_cheb)) /
                         (n_cheb + 1) for i in range(1, n_cheb + 1)])

    # Initialize variables
    r0 = 2 * np.pi / nr_Fermi
    cheb_coef_complex = np.zeros(nr_Fermi, dtype=np.complex64)

    # Compute coefficients for Fermi or one-minus-Fermi operator
    for i in range(nr_Fermi):
        energy = np.cos(i * r0)
        if one_minus_Fermi:
            cheb_coef_complex[i] = 1 - Fermi_dist(beta, mu, energy)
        else:
            cheb_coef_complex[i] = Fermi_dist(beta, mu, energy)

    # Fourier transform result
    cheb_coef_complex = np.fft.ifft(cheb_coef_complex)

    # Number of nonzero elements
    n_cheb = nr_Fermi // 2

    # Apply Jackson kernel
    kernel = jackson_kernel(n_cheb)
    cheb_coef = np.real(cheb_coef_complex[:n_cheb]) * kernel / n_cheb

    # Trim coefficients below threshold epsilon
    for i in range(n_cheb):
        if abs(cheb_coef[i]) < eps and abs(cheb_coef[i + 1]) < eps:
            n_cheb = i + 1
            cheb_coef = cheb_coef[:n_cheb]
            break

    return cheb_coef



def calc_corr_accond(hamiltonian,temperature,fermilevel,component = "xx", num_samples = 1, time_steps = 1024, energy_range = 20.0, seed = 42, wfn_check_step = 128, wfn_check_thr = 1e-5):
    """
    Calculate the correlation function of density of states (DOS) using PyTorch.

    Parameters:
    - num_samples: Number of random samples.
    - time_steps: Number of time steps.
    - rank: Rank of the current process (for distributed processing).
    - seed: Seed for random number generation.
    - wfn_check_step: Check step for wavefunction norm.
    - wfn_check_thr: Threshold for wavefunction norm check.
    - output_file: File to save results.

    Returns:
    - corr: Correlation function of DOS.
    """
    # Get parameters
    time_step = 2 * math.pi / energy_range  # Time step for Chebyshev evolution
    print("Assemblying Hamiltonian")
    H = hamiltonian.HR
    print("Hamiltonian Constructed")
    corr = np.zeros(time_steps + 1, dtype=np.complex64)
    H_rescaled, H_max = rescale_hamiltonian(H)
    beta = 1 / (k_B_eV * temperature)
    scaled_beta = beta / H_max
    scaled_fermilevel = fermilevel / H_max
    bessel = get_bessel_series(time_step,H_max)
    wf_dim = H.shape[0]

    time_step_fs = time_step * hbar_eV * 1e15
    print(f"Time step for propagation: {time_step_fs:7.3f} fs\n")

    # Set random seed
    np.random.seed(seed)

    # Convert H to COO format
    H_coo = H.tocoo()

    # Extract row, column, and data from COO format
    row = H_coo.row
    col = H_coo.col

    # Compute current coefficients
    orbital_indices_row = np.floor_divide(row, hamiltonian.num_orbitals)
    orbital_indices_col = np.floor_divide(col, hamiltonian.num_orbitals)

    # Get the orbital positions for each band
    dr_values  = hamiltonian.orbital_pos[orbital_indices_row] - hamiltonian.orbital_pos[orbital_indices_col]

    # Aliases for variables
    if component not in [a+b for a in "xyz" for b in "xyz"]:
        raise ValueError(f"Illegal component {component}")
    comp = np.array(["xyz".index(_) for _ in component], dtype=np.int32)


    # Create COO matrix with values = hop_dr and indexes of H_coo
    dr_mat_alpha = sp.coo_matrix((dr_values[:,comp[0]].cpu().numpy(), (row, col)), shape=H.shape)
    dr_mat_beta = sp.coo_matrix((dr_values[:,comp[1]].cpu().numpy(), (row, col)), shape=H.shape)
    sys_current_alpha = 1j * H_max * H_coo * dr_mat_alpha
    sys_current_beta = 1j * H_max * H_coo * dr_mat_beta



    # Get Fermi Chebyshev coefficients
    coef_F = get_Fermi_cheb_coef(scaled_beta, scaled_fermilevel, False)
    coef_omF = get_Fermi_cheb_coef(scaled_beta, scaled_fermilevel, True)

    print("Calculating AC-Conductivity correlation function.")

    # Loop over the number of random samples
    for i_sample in range(num_samples):
        print(f"Sample {i_sample + 1} of {num_samples}")

        np.random.seed(seed+i_sample)

        # Generate random initial state
        wf0 = np.random.rand(wf_dim) + 1j * np.random.rand(wf_dim)
        wf0 /= np.linalg.norm(wf0)
        psi1_alpha = chebyshev_Fermi(H_rescaled, sys_current_alpha @ wf0, coef_omF)
        psi2 = chebyshev_Fermi(H_rescaled, wf0, coef_F)

        # Reference norm for wavefunction checks
        norm_ref = np.dot(np.conj(psi2), psi2)

        # Iterate over remaining time steps
        for t in range(2, time_steps + 1):
            if t % wfn_check_step == 0:
                # Wavefunction norm check
                norm_diff = np.abs(np.dot(np.conj(psi2), psi2) - norm_ref)
                if norm_diff > wfn_check_thr:
                    print(f"Warning: Wavefunction norm exceeded threshold at timestep {t}: Difference = {norm_diff}")

            # Time evolution step
            psi1_alpha= chebyshev_time_evolution_H(H_rescaled, psi1_alpha, bessel)
            psi2 = chebyshev_time_evolution_H(H_rescaled, psi2, bessel)

            # Calculate correlations
            corrval = np.dot(psi2, sys_current_beta @ psi1_alpha)

            # Update correlation results
            corr[t] += corrval / num_samples

    # Save the correlation data
    return corr

def window_exp(i: int, tnr: int) -> float:
    """
    Exponential window function.

    :param i: summation index
    :param tnr: total length of summation
    :return: exponential window value
    """
    return np.exp(-2. * (i / tnr)**2)


def window_hanning(i, tnr):
    """
    Hanning window function.

    :param i: summation index
    :param tnr: total length of summation
    :return: Hanning window value
    """
    return 0.5 * (1 + np.cos(np.pi * i / tnr))


def calc_ac_cond(corr_ac, hamiltonian, temperature, window  = window_exp, time_steps = 1024, energy_range = 20.0, spin = False):
    """
    Calculate optical (AC) conductivity from correlation function.

    Reference: eqn. 300-301 of graphene note.

    The unit of AC conductivity in 2d case follows:
    [sigma] = [1/(h_bar * omega * A)] * [j^2] * [dt]
            = 1/(eV*nm^2) * e^2/h_bar^2 * (eV)^2 * nm^2 * h_bar/eV
            = e^2/h_bar
    which is consistent with the results from Lindhard function.

    The reason for nr_orbitals in the prefactor is that every electron
    contribute freely to the conductivity, and we have to take the number
    of electrons into consideration. See eqn. 222-223 of the note for more
    details.

    :param corr_ac: (4, nr_time_steps) complex128 array
        AC correlation function in 4 directions:
        xx, xy, yx, yy, respectively
        Unit should be e^2/h_bar^2 * (eV)^2 * nm^2.
    :param window: window function for integral
    :return: (omegas, ac_cond)
        omegas: (nr_time_steps,) float64 array
        frequencies in eV
        ac_cond: (4, nr_time_steps) complex128 array
        ac conductivity values corresponding to omegas for 4 directions
        (xx, xy, yx, yy, respectively)
        The unit is e^2/(h_bar*nm) in 3d case and e^2/h_bar in 2d case.
    """
    # Get parameters
    tnr = time_steps
    t_step = np.pi / energy_range
    beta = 1 / (k_B_eV * temperature)
    ac_prefactor = (hamiltonian.basis_len / hamiltonian.cell_volume)

    # Allocate working arrays
    omegas = np.array([i * energy_range / tnr for i in range(tnr)],
                        dtype=float)
    ac_cond = np.zeros((4, tnr), dtype=complex)

    # Get real part of AC conductivity
    ac_real = np.zeros(tnr, dtype=float)
    for i in range(tnr):
        omega = omegas.item(i)
        acv = 0.
        for k in range(tnr):
            acv += 2. * window(k + 1, tnr) \
                * math.sin(omega * k * t_step) * corr_ac.item(k).imag
        if omega == 0.:
            acv = 0.
        else:
            acv = ac_prefactor * t_step * acv \
                * (math.exp(-beta * omega) - 1) / omega
        ac_real[i] = acv

    # Get imaginary part of AC conductivity via Kramers-Kronig relations
    # (Hilbert transformation).
    ac_imag = np.zeros(tnr, dtype=float)
    sigma = np.zeros(2 * tnr, dtype=float)
    for i in range(tnr):
        sigma[tnr + i] = ac_real.item(i)
        sigma[tnr - i] = ac_real.item(i)
    ac_imag[:] = np.imag(hilbert(sigma))[tnr:2 * tnr]
    ac_cond = ac_real + 1j * ac_imag

    # Correct for spin
    if not spin:
        ac_cond = 2. * ac_cond

    return omegas, ac_cond



def calc_corr_dos(hamiltonian, num_samples = 1, time_steps = 1024, energy_range = 20.0, seed = 42, wfn_check_step = 128, wfn_check_thr = 1e-9):
    """
    Calculate the correlation function of density of states (DOS) using PyTorch.

    Parameters:
    - num_samples: Number of random samples.
    - time_steps: Number of time steps.
    - rank: Rank of the current process (for distributed processing).
    - seed: Seed for random number generation.
    - wfn_check_step: Check step for wavefunction norm.
    - wfn_check_thr: Threshold for wavefunction norm check.
    - output_file: File to save results.

    Returns:
    - corr: Correlation function of DOS.
    """
    # Get parameters
    time_step = 2 * math.pi / energy_range  # Time step for Chebyshev evolution
    print("Assemblying Hamiltonian")
    H = hamiltonian.HR
    print("Hamiltonian Constructed")
    corr = np.zeros(time_steps + 1, dtype=np.complex64)
    H_rescaled, H_max = rescale_hamiltonian(H)
    bessel = get_bessel_series(time_step,H_max)
    wf_dim = H.shape[0]

    time_step_fs = time_step * hbar_eV * 1e15
    print(f"Time step for propagation: {time_step_fs:7.3f} fs\n")

    # Set random seed
    np.random.seed(seed)

    print("Calculating DOS correlation function.")

    # Loop over the number of random samples
    for i_sample in range(num_samples):
        print(f"Sample {i_sample + 1} of {num_samples}")

        # Generate random initial state
        wf0 = np.random.rand(wf_dim) + 1j * np.random.rand(wf_dim)
        wf0 /= np.linalg.norm(wf0)

        # Calculate initial correlation
        corrval = np.dot(np.conj(wf0), wf0)
        corr[0] += corrval / num_samples

        # Chebyshev time evolution for the first step
        wf_t = chebyshev_time_evolution_H(H_rescaled, wf0, bessel)

        corrval = np.dot(np.conj(wf0), wf_t)
        corr[1] += corrval / num_samples

        # Reference norm for wavefunction checks
        norm_ref = np.dot(np.conj(wf_t), wf_t)

        # Iterate over remaining time steps
        for t in range(2, time_steps + 1):
            if t % wfn_check_step == 0:
                # Wavefunction norm check
                norm_diff = np.abs(np.dot(np.conj(wf_t), wf_t) - norm_ref)
                if norm_diff > wfn_check_thr:
                    print(f"Warning: Wavefunction norm exceeded threshold at timestep {t}: Difference = {norm_diff}")

            # Chebyshev evolution to the next timestep
            wf_t = chebyshev_time_evolution_H(H_rescaled, wf_t, bessel)
            corrval = np.dot(np.conj(wf0), wf_t)
            corr[t] += corrval / num_samples

    # Save the correlation data
    return corr

def calc_corr_ldos(hamiltonian, site_indices, num_samples=1, time_steps=1024, energy_range=20.0, seed=42, wfn_check_step=128, wfn_check_thr=1e-5, backend='torch'):
    """
    Calculate the correlation function of Local Density of States (LDOS) using PyTorch.

    Parameters:
    - site_indices: Indices of the sites to localize the LDOS calculation.
    - num_samples: Number of random samples.
    - time_steps: Number of time steps.
    - seed: Seed for random number generation.
    - wfn_check_step: Check step for wavefunction norm.
    - wfn_check_thr: Threshold for wavefunction norm check.

    Returns:
    - corr: Correlation function of LDOS.
    """
    # Get parameters
    time_step = 2 * math.pi / energy_range  # Time step for Chebyshev evolution
    H = hamiltonian.HR
    corr = torch.zeros(time_steps + 1, dtype=torch.complex64, device=H.device)
    H_rescaled, H_max = rescale_hamiltonian(H)
    bessel = get_bessel_series(time_step, H_max, backend=backend)
    wf_dim = H.shape[0]

    time_step_fs = time_step * hbar_eV * 1e15
    print(f"Time step for propagation: {time_step_fs:7.3f} fs\n")

    # Set random seed
    torch.manual_seed(seed)

    print("Calculating LDOS correlation function.")

    # Loop over the number of random samples
    for i_sample in range(num_samples):
        print(f"Sample {i_sample + 1} of {num_samples}")

        # Generate random initial state localized at specified site indices
        wf_t = torch.rand(wf_dim, dtype=torch.complex64, device=H.device) * (1 + 1j)
        wf0 = torch.zeros_like(wf_t)
        wf0[site_indices] = wf_t[site_indices]  # Localize wavefunction at site indices
        wf0 /= torch.linalg.norm(wf0)  # Normalize the localized state
        wf_t = wf0.clone()  # Initialize wf_t as a copy of wf0 for time evolution

        # Calculate initial correlation
        corrval = torch.dot(wf0.conj(), wf_t)
        corr[0] += corrval / num_samples

        # Reference norm for wavefunction checks
        norm_ref = torch.dot(wf_t.conj(), wf_t)

        # Iterate over remaining time steps
        for t in range(1, time_steps + 1):
            if t % wfn_check_step == 0:
                # Wavefunction norm check
                norm_diff = torch.abs(torch.dot(wf_t.conj(), wf_t) - norm_ref)
                if norm_diff > wfn_check_thr:
                    print(f"Warning: Wavefunction norm exceeded threshold at timestep {t}: Difference = {norm_diff}")

            # Chebyshev evolution to the next timestep
            wf_t = chebyshev_time_evolution_H(H_rescaled, wf_t, bessel)
            corrval = torch.dot(wf0.conj(), wf_t)
            corr[t] += corrval / num_samples

    # Return the LDOS correlation data
    return corr

def calc_dos_from_Cdos(corr_dos, time_steps = 1024, energy_range = 20.0, spin = False):
    """
    Calculate DOS from correlation function.

    Reference: eqn. 16-17 of feature article.

    The unit of dos follows:
        [dos] = [C_DOS] [dt] = h_bar / eV
    So possibly the formula misses a h_bar on the denominator.
    Anyway, the DOS is correct since it is explicitly normalized to 1.

    :return: (energies, dos)
        energies: (2*nr_time_steps,) float64 array
        energies in eV
        dos: (2*nr_time_steps,) float64 array
        DOS in 1/eV
    """
    # Get parameters
    en_step = 0.5 * energy_range / time_steps

    # Allocate working arrays
    energies = np.array([0.5 * i * energy_range / time_steps - energy_range / 2.
                         for i in range(time_steps * 2)], dtype=np.float32)
    dos = np.zeros(time_steps * 2, dtype=np.float32)

    # Get negative time correlation
    corr_neg_time = np.zeros(time_steps * 2, dtype=np.complex64)
    corr_neg_time[time_steps - 1] = corr_dos[0]
    corr_neg_time[2 * time_steps - 1] = window_hanning(time_steps - 1, time_steps) * corr_dos[time_steps]

    indices = np.arange(time_steps - 1)
    hanning_values = window_hanning(indices, time_steps)
    corr_neg_time[time_steps + indices] = hanning_values * corr_dos[indices + 1]
    corr_neg_time[time_steps - indices - 2] = hanning_values * corr_dos[indices + 1].conj()

    # Fourier transform
    corr_fft = np.fft.ifft(corr_neg_time)
    dos[time_steps:] = np.abs(corr_fft[:time_steps])
    dos[:time_steps] = np.abs(corr_fft[time_steps:])

    # Normalize and correct for spin
    dos = dos / (np.sum(dos) * en_step)
    if not spin:
        dos = 2. * dos

    return energies, dos
