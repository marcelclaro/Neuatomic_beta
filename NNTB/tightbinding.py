from NNTB.util import *
from cmath import exp
from scipy import linalg
import torch
import numpy as np
from ase.dft.kpoints import monkhorst_pack
import math
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, inv
from scipy.special import jv  # Bessel function of the first kind
import torch.multiprocessing as mp
import torch.sparse

from concurrent.futures import ProcessPoolExecutor, as_completed

"""
Copyright (c) 2024 Marcel S. Claro

GNU Lesser General Public License v3.0
"""


"""
Some algorithms were adapted from 
https://github.com/deepmodeling/tbplas
https://github.com/dean0x7d/pybinding
BSD licenses
"""


# Function to process a batch of edges
def process_edge_batch_periodic(batch, R_vec_lst, HR_lst, SR_lst, system):
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

            with torch.no_grad():  # Avoid creating computation graph for updates
                HR_lst[Ridx][row_ix, col_ix] += ham
                SR_lst[Ridx][row_ix, col_ix] += overlap
                if torch.all(Rvec == 0) and index[1] != index[0]:
                    HR_lst[Ridx][col_ix, row_ix] += ham
                    SR_lst[Ridx][col_ix, row_ix] += overlap

# Function to create and distribute batches
def process_edges_parallel_periodic(graph, R_vec_lst, HR_lst, SR_lst, system, n_process=4):
    # Convert lists to shared memory tensors
    HR_shared = [hr.share_memory_() for hr in HR_lst]
    SR_shared = [sr.share_memory_() for sr in SR_lst]

    # Prepare data batches
    edge_data = list(zip(
        graph.edge_index.t(), graph.reversed, graph.edge_shift.to(torch.float32), graph.hopping, graph.overlap
    ))
    batch_size = math.ceil(len(edge_data) / n_process)
    num_batches = math.ceil(len(edge_data) / batch_size)
    batches = [edge_data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    # Start multiprocessing
    processes = []
    for batch in batches:
        p = mp.Process(
            target=process_edge_batch_periodic,
            args=(batch, R_vec_lst, HR_shared, SR_shared, system)
        )
        p.start()
        processes.append(p)

    # Join processes
    for p in processes:
        p.join()

    # Return the updated lists
    return HR_shared, SR_shared

# Function to process a batch of edges
def process_edge_batch_sparse(batch, R_vec_lst, system, periodic):
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

            if periodic:
                indices_H.append(torch.stack([torch.full_like(row_ix.flatten(), Ridx), row_ix.flatten(), col_ix.flatten()]))
                values_H.append(ham.flatten())
                indices_S.append(torch.stack([torch.full_like(row_ix.flatten(), Ridx),row_ix.flatten(), col_ix.flatten()]))
                values_S.append(overlap.flatten())

                if torch.all(Rvec == 0) and index[1] != index[0]:
                    indices_H.append(torch.stack([torch.full_like(row_ix.flatten(), Ridx),col_ix.flatten(), row_ix.flatten()]))
                    values_H.append(ham.flatten())
                    indices_S.append(torch.stack([torch.full_like(row_ix.flatten(), Ridx),col_ix.flatten(), row_ix.flatten()]))
                    values_S.append(overlap.flatten())
            else:
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
def process_edges_parallel_sparse(graph, R_vec_lst, system, periodic = True, n_process=4):
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
        future_to_batch = {executor.submit(process_edge_batch_sparse, batch,R_vec_lst, system,periodic): batch for batch in batches}

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


class TBHamiltonian:
    
    def __init__(self, graph, system, periodic = True, orthogonalization = False ,sparse = False, n_process=4):
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

        #Get the element from the one-hot attribute
        elemlst=[system.element_decoding[i] for x in graph.elementonehot.to('cpu') for i, value in enumerate(system.type_onehot) if torch.equal(value, x)]
        ham_template = [(idx, orb) for idx, atom in enumerate(elemlst) for orb in system.orbitals]
        self.basis_len= len(ham_template)
        self.num_orbitals = len(system.orbitals)

        self.cell_volume = torch.linalg.det(torch.tensor(graph.lattice_vectors))
        self.orbital_pos = graph.pos

        if sparse:
            self.R_vec = [torch.tensor([0, 0, 0], dtype=torch.float32)]

            if periodic:
                Rs = graph.edge_shift.to(dtype=torch.float32)

                for Rvec in Rs:
                    Ridx = next((i for i, vec in enumerate(self.R_vec) if torch.equal(vec, Rvec)), None)
                    Ridxneg = next((i for i, vec in enumerate(self.R_vec) if torch.equal(vec, -Rvec)), None)
                    if Ridx is None and Ridxneg is None:
                        self.R_vec.append(Rvec)  # Add R in the list
                        self.R_vec.append(-Rvec)  # Add R in the list
            
            indices_H_all, values_H_all, indices_S_all, values_S_all = process_edges_parallel_sparse(graph, self.R_vec, system, periodic=periodic, n_process=n_process)

            if periodic:
                self.HR = torch.sparse_coo_tensor(indices_H_all, values_H_all, size=(len(self.R_vec),self.basis_len, self.basis_len))
                self.SR = torch.sparse_coo_tensor(indices_S_all, values_S_all, size=(len(self.R_vec),self.basis_len, self.basis_len))
            else:
                self.HR = torch.sparse_coo_tensor(indices_H_all, values_H_all, size=(self.basis_len, self.basis_len))
                self.SR = torch.sparse_coo_tensor(indices_S_all, values_S_all, size=(self.basis_len, self.basis_len))
        else:
            #Hamiltonia H(R) in my usual format
            self.R_vec,self.HR,self.SR = [torch.tensor([0,0,0],dtype=torch.float32)],[torch.zeros((self.basis_len, self.basis_len), dtype=hamtype)],[torch.zeros((self.basis_len, self.basis_len), dtype=hamtype)]
            
            if periodic:
                Rs = graph.edge_shift.to(dtype=torch.float32)
                for Rvec in Rs:
                    Ridx = next((i for i, vec in enumerate(self.R_vec) if torch.equal(vec, Rvec)), None)
                    Ridxneg = next((i for i, vec in enumerate(self.R_vec) if torch.equal(vec, -Rvec)), None)
                    if Ridx == None and Ridxneg == None:
                        self.R_vec.append(Rvec)  # Add R in the list
                        self.R_vec.append(-Rvec)  # Add R in the list
                        self.HR.append(torch.zeros((self.basis_len, self.basis_len), dtype=hamtype))
                        self.HR.append(torch.zeros((self.basis_len, self.basis_len), dtype=hamtype))
                        self.SR.append(torch.zeros((self.basis_len, self.basis_len), dtype=hamtype))
                        self.SR.append(torch.zeros((self.basis_len, self.basis_len), dtype=hamtype))

            self.HR, self.SR = process_edges_parallel_periodic(graph, self.R_vec, self.HR, self.SR, system, n_process=n_process)


            if not periodic:
                self.HR = self.HR[0]
                self.SR = self.SR[0]
                self.R_vec = torch.tensor([0,0,0],dtype=torch.float32)
            else:
                self.HR = torch.stack(self.HR)
                self.SR = torch.stack(self.SR)
                self.R_vec = torch.stack(self.R_vec)
            

        if orthogonalization:
            self.orthogonalize()

    def save_hamiltonian(self, filename):
        """
        Save the Hamiltonian (HR, SR, R_vec, periodic, orthogonal, sparse, basechange) to a file.
        Args:
            filename (str): The file path to save the Hamiltonian.
        """
        data = {
            'HR_data': self.HR._values().cpu().numpy() if self.sparse else self.HR.cpu().numpy(),
            'HR_indices': self.HR._indices().cpu().numpy() if self.sparse else None,
            'SR_data': self.SR._values().cpu().numpy() if self.sparse else self.SR.cpu().numpy(),
            'SR_indices': self.SR._indices().cpu().numpy() if self.sparse else None,
            'R_vec': self.R_vec.cpu().numpy(),
            'periodic': self.periodic,
            'orthogonal': self.orthogonal,
            'sparse': self.sparse,
            'basechange': self.basechange.cpu().numpy() if self.basechange is not None else None,
            'spin': self.spin,
            'type': 'torch'
        }
        np.savez_compressed(filename, **data)

    def load_hamiltonian(self, filename):
        """
        Load the Hamiltonian (HR, SR, R_vec, periodic, orthogonal, sparse, basechange) from a file.
        Args:
            filename (str): The file path to load the Hamiltonian from.
        """
        data = np.load(filename)
        assert data['type'] == 'torch', "File type mismatch. Expected 'torch' but got {data['type']}."
        if self.sparse:
            HR_indices = torch.tensor(data['HR_indices'])
            HR_data = torch.tensor(data['HR_data'])
            self.HR = torch.sparse_coo_tensor(HR_indices, HR_data, size=(len(self.R_vec), self.basis_len, self.basis_len))
            
            SR_indices = torch.tensor(data['SR_indices'])
            SR_data = torch.tensor(data['SR_data'])
            self.SR = torch.sparse_coo_tensor(SR_indices, SR_data, size=(len(self.R_vec), self.basis_len, self.basis_len))
        else:
            self.HR = torch.tensor(data['HR'])
            self.SR = torch.tensor(data['SR'])
        
        self.R_vec = torch.tensor(data['R_vec'])
        self.periodic = bool(data['periodic'])
        self.orthogonal = bool(data['orthogonal'])
        self.sparse = bool(data['sparse'])
        self.spin = bool(data['spin'])
        self.basechange = torch.tensor(data['basechange']) if data['basechange'] is not None else None

    def orthogonalize(self):
        if self.sparse:
                raise ValueError("Orthogonalization not implemented for sparse Hamiltonians.")
        else:
            if self.periodic:
                #Approximation for the general eigenvalue system
                dS = self.SR[0] - torch.eye(self.SR[0].size(0))
                S_1_2 = (torch.eye(dS.size(0))  # Identity matrix
                + (1/2.0) * dS
                - (1/8.0) * dS @ dS
                + (1/16.0) * dS @ dS @ dS
                - (5/128.0) * dS @ dS @ dS @ dS)
                # + (7/256.0) * dS @ dS @ dS @ dS @ dS
                # - (21/1024.0) * dS @ dS @ dS @ dS @ dS @ dS
                # + (33/2048.0) * dS @ dS @ dS @ dS @ dS @ dS @ dS)

                S_1_2_inv = torch.linalg.pinv(S_1_2)

                self.HR = [S_1_2_inv @ HR @ S_1_2_inv for HR in self.HR]
                self.SR = [S_1_2_inv @ SR @ S_1_2_inv for SR in self.SR]
                self.basechange = S_1_2_inv
            else:
                #Approximation for the general eigenvalue system
                dS = self.SR - torch.eye(self.SR.size(0))
                S_1_2 = (torch.eye(dS.size(0))  # Identity matrix
                + (1/2.0) * dS
                - (1/8.0) * dS @ dS
                + (1/16.0) * dS @ dS @ dS
                - (5/128.0) * dS @ dS @ dS @ dS)
                # + (7/256.0) * dS @ dS @ dS @ dS @ dS
                # - (21/1024.0) * dS @ dS @ dS @ dS @ dS @ dS
                # + (33/2048.0) * dS @ dS @ dS @ dS @ dS @ dS @ dS)

                S_1_2_inv = torch.linalg.pinv(S_1_2)

                self.HR = S_1_2_inv @ self.HR @ S_1_2_inv
                self.SR = torch.eye(self.SR.size(0))
                self.basechange = S_1_2_inv
            self.orthogonal = True


def bandfromgraph_torch(graph, system, kpoints = None):
    """
    Compute the band structure from a graph representation using PyTorch.
    Args:
        graph (Graph): The graph representation of the system, containing attributes such as spin, elementonehot, edge_shift, edge_index, reversed, hopping, and overlap.
        system (System): The system object containing attributes such as element_decoding, type_onehot, orbitalstrings, spin, and band_kpoints.
        kpoints (list, optional): List of k-points for which to compute the band structure. If None, system.band_kpoints will be used.
    Returns:
        torch.Tensor: A tensor containing the eigenvalues (band structure) for each k-point.
    """
    Ham = TBHamiltonian(graph, system)
               
    eigenvalues = [] #Band eigenvalues
    if kpoints is None:
        kpoints = system.band_kpoints
    for k in kpoints: #for each k-point append list of eigenvalues
        # The H(R) -> H(k) transformation
        sum_H = sum([torch.exp(2j * torch.pi * torch.dot(k, R))*HR for R, HR in zip(Ham.R_vec, Ham.HR)]) #if np.all(R >= 0)
        
        if Ham.SR is not None: #case overlap matrix is not identity
            sum_S = sum([torch.exp(2j * torch.pi * torch.dot(k, R))*SR for R, SR in zip(Ham.R_vec, Ham.SR)])


            #Approximation for the general eigenvalue system
            dS = sum_S - torch.eye(sum_S.size(0))
            S_1_2 = (torch.eye(dS.size(0))  # Identity matrix
            + (1/2.0) * dS
            - (1/8.0) * dS @ dS
            + (1/16.0) * dS @ dS @ dS
            - (5/128.0) * dS @ dS @ dS @ dS)
            # + (7/256.0) * dS @ dS @ dS @ dS @ dS
            # - (21/1024.0) * dS @ dS @ dS @ dS @ dS @ dS
            # + (33/2048.0) * dS @ dS @ dS @ dS @ dS @ dS @ dS)

            S_1_2_inv = torch.linalg.pinv(S_1_2)

            sum_H = S_1_2_inv @ sum_H @ S_1_2_inv
            
            eigenval = torch.linalg.eigvalsh(sum_H)
            eigenvalues.append(eigenval)

        else:
            eigenval = torch.linalg.eigvalsh(sum_H)
            eigenvalues.append(eigenval) #solve system
    
    return eigenvalues


def bandfromgraph_torch_sparse(graph, system, kpoints=None, number_bands=13):
    """
    Compute the band structure from a graph representation using PyTorch.
    Args:
        graph (Graph): The graph representation of the system, containing attributes such as spin, elementonehot, edge_shift, edge_index, reversed, hopping, and overlap.
        system (System): The system object containing attributes such as element_decoding, type_onehot, orbitalstrings, spin, and band_kpoints.
        kpoints (list, optional): List of k-points for which to compute the band structure. If None, system.band_kpoints will be used.
    Returns:
        torch.Tensor: A tensor containing the eigenvalues (band structure) for each k-point.
    """
    Ham = TBHamiltonian(graph, system,sparse=True)
    basis_len = Ham.basis_len

    eigenvalues = []  # Band eigenvalues
    if kpoints is None:
        kpoints = system.band_kpoints
    for k in kpoints:  # for each k-point append list of eigenvalues
        # The H(R) -> H(k) transformation

        # Initialize lists to collect combined indices and values
        combined_indices = []
        combined_values = []

        # Loop over each sparse matrix and apply the phase factor
        for R, HR in zip(Ham.R_vec, Ham.HR):
            # Compute phase factor
            phase_factor = torch.exp(2j * torch.pi * torch.dot(k, R))

            # Scale the values of HR by the phase factor
            scaled_values = phase_factor * HR._values()

            # Collect indices and scaled values
            combined_indices.append(HR._indices())
            combined_values.append(scaled_values)

        # Concatenate all indices and values
        combined_indices = torch.cat(combined_indices, dim=1)
        combined_values = torch.cat(combined_values)

        # Sum the matrices by coalescing
        sum_H = torch.sparse_coo_tensor(combined_indices, combined_values, size=(basis_len, basis_len)).coalesce()
        
        # Convert sum_H and sum_S to scipy sparse matrices
        sum_H_sparse = sp.csr_matrix((sum_H._values().cpu().numpy(), (sum_H._indices()[0].cpu().numpy(), sum_H._indices()[1].cpu().numpy())), shape=(basis_len, basis_len))
        
        if Ham.SR is not None:  # case overlap matrix is not identity
            combined_indices = []
            combined_values = []
            for R, SR in zip(Ham.R_vec, Ham.SR):
                phase_factor = torch.exp(2j * torch.pi * torch.dot(k, R))

                scaled_values = phase_factor * SR._values()

                combined_indices.append(SR._indices())
                combined_values.append(scaled_values)
            combined_indices = torch.cat(combined_indices, dim=1)
            combined_values = torch.cat(combined_values)
            sum_S = torch.sparse_coo_tensor(combined_indices, combined_values, size=(basis_len, basis_len)).coalesce()

            sum_S_sparse = sp.csr_matrix((sum_S._values().cpu().numpy(), (sum_S._indices()[0].cpu().numpy(), sum_S._indices()[1].cpu().numpy())), shape=(basis_len, basis_len))


            # Approximation for the general eigenvalue system
            dS = sum_S_sparse - sp.eye(sum_S_sparse.shape[0])
            S_1_2 = (sp.eye(dS.shape[0])  # Identity matrix
                    + (1 / 2.0) * dS
                    - (1 / 8.0) * dS @ dS
                    + (1 / 16.0) * dS @ dS @ dS
                    - (5 / 128.0) * dS @ dS @ dS @ dS)
            # + (7/256.0) * dS @ dS @ dS @ dS @ dS
            # - (21/1024.0) * dS @ dS @ dS @ dS @ dS @ dS
            # + (33/2048.0) * dS @ dS @ dS @ dS @ dS @ dS @ dS)

            S_1_2_inv = sp.linalg.inv(S_1_2)

            sum_H_sparse = S_1_2_inv @ sum_H_sparse @ S_1_2_inv

            eigenval,_ = sp.linalg.eigsh(sum_H_sparse,k=number_bands)
            eigenvalues.append(torch.tensor(eigenval, dtype=torch.float32))

        else:
            eigenval,_ = sp.linalg.eigsh(sum_H,k=number_bands)
            eigenvalues.append(eigenval)  # solve system

    return eigenvalues

def eigen_calc(hamiltonian, kpoints = None):
    eigenvalues = [] #Band eigenvalues
    eigenvectors = [] #Band eigenvalues
    if kpoints is None:
        print("No kpoints given, gamma-point calculation")
        kpoints = torch.tensor([[0,0,0]],dtype=torch.float32)
    for i_k, k in enumerate(kpoints): #for each k-point append list of eigenvalues
        # The H(R) -> H(k) transformation
        print(f"\rCalculating k-point {i_k+1}/{len(kpoints)}", end="")
        sum_H = sum([torch.exp(2j * torch.pi * torch.dot(k, R))*HR for R, HR in zip(hamiltonian.R_vec, hamiltonian.HR)]) if hamiltonian.periodic else hamiltonian.HR
        
        if hamiltonian.SR is not None: #case overlap matrix is not identity
            sum_S = sum([torch.exp(2j * torch.pi * torch.dot(k, R))*SR for R, SR in zip(hamiltonian.R_vec, hamiltonian.SR)]) if hamiltonian.periodic else hamiltonian.SR


            #Approximation for the general eigenvalue system
            dS = sum_S - torch.eye(sum_S.size(0))
            S_1_2 = (torch.eye(dS.size(0))  # Identity matrix
            + (1/2.0) * dS
            - (1/8.0) * dS @ dS
            + (1/16.0) * dS @ dS @ dS
            - (5/128.0) * dS @ dS @ dS @ dS)
            # + (7/256.0) * dS @ dS @ dS @ dS @ dS
            # - (21/1024.0) * dS @ dS @ dS @ dS @ dS @ dS
            # + (33/2048.0) * dS @ dS @ dS @ dS @ dS @ dS @ dS)

            S_1_2_inv = torch.linalg.pinv(S_1_2)

            sum_H = S_1_2_inv @ sum_H @ S_1_2_inv

            eigenval, eigenvec = torch.linalg.eigh(sum_H)
            eigenvalues.append(eigenval)
            eigenvectors.append(S_1_2_inv @ eigenvec)


        else:
            eigenval = torch.linalg.eigh(sum_H)
            eigenvalues.append(eigenval)
            eigenvectors.append(eigenvec)
    
    return torch.stack(eigenvalues,dim=0),torch.stack(eigenvectors,dim=0)

def gaussian(x, mu, sigma):
    """
    Gaussian type broadening function.

    :param x: incoming x
    :param mu: center of the Gaussian function
    :param sigma: half-width of the Gaussian function
    :return: normalized Gaussian function value at each x
    """
    part_a = 1.0 / (sigma * math.sqrt(2 * torch.pi))
    part_b = torch.exp(-(x - mu)**2 / (2 * sigma**2))
    return part_a * part_b


def lorentzian(x, mu, sigma):
    """
    Lorentzian type broadening function.

    :param x: incoming x
    :param mu: center of the Lorentzian function
    :param sigma: half-width of the Lorentzian function
    :return: normalized Lorentzian function value at each x
    """
    part_a = 1.0 / (torch.pi * sigma)
    part_b = sigma**2 / ((x - mu)**2 + sigma**2)
    return part_a * part_b


def calc_dos( hamiltonian,
                MPgrid = (4,4,4),
                e_min = None,
                e_max = None,
                e_step = 0.05,
                sigma = 0.05,
                basis = "Gaussian"):
    """
    Calculate density of states for given energy range and step.

    :param k_points: (num_kpt, 3) float64 array
        FRACTIONAL coordinates of k-points
    :param e_min: lower bound of the energy range in eV
    :param e_max: upper hound of the energy range in eV
    :param e_step: energy step in eV
    :param sigma: broadening parameter in eV
    :param basis: basis function to approximate the Delta function
    :param g_s: spin degeneracy
    :param kwargs: arguments for 'calc_bands'
    :return: (energies, dos)
        energies: (num_grid,) float64 array
        energy grid corresponding to e_min, e_max and e_step
        dos: (num_grid,) float64 array
        density of states in states/eV
    :raises ValueError: if basis is neither Gaussian nor Lorentzian,
        or the solver is neither lapack nor arpack
    """

    #Spin degenerecency
    if hamiltonian.spin:
        g_s = 1.0
    else:
        g_s = 2.0

    kpoints = torch.tensor(monkhorst_pack(MPgrid),dtype=torch.float32)

    # Get the band energies and projection
    eigenvals,_ = eigen_calc(hamiltonian, kpoints)

    # Create energy grid
    if e_min is None:
        e_min = torch.min(eigenvals)
    if e_max is None:
        e_max = torch.max(eigenvals)
    num_grid = int((e_max - e_min) / e_step)
    energies = torch.linspace(e_min, e_max, num_grid + 1)

    # Evaluate DOS by collecting contributions from all energies
    dos = torch.zeros(energies.shape, dtype=torch.float64)
    if basis == "Gaussian":
        basis_func = gaussian
    elif basis == "Lorentzian":
        basis_func = lorentzian
    else:
        raise ValueError(f"Illegal basis function {basis}")

    # Collect contributions
    for i_k in range(len(kpoints)):
        for i_b, eng_i in enumerate(eigenvals[i_k,:]):
            dos += basis_func(energies, eng_i, sigma)

    # Re-normalize dos
    # For each energy in bands, we use a normalized Gaussian or Lorentzian
    # basis function to approximate the Delta function. Totally, there are
    # bands.size basis functions. So we divide dos by this number.
    dos /= hamiltonian.basis_len
    dos *= g_s #Spin degenerecency
    return energies, dos

def calc_spindos(hamiltonian,system,graph,
                MPgrid = (4,4,4),
                e_min = None,
                e_max = None,
                e_step = 0.05,
                sigma = 0.05,
                basis = "Gaussian",
                g_s: int = 1):
    """
    Calculate density of states for given energy range and step.

    :param k_points: (num_kpt, 3) float64 array
        FRACTIONAL coordinates of k-points
    :param e_min: lower bound of the energy range in eV
    :param e_max: upper hound of the energy range in eV
    :param e_step: energy step in eV
    :param sigma: broadening parameter in eV
    :param basis: basis function to approximate the Delta function
    :param g_s: spin degeneracy
    :param kwargs: arguments for 'calc_bands'
    :return: (energies, dos)
        energies: (num_grid,) float64 array
        energy grid corresponding to e_min, e_max and e_step
        dos: (num_grid,) float64 array
        density of states in states/eV
    :raises ValueError: if basis is neither Gaussian nor Lorentzian,
        or the solver is neither lapack nor arpack
    """
    kpoints = torch.tensor(monkhorst_pack(MPgrid),dtype=torch.float32)

    # Get the band energies and projection
    eigenvals,eigenvecs = eigen_calc(hamiltonian, kpoints)

    # Create energy grid
    if e_min is None:
        e_min = torch.min(eigenvals)
    if e_max is None:
        e_max = torch.max(eigenvals)
    num_grid = int((e_max - e_min) / e_step)
    energies = torch.linspace(e_min, e_max, num_grid + 1)

    # Evaluate DOS by collecting contributions from all energies
    dosup = torch.zeros(energies.shape, dtype=torch.float64)
    dosdown = torch.zeros(energies.shape, dtype=torch.float64)
    if basis == "Gaussian":
        basis_func = gaussian
    elif basis == "Lorentzian":
        basis_func = lorentzian
    else:
        raise ValueError(f"Illegal basis function {basis}")
    
    spinup_idx = [i for i, orb in enumerate(system.orbitals*len(graph.x)) if orb[3] == -1]
    spindown_idx = [i for i, orb in enumerate(system.orbitals*len(graph.x)) if orb[3] == 1]

    # Collect contributions
    for i_k in range(len(kpoints)):
        for i_b, eng_i in enumerate(eigenvals[i_k,:]):
            delta = basis_func(energies, eng_i, sigma)
            dosup +=  delta * torch.sum(torch.abs(eigenvecs[i_k,i_b,spinup_idx])**2)
            dosdown += delta * torch.sum(torch.abs(eigenvecs[i_k,i_b,spindown_idx])**2)

    # Re-normalize dos
    # For each energy in bands, we use a normalized Gaussian or Lorentzian
    # basis function to approximate the Delta function. Totally, there are
    # bands.size basis functions. So we divide dos by this number.
    dosup /= hamiltonian.basis_len
    dosdown /= hamiltonian.basis_len
    
    return energies, dosup, dosdown


def calc_ac_cond(hamiltonian,graph, omegas_eV, MPgrid = (4,4,4),temperature = 300, delta = 0.005, fermilevel = 0.0, component = "xx", dimension = 3):
    """
    Calculate AC conductivity using Kubo-Greenwood formula.

    Reference: section 12.2 of Wannier90 user guide.

    """
    # Aliases for variables
    if component not in [a+b for a in "xyz" for b in "xyz"]:
        raise ValueError(f"Illegal component {component}")
    comp = np.array(["xyz".index(_) for _ in component], dtype=np.int32)

    kmesh_frac = torch.tensor(monkhorst_pack(MPgrid),dtype=torch.float32)
    recip_lat_vec = torch.linalg.pinv(torch.tensor(graph.lattice_vectors*0.1,dtype=torch.float32)).T
    conversion_matrix = recip_lat_vec.T
    kmesh = torch.matmul(conversion_matrix,kmesh_frac.T).T

    beta = 1 / (k_B_eV * temperature)

    # Get eigenvalues and eigenstates
    bands, states = eigen_calc(hamiltonian,kmesh_frac)

    # Allocate working arrays
    num_kpt = kmesh_frac.shape[0]
    num_omega = omegas_eV.size(0)
    num_bands = bands.size(1)
    delta_eng = torch.zeros((num_kpt, num_bands, num_bands), dtype=torch.float32)
    prod_df = torch.zeros((num_kpt, num_bands, num_bands), dtype=torch.complex64)
    ac_cond = torch.zeros(num_omega, dtype=torch.complex64)

    if hamiltonian.spin:
        g_s = 1.0
    else:
        g_s = 2.0
    
    for ik in range(num_kpt):
        # Assuming num_bands, ik, hamiltonian, kmesh, and comp are defined
        vmat1 = torch.zeros((num_bands, num_bands), dtype=torch.complex64)
        vmat2 = torch.zeros((num_bands, num_bands), dtype=torch.complex64)
        hop_dr = torch.zeros((num_bands, num_bands), dtype=torch.float32)
        # Calculate orbital indices for each band
        orbital_indices = torch.div(torch.arange(num_bands), hamiltonian.num_orbitals, rounding_mode='floor')

        # Get the orbital positions for each band
        orbital_pos_1 = hamiltonian.orbital_pos[orbital_indices].unsqueeze(1)
        orbital_pos_2 = hamiltonian.orbital_pos[orbital_indices].unsqueeze(0)

        # Compute the difference in positions in a vectorized manner
        hop_dr = orbital_pos_1 - orbital_pos_2

        k_dot_r = torch.einsum('k,ijk->ij', kmesh[ik], hop_dr)

        phase = (torch.cos(k_dot_r) + 1j * torch.sin(k_dot_r)) * hamiltonian.HR[0]

        # Perform the operation only in the upper triangular part
        upper_tri_indices = torch.triu_indices(num_bands, num_bands, offset=1)
        vmat1[upper_tri_indices[0], upper_tri_indices[1]] += 1j * phase[upper_tri_indices[0], upper_tri_indices[1]] * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[0]]
        vmat1[upper_tri_indices[1], upper_tri_indices[0]] += -1j * phase[upper_tri_indices[0], upper_tri_indices[1]].conj() * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[0]]
        vmat2[upper_tri_indices[0], upper_tri_indices[1]] += 1j * phase[upper_tri_indices[0], upper_tri_indices[1]] * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[1]]
        vmat2[upper_tri_indices[1], upper_tri_indices[0]] += -1j * phase[upper_tri_indices[0], upper_tri_indices[1]].conj() * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[1]]  

        # Build delta_eng and prod_df
        # Step 1: Precompute energy differences and Fermi-Dirac factors
        eng_m = bands[ik].unsqueeze(1)               # Shape: [num_bands, 1]
        eng_n = bands[ik].unsqueeze(0)               # Shape: [1, num_bands]
        delta_eng[ik] = eng_m - eng_n                    # Shape: [num_bands, num_bands]

        f_m = 1.0 / (1.0 + torch.exp(beta * (eng_m - fermilevel)))  # Shape: [num_bands, 1]
        f_n = 1.0 / (1.0 + torch.exp(beta * (eng_n - fermilevel)))  # Shape: [1, num_bands]

        # Step 2: Calculate `prod1` and `prod2` with matrix multiplications
        # Reshape states for compatibility with `matmul`
        states_conj = states[ik].conj()  # Shape: [num_bands, num_bands, num_bands]

        # `prod1` is the sum of (states_conj @ vmat1 @ states) across ib1 and ib2
        prod1 = torch.einsum('ni,ij,mj->mn', states_conj, vmat1, states[ik])  # Shape: [num_bands, num_bands, num_bands]
        prod2 = torch.einsum('mi,ij,nj->mn', states_conj, vmat2, states[ik])

        # Step 3: Mask where `delta_eng` is small, to avoid division by very small values
        mask = torch.abs(delta_eng[ik]) >= 1.0e-7

        # Step 4: Calculate `prod_df` using broadcasting
        prod_df[ik][mask] = prod1[mask] * prod2[mask] * (f_m - f_n)[mask] / delta_eng[ik][mask]
   
    for iw in range(num_omega):
        print(f"\rCalculating omega point {iw+1}/{num_omega}", end="")
        omega = omegas_eV[iw]
        ac_sum = 0.0
        ac_sum = torch.sum( prod_df / (delta_eng - omega - 1j * delta) )
        ac_cond[iw] = ac_sum

    # Multiply prefactor
    # NOTE: there is not such g_s factor in the reference.
    if dimension == 3:
        volume = hamiltonian.cell_volume 
        prefactor = g_s * 1j / (volume * num_kpt)
    else:
        raise NotImplementedError(f"Dimension {dimension} not "
                                    f"implemented")
    ac_cond *= prefactor
    return ac_cond

def calc_epsilon_q0(omegas_eV,
                    ac_cond, background_epsilon = 1.0) -> np.ndarray:
    """
    Calculate dielectric function from AC conductivity for q=0.

    :param omegas_eV: (num_omega,) float64 array
        energies in eV
    :param ac_cond: (num_omega,) complex128 array
        AC conductivity in e**2/(h_bar*nm) in 3d case
    :return: (num_omega,) complex128 array
        relative dielectric function
    :raises ValueError: if dimension is not 3
    """
    prefactor = 4 * torch.pi / (background_epsilon * eps_0)
    return 1 + 1j * prefactor * ac_cond / omegas_eV

def calc_sigma_dc_E(hamiltonian, graph, MPgrid=(4,4,4), temperature=300, delta=0.005, mu=0.0, component="xx", dimension=3, num_bins=100):
    """
    Calculate DC conductivity as a function of energy using the Kubo-Greenwood formula.
    """

    # Aliases for variables
    if component not in [a+b for a in "xyz" for b in "xyz"]:
        raise ValueError(f"Illegal component {component}")
    comp = np.array(["xyz".index(_) for _ in component], dtype=np.int32)

    kmesh_frac = torch.tensor(monkhorst_pack(MPgrid), dtype=torch.float32)
    recip_lat_vec = torch.linalg.pinv(torch.tensor(graph.lattice_vectors * 0.1, dtype=torch.float32)).T
    conversion_matrix = recip_lat_vec.T
    kmesh = torch.matmul(conversion_matrix, kmesh_frac.T).T

    beta = 1 / (k_B_eV * temperature)

    # Get eigenvalues and eigenstates
    bands, states = eigen_calc(hamiltonian, kmesh_frac)

    # Allocate working arrays
    num_kpt = kmesh_frac.shape[0]
    num_bands = bands.size(1)
    delta_eng = torch.zeros((num_kpt, num_bands, num_bands), dtype=torch.float32)
    prod_df = torch.zeros((num_kpt, num_bands, num_bands), dtype=torch.complex64)
    dc_cond_kpt_energy = torch.zeros((num_kpt, num_bands), dtype=torch.float32)  # DC conductivity at each k-point and band

    if hamiltonian.spin:
        g_s = 1.0
    else:
        g_s = 2.0

    # Loop over k-points
    for ik in range(num_kpt):
        # Initialize matrices for velocity operators
        vmat1 = torch.zeros((num_bands, num_bands), dtype=torch.complex64)
        vmat2 = torch.zeros((num_bands, num_bands), dtype=torch.complex64)
        hop_dr = torch.zeros((num_bands, num_bands), dtype=torch.float32)

        # Calculate orbital positions
        orbital_indices = torch.div(torch.arange(num_bands), hamiltonian.num_orbitals, rounding_mode='floor')
        orbital_pos_1 = hamiltonian.orbital_pos[orbital_indices].unsqueeze(1)
        orbital_pos_2 = hamiltonian.orbital_pos[orbital_indices].unsqueeze(0)
        hop_dr = orbital_pos_1 - orbital_pos_2

        # Calculate phase for hopping terms
        k_dot_r = torch.einsum('k,ijk->ij', kmesh[ik], hop_dr)
        phase = (torch.cos(k_dot_r) + 1j * torch.sin(k_dot_r)) * hamiltonian.HR[0]

        # Fill velocity matrices
        upper_tri_indices = torch.triu_indices(num_bands, num_bands, offset=1)
        vmat1[upper_tri_indices[0], upper_tri_indices[1]] += 1j * phase[upper_tri_indices[0], upper_tri_indices[1]] * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[0]]
        vmat1[upper_tri_indices[1], upper_tri_indices[0]] += -1j * phase[upper_tri_indices[0], upper_tri_indices[1]].conj() * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[0]]
        vmat2[upper_tri_indices[0], upper_tri_indices[1]] += 1j * phase[upper_tri_indices[0], upper_tri_indices[1]] * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[1]]
        vmat2[upper_tri_indices[1], upper_tri_indices[0]] += -1j * phase[upper_tri_indices[0], upper_tri_indices[1]].conj() * hop_dr[upper_tri_indices[0], upper_tri_indices[1], comp[1]]

        # Calculate energy differences and Fermi-Dirac factors
        eng_m = bands[ik].unsqueeze(1)
        eng_n = bands[ik].unsqueeze(0)
        delta_eng[ik] = eng_m - eng_n
        f_m = 1.0 / (1.0 + torch.exp(beta * (eng_m - mu)))
        f_n = 1.0 / (1.0 + torch.exp(beta * (eng_n - mu)))

        # Calculate velocity matrix product for DC limit
        states_conj = states[ik].conj()
        prod1 = torch.einsum('ni,ij,mj->mn', states_conj, vmat1, states[ik])
        prod2 = torch.einsum('mi,ij,nj->mn', states_conj, vmat2, states[ik])

        # Avoid division by very small values in delta_eng
        mask = torch.abs(delta_eng[ik]) >= 1.0e-7
        prod_df[ik][mask] = prod1[mask] * prod2[mask] * (f_m - f_n)[mask] / delta_eng[ik][mask]

        # Sum DC conductivity contributions for each band at k-point
        dc_cond_kpt_energy[ik] = torch.sum(
                                            prod_df[ik] / (delta_eng[ik]  + delta ** 2),
                                            dim=1
                                        )

    # Energy binning for sigma_DC(E)
    min_energy, max_energy = bands.min(), bands.max()
    energy_bins = torch.linspace(min_energy, max_energy, num_bins)
    sigma_dc_E = torch.zeros(num_bins, dtype=torch.float32)

    # Aggregate DC conductivity into bins
    for ik in range(num_kpt):
        for ib in range(num_bands):
            energy = bands[ik, ib]
            bin_index = torch.bucketize(energy, energy_bins) - 1
            sigma_dc_E[bin_index] += dc_cond_kpt_energy[ik, ib]

    # Normalize by k-points and volume prefactor
    if dimension == 3:
        volume = hamiltonian.cell_volume
        prefactor = g_s / (volume * num_kpt)
    else:
        raise NotImplementedError(f"Dimension {dimension} not implemented")
    sigma_dc_E *= prefactor

    return energy_bins, sigma_dc_E
