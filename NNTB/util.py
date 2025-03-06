import numpy as np
import torch
from e3nn.o3 import Irreps, Irrep, wigner_3j, FullTensorProduct


"""
Copyright (c) 2024 Marcel S. Claro

GNU Lesser General Public License v3.0
"""


"""
Convert
"""
# convert Rydberg Constant to Electron-volt
Ry_to_eV = 13.605698066

# convert Angstrom to Bohr Radius
Ang_to_Bohr = 1.8897259886

# convert Joule to Electron-volt
J_to_eV =  6.241506363094e18


"""
Constant
"""
# Boltzmann constant
k_B_SI = 1.380649e-23    # J/K
k_B_eV = k_B_SI * J_to_eV  # eV/K
k_B_Ry = k_B_SI * J_to_eV / Ry_to_eV # Ry/K

# elementary charge
elem_charge_SI = 1.60217662e-19  # C

# Planck constant
h_SI = 6.62607004e-34  # m^2 kg / s
hbar_SI = 1.054571628e-34  # m^2 kg / s
hbar_eV = hbar_SI * J_to_eV  # eV s

# von Klitzing constant = h / e^2 
R_k = 25812.80745  # \omega

# Norm 1.0 numpy vectors
x_vec = np.array([1,0,0])
y_vec = np.array([0,1,0])
z_vec = np.array([0,0,1])


# Dielectric constant of vacuum in eV/nm/q units
eps_0 = 0.6944615417149689


# Function to convert Cartesian coordinates to spherical coordinates
def cartesian_to_spherical(cartesian):
    r = np.linalg.norm(cartesian)
    if r != 0:
        theta = np.arccos(cartesian[2] / r)
        phi = np.arctan2(cartesian[1], cartesian[0])
    else:
        theta = 0
        phi = 0
    return r, theta, phi

#Convert a string to a list of tuples (n,l,m,s) orbitals coreesponding to that string in abacus order m=0,1,-1,2,-2
def orbitals_from_str_abacus(input_string, spin = False):
    """
    Generate a list of orbitals based on an input string and optional spin consideration.
    Args:
        input_string (str): A string where each pair of characters represents an integer and a key.
                            The integer indicates the number of orbitals, and the key indicates the type of orbital ('s', 'p', 'd', 'f').
        spin (bool, optional): If True, includes spin in the orbitals. Defaults to False.
    Returns:
        list: A list of tuples representing the orbitals. Each tuple contains:
                - Principal quantum number (n)
                - Azimuthal quantum number (l)
                - Magnetic quantum number (m)
                - Spin quantum number (if spin is True)
    """
    orb_radials = {}
    orbitals = []
    # Iterate over the string with a step of 2 to get pairs of integer and key
    for i in range(0, len(input_string), 2):
        # Extract integer and key from the current pair
        current_int = int(input_string[i])
        current_key = input_string[i + 1]
        
        # Add the pair to the dictionary
        orb_radials[current_key] = current_int
    if spin:
        #TODO: Check s order
        #add orbitals s
        if 's' in orb_radials:
            for n in range(1, orb_radials['s'] + 1):
                orbitals += [(n,0,0,+1),(n,0,0,-1)]
        #add orbitals p and so on..
        if 'p' in orb_radials:
            for n in range(1, orb_radials['p'] + 1):
                orbitals += [(n,1,0,+1),(n,1,0,-1)]
                for m in range(1,1+1):
                        orbitals += [(n,1,m,+1),(n,1,m,-1),(n,1,-m,+1),(n,1,-m,-1)]
        if 'd' in orb_radials:
            for n in range(1, orb_radials['d'] + 1):
                orbitals += [(n,2,0,+1),(n,2,0,-1)]
                for m in range(1,2+1):
                        orbitals += [(n,2,m,+1),(n,2,m,-1),(n,2,-m,+1),(n,2,-m,-1)]
        if 'f' in orb_radials:
            for n in range(1, orb_radials['f'] + 1):
                orbitals += [(n,3,0,+1),(n,3,0,-1)]
                for m in range(1,3+1):
                        orbitals += [(n,3,m,+1),(n,3,m,-1),(n,3,-m,+1),(n,3,-m,-1)]

    else:   
        #add orbitals s
        if 's' in orb_radials:
            for n in range(1, orb_radials['s'] + 1):
                orbitals += [(n,0,0)]
        #add orbitals p and so on..
        if 'p' in orb_radials:
            for n in range(1, orb_radials['p'] + 1):
                orbitals += [(n,1,0)]
                for m in range(1,1+1):
                        orbitals += [(n,1,m),(n,1,-m)]
        if 'd' in orb_radials:
            for n in range(1, orb_radials['d'] + 1):
                orbitals += [(n,2,0)]
                for m in range(1,2+1):
                        orbitals += [(n,2,m),(n,2,-m)]
        if 'f' in orb_radials:
            for n in range(1, orb_radials['f'] + 1):
                orbitals += [(n,3,0)]
                for m in range(1,3+1):
                        orbitals += [(n,3,m),(n,3,-m)]

    return orbitals

#Convert a string to a list of tuples (n,l,m) orbitals coreesponding to that string m=..-2,-1,0,1,2...
def orbitals_from_str_yzx(input_string, spin=False):
    """
    Generate a list of orbitals from a given input string.
    The input string should contain pairs of integers and keys, where each pair
    represents the number of orbitals for a specific type (s, p, d, f). The function
    will parse this string and generate a list of orbitals accordingly.
    Args:
        input_string (str): A string containing pairs of integers and keys.
                            For example, "2s3p1d" means 2 s-orbitals, 3 p-orbitals, and 1 d-orbital.
        spin (bool, optional): If True, include spin in the orbitals. Defaults to False.
    Returns:
        list: A list of tuples representing the orbitals. Each tuple contains:
                - n (int): Principal quantum number.
                - l (int): Azimuthal quantum number.
                - m (int): Magnetic quantum number.
                - s (int, optional): Spin quantum number (included if spin is True).
    """
    orb_radials = {}
    orbitals = []
    # Iterate over the string with a step of 2 to get pairs of integer and key
    for i in range(0, len(input_string), 2):
        # Extract integer and key from the current pair
        current_int = int(input_string[i])
        current_key = input_string[i + 1]
        
        # Add the pair to the dictionary
        orb_radials[current_key] = current_int
        
    if spin:
        for s in [1,-1]:  #TODO: Check s order
            #add orbitals s
            if 's' in orb_radials:
                orbitals += [(n, 0, 0,s) for n in range(1, orb_radials['s'] + 1)]
            #add orbitals p and so on..
            if 'p' in orb_radials:
                for n in range(1, orb_radials['p'] + 1):
                    for m in range(-(1),(1+1)):
                        orbitals += [(n,1,m,s)]

            if 'd' in orb_radials:
                for n in range(1, orb_radials['d'] + 1):
                    for m in range(-(2),(2+1)):
                        orbitals += [(n,2,m,s)]
            if 'f' in orb_radials:
                for n in range(1, orb_radials['f'] + 1):
                    for m in range(-(3),(3+1)):
                        orbitals += [(n,3,m,s)]
    else:
        #add orbitals s
        if 's' in orb_radials:
            orbitals += [(n, 0, 0) for n in range(1, orb_radials['s'] + 1)]
        #add orbitals p and so on..
        if 'p' in orb_radials:
            for n in range(1, orb_radials['p'] + 1):
                for m in range(-(1),(1+1)):
                    orbitals += [(n,1,m)]
        if 'd' in orb_radials:
            for n in range(1, orb_radials['d'] + 1):
                for m in range(-(2),(2+1)):
                    orbitals += [(n,2,m)]
        if 'f' in orb_radials:
            for n in range(1, orb_radials['f'] + 1):
                for m in range(-(3),(3+1)):
                    orbitals += [(n,3,m)]
    return orbitals


# Gets the rotatio matrix that align it to y so only m=0 components are != 0
#Based on eSCN code
"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
def edge_rot_mat(edge_distance_vec):
    """
    Computes a 3D rotation matrix for each edge vector in the input tensor.
    This function takes a tensor of edge distance vectors and computes a corresponding
    3D rotation matrix for each vector. The rotation matrix aligns the edge vector with
    the x-axis. If any edge vector is zero, the identity matrix is used for those cases.
    Args:
        edge_distance_vec (torch.Tensor): A tensor of shape (N, 3) representing the edge
                                          distance vectors.
    Returns:
        torch.Tensor: A tensor of shape (N, 3, 3) containing the 3D rotation matrices for
                      each edge vector.
    """
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1)+1e-8)
    
    # Check if any norm_x is zero, and if so, use the identity matrix for those cases
    zero_mask = edge_vec_0_distance == 0

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)
    
    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    
    
    # If any norm_x was zero, set the corresponding rotation matrix to identity
    identity_matrix = torch.eye(3).unsqueeze(0).repeat(edge_rot_mat_inv.size(0), 1, 1)
    edge_rot_mat_inv = torch.where(zero_mask.view(-1, 1, 1), identity_matrix, edge_rot_mat_inv)
    
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()