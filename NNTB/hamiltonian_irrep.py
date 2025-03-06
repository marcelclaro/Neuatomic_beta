import numpy as np
import torch
from e3nn.o3 import Irreps, Irrep, wigner_3j, FullTensorProduct
from .util import orbitals_from_str_yzx

# Begining of code from E3NN
##
##
"""
Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy), 
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
and Kostiantyn Lapchevskyi. All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from e3nn.util import explicit_default_types
def change_basis_real_to_complex(l: int, dtype=None, device=None, makereal = True) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = 1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] =  (-1) ** m * 1  / 2**0.5 
        q[l + m, l - abs(m)] = (-1) ** m * -1j / 2**0.5 
    if makereal:
        q = (-1j) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    dtype, device = explicit_default_types(dtype, device)
    dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]
    # make sure we always get:
    # 1. a copy so mutation doesn't ruin the stored tensors
    # 2. a contiguous tensor, regardless of what transpositions happened above
    return q.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)

def change_basis_complex_to_real(l: int, dtype=None, device=None, makereal = True) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] =  -1j / 2**0.5
        q[l + m, l - abs(m)] = (-1) ** m * 1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1): 
        q[l + m, l + abs(m)] =  (-1) ** m * 1  / 2**0.5 #(-1) ** m
        q[l + m, l - abs(m)] =  1  / 2**0.5 #* (-1) ** m
    if makereal:
        q = (-1j) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    dtype, device = explicit_default_types(dtype, device)
    dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]
    # make sure we always get:
    # 1. a copy so mutation doesn't ruin the stored tensors
    # 2. a contiguous tensor, regardless of what transpositions happened above
    return q.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)

def su2_clebsch_gordan_coeff(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient
    for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    from fractions import Fraction
    from math import factorial

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return factorial(round(n))

    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5

    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1) ** int(v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v), f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3)
        )
    C = C * S
    return C

def su2_clebsch_gordan(j1, j2, j3):
    """Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    Returns
    -------
    cg_matrix : numpy.array
        Requested Clebsch-Gordan matrix.
    """
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = torch.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)), dtype=torch.complex64)
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = su2_clebsch_gordan_coeff(
                        (j1, m1), (j2, m2), (j3, m1 + m2)
                    )
    
    return mat

# End of code from E3NN
##
##


#Here starts the original code!

"""
Makes the transformation of a hamiltonian to an irreducible basis so th e3nn can operate on
"""
class HamiltonianIrrepTransformer():
    """
    A class to transform Hamiltonians into irreducible representations (irreps) and vice versa.
    Attributes:
    -----------
    basechange : bool
        Specifies the change of basis yzx -> xyz.
    slices : list
        Keeps track of Hamiltonian l slices.
    blocks : list
        Keeps track of Hamiltonian blocks.
    irep_prod : list
        Keeps track of irreducible representation products, corresponding to Hamiltonian blocks.
    ham_irrep : Irreps
        The total Hamiltonian representation, corresponding to irreducible representation vector returned.
    base_irrep : Irreps
        The irreducible representation of atomic base (in orbitals).
    soc_irrep : Irreps or None
        The total SOC Hamiltonian representation (complex scalars).
    mat_n : int
        Dimension of Hamiltonian (n,n). If with spin, dimension is the dimension of the block in H(4x4).
    spin : bool
        Indicates if spin is considered.
    change_of_coord : torch.Tensor
        Change of basis matrix yzx -> xyz.
    Dbase : torch.Tensor
        Transformation matrix for the base irreducible representation.
    irreducible_dim : int
        Dimension for the irreducible vector.
    Methods:
    --------
    to_irrep_Hamiltonian(H, S=None, selfenergy=None):
        Transforms a Hamiltonian matrix into its irreducible representation.
    to_irrep_Overlap(H):
        Transforms an overlap matrix into its irreducible representation.
    from_irrep_Hamiltonian(irred_H, irred_H_soc=None, S=None, selfenergy=None):
        Transforms an irreducible representation back into a Hamiltonian matrix.
    from_irrep_Overlap(irred_S):
        Transforms an irreducible representation back into an overlap matrix.
    """
    def __init__(self,orbitals_str,basechange = True, spin = False) -> None:
        self.basechange = basechange # this specifies the change of basis yzx -> xyz
        self.slices = []  #keep track of hamiltonian l slices
        self.blocks = []  #keep track of hamiltonian blocks
        self.irep_prod = [] #keep track of irred representation products, correspond to hamiltonian blocks
        self.ham_irrep = Irreps() #The total hamiltonian representation, correspond to irreducible representation vector returned
        self.base_irrep = Irreps() #The irreducible representation of atomic base (in orbitals)
        self.soc_irrep = None #The total SOC hamiltonian representation (complex scalars)
        orbitals = orbitals_from_str_yzx(orbitals_str,spin=False)  #Get orbital (n,l,m) list from the string
        self.mat_n = len(orbitals)  #n dimension of Hamiltonian (n,n), if with spin, dimension is the dimension of the block in H(4x4)
        self.spin=spin 
        
        #Go over orbitals creating irreps and slices
        for _,l,m in orbitals:
            if m == 0: #once per l
                self.base_irrep += Irrep((l,(-1)**(l)))
                self.slices.append(l)

        #Go over slices to create blocks
        for a in self.slices:
            for b in self.slices:
                self.blocks.append((a,b))

        #There is a SOC parameter for each block
        self.soc_irrep = Irreps(f"{len(self.blocks)}x0e")
                
        for _,ir_a in self.base_irrep:
            for _,ir_b in self.base_irrep:
                prod_irr = FullTensorProduct(ir_a,ir_b).irreps_out
                self.ham_irrep += prod_irr
                self.irep_prod.append(prod_irr)

        #change of basis yzx -> xyz matrix
        self.change_of_coord = torch.tensor([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
        ],dtype=torch.float32,requires_grad=False)
        self.Dbase = self.base_irrep.D_from_matrix(self.change_of_coord)
        
        #Dimension for the irred. vector
        self.irreducible_dim = self.ham_irrep.dim 

    #If spin, S or selfenergy should be provided
    def to_irrep_Hamiltonian(self,H,selfenergy=None,S=None):
        """
        Transforms a given Hamiltonian into its irreducible representation form.
        Parameters:
        -----------
        H : torch.Tensor
            The Hamiltonian tensor. It can be a single Hamiltonian (rank 2 tensor) or a batch of Hamiltonians (rank 3 tensor).
        S : torch.Tensor, optional
            The overlap matrix. Required if spin-orbit coupling (SOC) is used. Default is None.
        selfenergy : list or torch.Tensor, optional
            Self-energy values. Used to create the overlap matrix if S is not provided. Default is None.
        Returns:
        --------
        irred_H : torch.Tensor
            The Hamiltonian in its irreducible representation form.
        irred_H_soc : torch.Tensor, optional
            The SOC coefficients in the irreducible representation form. Returned only if spin-orbit coupling is used.
        Raises:
        -------
        AssertionError
            If the Hamiltonian dimensions do not match the expected dimensions.
            If the imaginary part of the non-SOC Hamiltonian is too large.
            If the overlap matrix or self-energy is not provided when using SOC.
            If the Hamiltonian is not square.
            If the Hamiltonian shape is not valid for a single or batch Hamiltonian.
        """

        Hsize = H.size()
        Hrank = len(Hsize)

        device = H.device

        if self.spin:
            assert (S!=None or selfenergy!=None), "If using SOC overlap (S) or selfenergy is required"
       
        if Hrank == 2:   #Single Hamiltonian, it is not a batch       
        
            
            if self.spin:

                # Split the tensor into 4 blocks
                # Each block will have the shape [batch, i/2, i/2]
                block_size = self.mat_n

                # Top-left block
                block_00 = H[ :block_size, :block_size]
                # Bottom-right block
                block_11 = H[ block_size:, block_size:]

                # Concatenate blocks along a new dimension
                H_nosoc = (block_00+block_11) / 2.0
                
                assert torch.all(torch.abs(torch.imag(H_nosoc)) < 1e-5), f"Imaginary part of non-SOC hamiltonian is too big ({torch.max(torch.abs(torch.imag(H_nosoc)))})"

                newH = torch.real(H_nosoc).to(torch.float32)
                
                #Pertubative term for SOC: Hsoc
                Hsoc = H.detach().clone()
                Hsoc[ :block_size, :block_size] -= H_nosoc
                Hsoc[ block_size:, block_size:] -= H_nosoc
                
            else:
                newH = H  
            
            assert newH.size(0) == self.mat_n, "Hamiltonian dimension does not match"
            assert newH.size(0)== newH.size(1), "Hamiltonian to transform is not square"
        
            #initialize output vectors
            irred_H = torch.zeros((self.irreducible_dim),device=device)
            if self.spin:
                irred_H_soc = torch.zeros((len(self.blocks)),dtype=torch.complex64,device=device)

                if S==None: #Case overlap was not given, one is created. Identity only for on-site atoms...
                    assert isinstance(selfenergy, bool), "Selfenergy should be a list or tensor!"
                    S=torch.eye(2*self.mat_n,dtype=torch.complex64,device=device) if selfenergy else torch.zeros((2*self.mat_n,2*self.mat_n),dtype=torch.complex64,device=device)

            #if change of base yzx <-> xyz:
            if self.basechange :
                Dbase = self.Dbase.to(device=device)
                currentH = torch.einsum('ij,jk,kl->il', Dbase, newH, Dbase.t())
            else:
                currentH = newH.to(copy=True, memory_format=torch.contiguous_format)

            #initialize index to keep track of positions
            col_index = 0
            row_index = 0
            irrep_index = 0

            
            #for each block...
            for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                matshape = (2*bl[0]+1,2*bl[1]+1) #Block shape
                
                matslice = currentH[col_index:col_index+matshape[0],row_index:row_index+matshape[1]] #Corresponding Hamiltonian block (could be reducible)
                
                if self.spin:
                    indices_col = torch.cat([torch.arange(col_index,col_index+matshape[0]), torch.arange(block_size+col_index,block_size+col_index+matshape[0])])
                    indices_row = torch.cat([torch.arange(row_index,row_index+matshape[1]), torch.arange(block_size+row_index,block_size+row_index+matshape[1])])
                    matslicesoc = Hsoc[indices_col,:][:,indices_row]
                    
                    
                    l_right=bl[1]
                    jlist=[]
                    eigvalues_right = []
                    for j in np.arange(0.5,l_right+1.5,1.0):
                        jlist.append(j)
                        eigvalues_right.append(j*(j+1)-l_right*(l_right+1)-0.5*(0.5+1)*np.ones(int(2*j+1)))
                    transformation_right = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_right,0.5,j),change_basis_real_to_complex(l_right,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_right+1),int(2*j+1)) for j in jlist],dim=1)
                    transformation_right = transformation_right.to(device=device)

                    l_left=bl[0]
                    jlist=[]
                    for j in np.arange(0.5,l_left+1.5,1.0):
                        jlist.append(j)
                    transformation_left = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_left,0.5,j),change_basis_real_to_complex(l_left ,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_left +1),int(2*j+1)) for j in jlist],dim=1)
                    transformation_left = transformation_left.to(device=device)

                    #get the overlap matrix corresponding to current block
                    overlapslice = S[indices_col,:][:,indices_row].clone()
                    overlapslice = torch.einsum('ij,jk,kl->il', transformation_left.conj().t(), overlapslice, transformation_right)

                    #L*s|psi> in the |J,mj> has the eigen value J**2 - L**2 - s**2
                    #Hsoc transformed back to the abacus base 
                    soc_base = torch.tensor(np.concatenate(eigvalues_right),dtype=torch.complex64,device=device)*overlapslice
                    soc_base = torch.einsum('ij,jk,kl->il', transformation_left, soc_base, transformation_right.conj().t())
                    
                    #SOC coefficient is then estimated
                    mod = torch.sum(soc_base * soc_base.conj())
                    if mod != 0:
                        c = (torch.sum(soc_base * matslicesoc.conj()) / mod).item()
                    else:
                        c = 0.0 + 0.0j
                    irred_H_soc[n] = c
                
                
                for ir in irs:
                    #For each each irreducible representations transform using clebsch-gordon coef. for spin 0
                    irred_H[irrep_index:(irrep_index+ir.dim)]=torch.einsum('ijk,ij->k', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device), matslice )
                    irrep_index += ir.dim
                
                #Col,row strider            
                row_index += matshape[1]
                if row_index >= self.mat_n:
                    row_index = 0
                    col_index += matshape[0]
                    
        #batch case
        elif Hrank == 3:
            batch_size = Hsize[0]

            if self.spin:

                # Split the tensor into 4 blocks
                # Each block will have the shape [batch, i/2, i/2]
                block_size = self.mat_n
                
                # Top-left block
                block_00 = H[:, :block_size, :block_size]
                # Bottom-right block
                block_11 = H[:, block_size:, block_size:]

                #Get H0 (non-SOC) hamiltonian from up-up, down-down blocks
                H_nosoc = (block_00+block_11) / 2.0
                
                assert torch.all(torch.abs(torch.imag(H_nosoc)) < 1e-5), f"Imaginary part of non-SOC hamiltonian is too big ({torch.max(torch.abs(torch.imag(H_nosoc)))})"

                newH = torch.real(H_nosoc).to(torch.float32)
                
                #Pertubative term for SOC: Hsoc
                Hsoc = H.detach().clone()
                Hsoc[:, :block_size, :block_size] -= H_nosoc
                Hsoc[:, block_size:, block_size:] -= H_nosoc

                if S==None: #Case overlap was not given, one is created. Identity only for on-site atoms...
                    assert isinstance(selfenergy, (list, torch.Tensor)), "Selfenergy should be a list or tensor!"
                    S=torch.stack([torch.eye(2*self.mat_n,dtype=torch.complex64,device=device) if value else torch.zeros((2*self.mat_n,2*self.mat_n),dtype=torch.complex64,device=device) for value in selfenergy])

            else:
                newH = H          
         
            assert newH.size(1) == self.mat_n, f"Hamiltonian dimension {H.shape} (spin={self.spin}) does not match ({self.mat_n},{self.mat_n})"
            assert newH.size(1) == newH.size(2), f"Hamiltonian {H.shape} to be transformed is not square"
            
            #initialize output vectors
            irred_H = torch.zeros((batch_size,self.irreducible_dim),device=device)
            if self.spin:
                irred_H_soc = torch.zeros((batch_size,len(self.blocks)),dtype=torch.complex64,device=device)

            #if change of base yzx -> xyz:
            if self.basechange :
                Dbase = self.Dbase.to(device=device)
                newH_b = torch.einsum('ij,bjk,kl->bil', Dbase, newH, Dbase.t())
            else:
                newH_b = newH

            for i in range(batch_size):
                #initialize index to keep track of positions
                col_index = 0
                row_index = 0
                irrep_index = 0
                             
                #for each block...
                currentH = newH_b[i]
                for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                    matshape = (2*bl[0]+1,2*bl[1]+1) #Block shape
                    matslice = currentH[col_index:col_index+matshape[0],row_index:row_index+matshape[1]] #Corresponding Hamiltonian block (could be reducible)
                   
                    
                    if self.spin:
                        #Indices for each block is created (4 spin blocks...)
                        indices_col = torch.cat([torch.arange(col_index,col_index+matshape[0]), torch.arange(block_size+col_index,block_size+col_index+matshape[0])])
                        indices_row = torch.cat([torch.arange(row_index,row_index+matshape[1]), torch.arange(block_size+row_index,block_size+row_index+matshape[1])])
                        matslicesoc = Hsoc[i,indices_col,:][:,indices_row] #Corresponding Hsoc block
                        
                        #Base transformation to |J,mj> space, Abacus uses real sperical harmonics, a change of base is required
                        #Left and right transformation matrices are calculated
                        l_right=bl[1]
                        jlist=[]
                        eigvalues_right = []
                        for j in np.arange(0.5,l_right+1.5,1.0):
                            jlist.append(j)
                            eigvalues_right.append(j*(j+1)-l_right*(l_right+1)-0.5*(0.5+1)*np.ones(int(2*j+1)))
                        transformation_right = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_right,0.5,j),change_basis_real_to_complex(l_right,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_right+1),int(2*j+1)) for j in jlist],dim=1)
                        transformation_right = transformation_right.to(device=device)

                        l_left=bl[0]
                        jlist=[]
                        for j in np.arange(0.5,l_left+1.5,1.0):
                            jlist.append(j)
                        transformation_left = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_left,0.5,j),change_basis_real_to_complex(l_left ,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_left +1),int(2*j+1)) for j in jlist],dim=1)
                        transformation_left = transformation_left.to(device=device)

                        #get the overlap matrix corresponding to the current block
                        overlapslice = S[i,indices_col,:][:,indices_row].clone()
                        #Base transformation to |J,mj> space 
                        overlapslice = torch.einsum('ij,jk,kl->il', transformation_left.conj().t(), overlapslice, transformation_right)
                        
                        #L*s|psi> in the |J,mj> has the eigen value J**2 - L**2 - s**2
                        #Hsoc transformed back to the abacus base 
                        soc_base = torch.tensor(np.concatenate(eigvalues_right),dtype=torch.complex64,device=device)*overlapslice
                        soc_base = torch.einsum('ij,jk,kl->il', transformation_left, soc_base, transformation_right.conj().t())
                        
                        #SOC coefficient is then estimated
                        mod = torch.sum(soc_base * soc_base.conj())
                        if mod != 0: #and selfenergy[i]
                            c = (torch.sum(matslicesoc * soc_base.conj()) / mod).item()
                        else:
                            c = 0.0 + 0.0j
                        irred_H_soc[i,n] = c


                    for ir in irs:
                        #For each each irreducible representations transform using clebsch-gordon coef.
                        irred_H[i,irrep_index:(irrep_index+ir.dim)]=torch.einsum('ijk,ij->k', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device), matslice )
                        irrep_index += ir.dim

                    #Col,row strider            
                    row_index += matshape[1]
                    if row_index >= self.mat_n:
                        row_index = 0
                        col_index += matshape[0]
        else:
            assert False, f"H has the shape {H.shape} and cannot be a batch or single hamiltonian"
                  
        if self.spin:
            return irred_H, torch.real(irred_H_soc) #The radial integral should be real constant (ps.: Tested on PbO case)
        else:
            return irred_H

    #Overlap is block diagonal with equal blocks
    def to_irrep_Overlap(self,S):
        """
        Transforms a Hamiltonian matrix or batch of Hamiltonian matrices into their irreducible representations.
        Parameters:
        -----------
        H : torch.Tensor
            The Hamiltonian matrix or batch of Hamiltonian matrices to be transformed. 
            If a single matrix, it should have shape (n, n). 
            If a batch, it should have shape (batch_size, n, n).
        Returns:
        --------
        irred_S : torch.Tensor
            The transformed Hamiltonian in its irreducible representation. 
            If a single matrix, it will have shape (irreducible_dim,). 
            If a batch, it will have shape (batch_size, irreducible_dim).
        Raises:
        -------
        AssertionError
            If the input Hamiltonian does not have the correct shape or if the imaginary part of the overlap is too large.
        Notes:
        ------
        - If `self.spin` is True, the Hamiltonian is split into blocks and only the top-left block is used.
        - If `self.basechange` is True, a change of basis is applied using `self.Dbase`.
        - The transformation uses Clebsch-Gordon coefficients for spin 0.
        """

        Hsize = S.size()
        Hrank = len(Hsize)

        device = S.device
      
        if Hrank == 2:          #Not a batch
                    
            if self.spin:

                # Split the tensor into 4 blocks
                # Each block will have the shape [batch, i/2, i/2]
                block_size = self.mat_n

                # Top-left block
                block_00 = S[ :block_size, :block_size]

                # Concatenate blocks along a new dimension
                H_nosoc = block_00
                
                assert torch.all(torch.abs(torch.imag(H_nosoc)) < 1e-5), f"Imaginary part of overlap is too big ({torch.max(torch.abs(torch.imag(H_nosoc)))})"
                newH = torch.real(H_nosoc).to(torch.float32)
                
                
            else:
                newH = S  
            
            assert newH.size(0) == self.mat_n, "Overlap dimension does not match"
            assert newH.size(0)== newH.size(1), "Overlap to transform is not square"
        
            #initialize output vectors
            irred_S = torch.zeros((self.irreducible_dim),device=device)

            #if change of base yzx <-> xyz:
            if self.basechange :
                currentH = torch.einsum('ij,jk,kl->il', self.Dbase, newH, self.Dbase.t())
            else:
                currentH = newH.to(copy=True, memory_format=torch.contiguous_format)

            #initialize index to keep track of positions
            col_index = 0
            row_index = 0
            irrep_index = 0

            
            #for each block...
            for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                matshape = (2*bl[0]+1,2*bl[1]+1) #Block shape
                
                matslice = currentH[col_index:col_index+matshape[0],row_index:row_index+matshape[1]] #Corresponding Hamiltonian block (could be reducible)
                
                for ir in irs:
                    #For each each irreducible representations transform using clebsch-gordon coef. for spin 0
                    irred_S[irrep_index:(irrep_index+ir.dim)]=torch.einsum('ijk,ij->k', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device), matslice )
                    irrep_index += ir.dim
                
                #Col,row strider            
                row_index += matshape[1]
                if row_index >= self.mat_n:
                    row_index = 0
                    col_index += matshape[0]
                    
        #batch case
        elif Hrank == 3:
            batch_size = Hsize[0]

            if self.spin:

                # Split the tensor into 4 blocks
                # Each block will have the shape [batch, i/2, i/2]
                block_size = self.mat_n

                # Top-left block
                block_00 = S[:, :block_size, :block_size]

                #Get H0 (non-SOC) hamiltonian from up-up, down-down blocks
                H_nosoc = block_00
                
                assert torch.all(torch.abs(torch.imag(H_nosoc)) < 1e-5), f"Imaginary part of overlap is too big ({torch.max(torch.abs(torch.imag(H_nosoc)))})"
                newH = torch.real(H_nosoc).to(torch.float32)
                

            else:
                newH = S          
         
            assert newH.size(1) == self.mat_n, f"Hamiltonian dimension {S.shape} does not match ({self.mat_n},{self.mat_n})"
            assert newH.size(1) == newH.size(2), f"Hamiltonian {S.shape} to be transformed is not square"
            
            #initialize output vectors
            irred_S = torch.zeros((batch_size,self.irreducible_dim),device=device)

            #if change of base yzx -> xyz:
            if self.basechange :
                newH_b = torch.einsum('ij,bjk,kl->bil', self.Dbase, newH, self.Dbase.t())
            else:
                newH_b = newH

            for i in range(batch_size):
                #initialize index to keep track of positions
                col_index = 0
                row_index = 0
                irrep_index = 0
                             
                #for each block...
                currentH = newH_b[i]
                for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                    matshape = (2*bl[0]+1,2*bl[1]+1) #Block shape
                    matslice = currentH[col_index:col_index+matshape[0],row_index:row_index+matshape[1]] #Corresponding Hamiltonian block (could be reducible)
                   
                    for ir in irs:
                        #For each each irreducible representations transform using clebsch-gordon coef.
                        irred_S[i,irrep_index:(irrep_index+ir.dim)]=torch.einsum('ijk,ij->k', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device), matslice )
                        irrep_index += ir.dim

                    #Col,row strider            
                    row_index += matshape[1]
                    if row_index >= self.mat_n:
                        row_index = 0
                        col_index += matshape[0]
        else:
            assert False, f"H has the shape {S.shape} and cannot be a batch or single hamiltonian"
                  
        return irred_S

    def fullsoc_to_nodeonly(self,irred_H_soc,graph):
        """
        Transforms a full spin-orbit coupling Hamiltonian to a node-only representation.

        Parameters:
        irred_H_soc (numpy.ndarray): The full spin-orbit coupling Hamiltonian.
        graph (Graph): The graph object containing the node indices.

        Returns:
        numpy.ndarray: The node-only representation of the Hamiltonian.
        """

        return irred_H_soc[graph.self_idx,:]
    
    def nodeonlysoc_to_full(self,irred_H_soc_node,graph):
        """
        Converts a node-only spin-orbit coupling (SOC) Hamiltonian to a full Hamiltonian.

        Args:
            irred_H_soc_node (torch.Tensor): The irreducible SOC Hamiltonian for nodes.
            graph (Graph): The graph object containing the hopping information and self indices.

        Returns:
            torch.Tensor: The full irreducible SOC Hamiltonian.
        """

        irred_H_soc = torch.zeros((graph.edge_index.size(1),irred_H_soc_node.size(1)),dtype=torch.float32,device=irred_H_soc_node.device)
        irred_H_soc[graph.self_idx,:] = irred_H_soc_node
        return irred_H_soc

    #If spin, S or selfenergy should be provided            
    def from_irrep_Hamiltonian(self, irred_H, irred_H_soc=None, selfenergy=None, S=None):
        """
        Constructs the Hamiltonian matrix from its irreducible representation.
        Parameters:
        -----------
        irred_H : torch.Tensor
            The irreducible Hamiltonian tensor.
        irred_H_soc : torch.Tensor, optional
            The spin-orbit coupling irreducible Hamiltonian tensor. Required if `self.spin` is True.
        S : torch.Tensor, optional
            The overlap matrix. Required if `self.spin` is True and `selfenergy` is not provided.
        selfenergy : bool or list of bool, optional
            If True, the overlap matrix is the identity matrix. If False, the overlap matrix is zero.
            If a list, it specifies the overlap matrix for each batch element.
        Returns:
        --------
        torch.Tensor
            The constructed Hamiltonian matrix.
        Raises:
        -------
        AssertionError
            If the dimensions of the input tensors do not match the expected dimensions.
            If `self.spin` is True and `irred_H_soc` or `S`/`selfenergy` is not provided.
            If `selfenergy` is not a boolean when `Hrank` is 1.
            If the shape of `H` is not compatible with a batch or single Hamiltonian irreducible vector.
        """
        Hrank = len(irred_H.size())

        device = irred_H.device

        if self.spin:
            assert irred_H_soc!=None, "If using SOC irred_H_soc is required"
            assert irred_H.device and irred_H_soc.device, f"irred_H is on {irred_H.device} but irred_H_soc is on {irred_H_soc.device}"
            assert (S!=None or selfenergy!=None), "If using SOC, overlap (S) or selfenergy is required"
              
        if Hrank == 1:  #Not a batch
            
            assert irred_H.size(0) == self.irreducible_dim, "Hamiltonian dimension does not match" 
            #initialize Ham matrix
            if self.spin:
                H_soc = torch.zeros((self.mat_n*2,self.mat_n*2),dtype=torch.complex64,device=device)
                if S == None: #Overlap Identity matrix if it is not given:
                    assert isinstance(selfenergy, bool), "If the Hamiltonian is not a batch selfenergy is single valued (True or False)"
                    if selfenergy:
                        S=torch.eye((self.mat_n*2,self.mat_n*2),dtype=torch.complex64,device=device)
                    else:
                        S=torch.zeros((self.mat_n*2,self.mat_n*2),dtype=torch.complex64,device=device)
            H = torch.zeros((self.mat_n,self.mat_n),dtype=torch.float32,device=device)

            col_index = 0
            row_index = 0
            irrep_index = 0

            block_size = self.mat_n

             #for each block...
            for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                matshape = (2*bl[0]+1,2*bl[1]+1) #Block shape
                for idx, ir in enumerate(irs):
                    #Corresponding Hamiltonian block (could be reducible)
                    #For each each irreducible representations transform using clebsch-gordon coef.
                    H[col_index:col_index+matshape[0],row_index:row_index+matshape[1]]+= torch.einsum('ijk,k->ij', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device),irred_H[irrep_index:(irrep_index+ir.dim)])
                    irrep_index += ir.dim

                    if self.spin:
                        #Indices for each block is created (4 spin blocks...)
                        indices_col = torch.cat([torch.arange(col_index,col_index+matshape[0]), torch.arange(block_size+col_index,block_size+col_index+matshape[0])])
                        indices_row = torch.cat([torch.arange(row_index,row_index+matshape[1]), torch.arange(block_size+row_index,block_size+row_index+matshape[1])])
                        
                        
                        #Base transformation to |J,mj> space, Abacus uses real sperical harmonics, a change of base is required
                        #Left and right transformation matrices are calculated
                        l_right=bl[1]
                        jlist=[]
                        eigvalues_right = []
                        for j in np.arange(0.5,l_right+1.5,1.0):
                            jlist.append(j)
                            eigvalues_right.append(j*(j+1)-l_right*(l_right+1)-0.5*(0.5+1)*np.ones(int(2*j+1)))
                        transformation_right = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_right,0.5,j),change_basis_real_to_complex(l_right,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_right+1),int(2*j+1)) for j in jlist],dim=1)
                        transformation_right = transformation_right.to(device=device)


                        l_left=bl[0]
                        jlist=[]
                        for j in np.arange(0.5,l_left+1.5,1.0):
                            jlist.append(j)
                        transformation_left = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_left,0.5,j),change_basis_real_to_complex(l_left ,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_left +1),int(2*j+1)) for j in jlist],dim=1)
                        transformation_left = transformation_left.to(device=device)

                        #get the overlap matrix corresponding to the current block
                        overlapslice = S[indices_col,:][:,indices_row].clone()
                        #Base transformation to |J,mj> space 
                        overlapslice = torch.einsum('ij,jk,kl->il', transformation_left.conj().t(), overlapslice, transformation_right)
                        
                        #L*s|psi> in the |J,mj> has the eigen value J**2 - L**2 - s**2
                        #Hsoc transformed back to the abacus base 
                        soc_base = torch.tensor(np.concatenate(eigvalues_right),dtype=torch.complex64,device=device)*overlapslice
                        soc_base = torch.einsum('ij,jk,kl->il', transformation_left, soc_base, transformation_right.conj().t())
                                                
                        #Hsoc based on coefficient and mmatrix
                        soc_base = irred_H_soc[n]*soc_base
                        H_soc[indices_col[:, None], indices_row] = soc_base
                
                #Col,row strider
                row_index += matshape[1]
                if row_index >= self.mat_n:
                    row_index = 0
                    col_index += matshape[0]
            
            #if change of base yzx <-> xyz:        
            if self.basechange :
                Dbase = self.Dbase.to(device=device)
                H = torch.einsum('ij,jk,kl->il', Dbase.t(), H, Dbase)

            if self.spin:
                #Final matrix construction
                #Add H0 (non-soc) values to Hsoc
                H_soc[ :self.mat_n, :self.mat_n] += H
                H_soc[ self.mat_n:, self.mat_n:] += H
                H = H_soc

        #batch case
        elif Hrank == 2:
            
            batch_size = irred_H.size(0)
            
            assert irred_H.size(1) == self.irreducible_dim, "Hamiltonian dimension does not match" 
            
            #initialize Ham matrix
            if self.spin:
                H_soc = torch.zeros((batch_size,self.mat_n*2,self.mat_n*2),dtype=torch.complex64,device=device)
                if S==None:
                    S=torch.stack([torch.eye(2*self.mat_n,dtype=torch.complex64,device=device) if value else torch.zeros((2*self.mat_n,2*self.mat_n),dtype=torch.complex64,device=device) for value in selfenergy])
            
            H = torch.zeros((batch_size,self.mat_n,self.mat_n),device=device)

            block_size = self.mat_n
             
            for i in range(batch_size): 
                col_index = 0
                row_index = 0
                irrep_index = 0

                #for each block...
                for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                    matshape = (2*bl[0]+1,2*bl[1]+1)
                    for ir in irs:
                        #Corresponding Hamiltonian block (could be reducible)
                        #For each each irreducible representations transform using clebsch-gordon coef.
                        H[i,col_index:col_index+matshape[0],row_index:row_index+matshape[1]]+= torch.einsum('ijk,k->ij', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device),irred_H[i,irrep_index:(irrep_index+ir.dim)])
                        irrep_index += ir.dim

                    if self.spin:
                        #Indices for each block is created (4 spin blocks...)
                        indices_col = torch.cat([torch.arange(col_index,col_index+matshape[0]), torch.arange(block_size+col_index,block_size+col_index+matshape[0])])
                        indices_row = torch.cat([torch.arange(row_index,row_index+matshape[1]), torch.arange(block_size+row_index,block_size+row_index+matshape[1])])
                        

                        #Base transformation to |J,mj> space, Abacus uses real sperical harmonics, a change of base is required
                        #Left and right transformation matrices are calculated
                        l_right=bl[1]
                        jlist=[]
                        eigvalues_right = []
                        for j in np.arange(0.5,l_right+1.5,1.0):
                            jlist.append(j)
                            eigvalues_right.append(j*(j+1)-l_right*(l_right+1)-0.5*(0.5+1)*np.ones(int(2*j+1)))
                        transformation_right = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_right,0.5,j),change_basis_real_to_complex(l_right,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_right+1),int(2*j+1)) for j in jlist],dim=1)
                        transformation_right = transformation_right.to(device=device)

                        l_left=bl[0]
                        jlist=[]
                        for j in np.arange(0.5,l_left+1.5,1.0):
                            jlist.append(j)
                        transformation_left = torch.cat([torch.einsum('ilm,ij->jlm',su2_clebsch_gordan(l_left,0.5,j),change_basis_real_to_complex(l_left ,makereal=False)).permute(1, 0, 2).reshape(2*(2*l_left +1),int(2*j+1)) for j in jlist],dim=1)
                        transformation_left = transformation_left.to(device=device)

                        #get the overlap matrix corresponding to the current block
                        overlapslice = S[i,indices_col,:][:,indices_row].clone()
                        #Base transformation to |J,mj> space 
                        overlapslice = torch.einsum('ij,jk,kl->il', transformation_left.conj().t(), overlapslice, transformation_right)
                        
                        #L*s|psi> in the |J,mj> has the eigen value J**2 - L**2 - s**2
                        #Hsoc transformed back to the abacus base 
                        soc_base = torch.tensor(np.concatenate(eigvalues_right),dtype=torch.complex64,device=device)*overlapslice
                        soc_base = torch.einsum('ij,jk,kl->il', transformation_left, soc_base, transformation_right.conj().t())
                        soc_base = irred_H_soc[i,n]*soc_base

                        #Add SOC contribution
                        if selfenergy != None:
                            if selfenergy[i]:  #only on-site contribution
                                H_soc[i, indices_col[:, None], indices_row] += soc_base
                        else: #Overlap aproximation
                            H_soc[i, indices_col[:, None], indices_row] += soc_base
                    
                    #Col,row strider
                    row_index += matshape[1]
                    if row_index >= self.mat_n:
                        row_index = 0
                        col_index += matshape[0]
                
                
                #if change of base yzx -> xyz:        
                if self.basechange :
                    Dbase = self.Dbase.to(device=device)
                    H[i] = torch.einsum('ij,jk,kl->il', Dbase.t(), H[i], Dbase)
                        
                if self.spin:
                    #Add H0 (non-soc) values
                    H_soc[i, :self.mat_n, :self.mat_n] += H[i]
                    H_soc[i, self.mat_n:, self.mat_n:] += H[i]


            if self.spin:
                H = H_soc
        else:
            assert False, f"irred_H has the shape {irred_H.shape} and cannot be a batch or single hamiltonian irreducible vector"
   
        return H

    #Overlap is block diagonal with equal blocks
    def from_irrep_Overlap(self, irred_S):
        """
        Constructs the Overlap matrix from its irreducible representation.
        Parameters:
        -----------
        irred_S : torch.Tensor
            The irreducible representation of the Overlap. It can be either a 
            single matrix (2D tensor) or a batch of matrices (3D tensor).
        Returns:
        --------
        torch.Tensor
            The constructed Overlap matrix. If `spin` is True, the matrix will 
            include spin-orbit coupling terms and will be of complex type. Otherwise, 
            it will be a real-valued matrix.
        Raises:
        -------
        AssertionError
            If the dimension of `irred_S` does not match the expected irreducible dimension.
        """
        Hrank = len(irred_S.size())

        device = irred_S.device

              
        if Hrank == 1:  #Not a batch
            
            assert irred_S.size(0) == self.irreducible_dim, "Hamiltonian dimension does not match" 
            #initialize Ham matrix
            if self.spin:
                H_soc = torch.zeros((self.mat_n*2,self.mat_n*2),dtype=torch.complex64,device=device)

            H = torch.zeros((self.mat_n,self.mat_n),dtype=torch.float32,device=device)

            col_index = 0
            row_index = 0
            irrep_index = 0

             #for each block...
            for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                matshape = (2*bl[0]+1,2*bl[1]+1) #Block shape
                for idx, ir in enumerate(irs):
                    #Corresponding Hamiltonian block (could be reducible)
                    #For each each irreducible representations transform using clebsch-gordon coef.
                    H[col_index:col_index+matshape[0],row_index:row_index+matshape[1]]+= torch.einsum('ijk,k->ij', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device),irred_S[irrep_index:(irrep_index+ir.dim)])
                    irrep_index += ir.dim

                #Col,row strider
                row_index += matshape[1]
                if row_index >= self.mat_n:
                    row_index = 0
                    col_index += matshape[0]
            
            #if change of base yzx <-> xyz:        
            if self.basechange :
                Dbase = self.Dbase.to(device=device)
                H = torch.einsum('ij,jk,kl->il', Dbase.t(), H, Dbase)

            if self.spin:
                #Final matrix construction
                #Add H0 (non-soc) values to Hsoc
                H_soc[ :self.mat_n, :self.mat_n] += H
                H_soc[ self.mat_n:, self.mat_n:] += H
                H = H_soc

    
        elif Hrank == 2: #Is a batch
            
            batch_size = irred_S.size(0)
            
            assert irred_S.size(1) == self.irreducible_dim, "Hamiltonian dimension does not match" 
            
            #initialize Ham matrix
            if self.spin:
                H_soc = torch.zeros((batch_size,self.mat_n*2,self.mat_n*2),dtype=torch.complex64,device=device)
  
            H = torch.zeros((batch_size,self.mat_n,self.mat_n),device=device)
             
            for i in range(batch_size): 
                col_index = 0
                row_index = 0
                irrep_index = 0

                #for each block...
                for n, (bl, irs) in enumerate(zip(self.blocks,self.irep_prod)):
                    matshape = (2*bl[0]+1,2*bl[1]+1)
                    for ir in irs:
                        #Corresponding Hamiltonian block (could be reducible)
                        #For each each irreducible representations transform using clebsch-gordon coef.
                        H[i,col_index:col_index+matshape[0],row_index:row_index+matshape[1]]+= torch.einsum('ijk,k->ij', np.sqrt(2*ir[1].l+1) * wigner_3j(bl[0], bl[1], ir[1].l,device=device),irred_S[i,irrep_index:(irrep_index+ir.dim)])
                        irrep_index += ir.dim

                    #Col,row strider
                    row_index += matshape[1]
                    if row_index >= self.mat_n:
                        row_index = 0
                        col_index += matshape[0]
                
                
                #if change of base yzx -> xyz:        
                if self.basechange :
                    Dbase = self.Dbase.to(device=device)
                    H[i] = torch.einsum('ij,jk,kl->il', Dbase.t(), H[i], Dbase)
                        
                if self.spin:
                    #Add H0 (non-soc) values
                    H_soc[i, :self.mat_n, :self.mat_n] += H[i]
                    H_soc[i, self.mat_n:, self.mat_n:] += H[i]


            if self.spin:
                H = H_soc
        else:
            assert False, f"H has the shape {H.shape} and cannot be a batch or single hamiltonian irreducible vector"
   
        return H
    
