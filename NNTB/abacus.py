import numpy as np
from .util import *
from scipy.sparse import csr_matrix

"""
Copyright (c) 2024 Marcel S. Claro

Based on PYATB, but fixes issues when matrix is not triangular!
"""

"""
PYATB is licensed under GPLv3.0, making it freely available for use and modification by the scientific community. The
main developers of PYATB are Gan Jin, Hongsheng Pang, Yuyang Ji, Zujian Dai, under the supervision of Prof. Lixin
He at the University of Science and Technology of China.

GNU Lesser General Public License v3.0
Permissions of this copyleft license are conditioned on making available complete source code of licensed works and modifications under the same license or the GNU GPLv3. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights. However, a larger work using the licensed work through interfaces provided by the licensed work may be distributed under different terms and without source code for the larger work.
Permissions:
Commercial use
Modification
Distribution
Patent use
Private use

Limitations:
Liability
Warranty

Conditions:
License and copyright notice
Disclose source
State changes
Same license (library)
"""


class multiXR:
    """
    Class for holding XR (eg HR, SR) matrices.

    Attributes
    ----------
    XR_des : tuple
        Describe the type of multiXR.

    des : str
        One of 'H', 'S', 'r_x', 'r_y', 'r_z'.

    R_num : int
        The number of R indicators.

    R_direct_coor : np.ndarray[int]
        shape=(R_num, 3). R's fractional coordinates.

    basis_num : int
        Matrix dimension of XR.

    XR : scipy.sparse.csc_matrix
        shape=(R_num, (basis_num + 1)*basis_num / 2). Data of HR or SR or rR. Each row of the sparse matrix corresponds to the upper triangular 
        part of each X[R]. XR's unit is eV for HR, is angstrom for rR.
    """

    XR_des = ('H', 'S', 'r_x', 'r_y', 'r_z')

    def __init__(self, des):
        # Is it HR, SR, or rR
        if des in multiXR.XR_des:
            self.des = des
        else:
            raise ValueError("multiXR des must be one of 'H', 'S', 'r_x', 'r_y', 'r_z'")
        
    def set_XR(self, R_num, R_direct_coor, basis_num, XR):
        """
        Set the data related to XR, the number and fractional coordinates of R, 
        the dimension of the matrix (square matrix) corresponding to each R, 
        and the specific value of XR.

        Parameters
        ----------
        R_num : int, 
            the number of R indicators.

        R_direct_coor : np.ndarray[int]
            shape=(R_num, 3). R's fractional coordinates.

        basis_num : int 
            matrix dimension of XR.

        XR : scipy.sparse.csc_matrix[complex] or np.ndarray[complex]
            if the type is csc_matrix, shape=(R_num, (basis_num + 1)*basis_num / 2); if the type is 
            ndarray, shape=(R_num, basis_num, basis_num). Data of HR or SR or rR. Each row of the sparse 
            matrix corresponds to the upper triangular part of each X[R]. XR's unit is eV for HR, 
            is angstrom for rR.
        """
        self.R_num = R_num
        self.R_direct_coor = R_direct_coor
        self.basis_num = basis_num
        
        if isinstance(XR, csr_matrix):
            self.XR = XR
        elif isinstance(XR, np.ndarray):
            self.XR = XR


import numpy as np
from scipy.sparse import csr_matrix
import struct
import re


def abacus_readHR(nspin, HR_route, HR_unit, **kwarg):
    """
    Reads the Hamiltonian matrix from a file and returns it in a structured format.
    Parameters:
    nspin (int): Number of spin components. If nspin is 4, the Hamiltonian is complex.
    HR_route (str): Path to the file containing the Hamiltonian data.
    HR_unit (str): Unit of the Hamiltonian values ('eV' or 'Ry').
    **kwarg: Additional keyword arguments.
    Returns:
    m_HR: An instance of multiXR containing the Hamiltonian matrix and related data.
    Notes:
    - The function reads the Hamiltonian matrix from the specified file.
    - The file is expected to contain specific keywords and data in a predefined format.
    - The Hamiltonian matrix is stored in a compressed sparse row (CSR) format and then converted to a dense matrix.
    - The function supports both real and complex Hamiltonian matrices based on the value of nspin.
    """
    if HR_unit == 'eV':
        unit = 1.0
    elif HR_unit == 'Ry':
        unit = Ry_to_eV

    with open(HR_route, 'r') as fread:
        while True:
            line = fread.readline().split()
            if line[0] == 'Matrix':
                break
        basis_num = int(line[-1])
        line = fread.readline().split()
        R_num = int(line[-1])
        R_direct_coor = np.zeros([R_num, 3], dtype=int)

        if nspin != 4:
            HR =  np.zeros((R_num,basis_num,basis_num), dtype=float)
        else:
            HR =  np.zeros((R_num,basis_num,basis_num), dtype=complex)

 
        for iR in range(R_num):
            line = fread.readline().split()
            R_direct_coor[iR, 0] = int(line[0])
            R_direct_coor[iR, 1] = int(line[1])
            R_direct_coor[iR, 2] = int(line[2])
            data_size = int(line[3])
            
            if nspin != 4:
                data = np.zeros((data_size,), dtype=float)
            else:
                data = np.zeros((data_size,), dtype=complex)

            indices = np.zeros(data_size, dtype=int)
            indptr = np.zeros((basis_num+1,), dtype=int)

            if data_size != 0:
                if nspin != 4:
                    line = fread.readline().split()
                    for index in range(data_size):
                        data[index] = float(line[index]) * unit
                else:
                    line = re.findall('[(](.*?)[)]', fread.readline())
                    for index in range(data_size):
                        value = line[index].split(',')
                        data[index] = complex(float(value[0]), float(value[1])) * unit

                line = fread.readline().split()
                for index in range(data_size):
                    indices[index] = int(line[index])

                line = fread.readline().split()
                for index in range(basis_num+1):
                    indptr[index] = int(line[index])
            
        
            HR[iR,:,:] =  csr_matrix((data, indices, indptr),shape=(basis_num, basis_num)).todense()

    m_HR = multiXR('H')
    m_HR.set_XR(R_num, R_direct_coor, basis_num, HR)

    return m_HR

def abacus_readSR(nspin, SR_route, **kwarg):
    """
    Reads the SR matrix from a file and returns it in a structured format.
    Parameters:
    -----------
    nspin : int
        The number of spin components. If nspin is 4, the data is treated as complex.
    SR_route : str
        The file path to the SR matrix file.
    **kwarg : dict
        Additional keyword arguments.
    Returns:
    --------
    m_SR : multiXR
        An instance of the multiXR class containing the SR matrix data.
    Notes:
    ------
    The function reads the SR matrix from the specified file, processes the data,
    and stores it in a multiXR object. The SR matrix can be either real or complex
    depending on the value of nspin.
    """
    unit = 1.0

    with open(SR_route, 'r') as fread:
        while True:
            line = fread.readline().split()
            if line[0] == 'Matrix':
                break
        basis_num = int(line[-1])
        line = fread.readline().split()
        R_num = int(line[-1])
        R_direct_coor = np.zeros([R_num, 3], dtype=int)

        if nspin != 4:
            SR =  np.zeros((R_num,basis_num,basis_num), dtype=float)
        else:
            SR =  np.zeros((R_num,basis_num,basis_num), dtype=complex)

        for iR in range(R_num):
            line = fread.readline().split()
            R_direct_coor[iR, 0] = int(line[0])
            R_direct_coor[iR, 1] = int(line[1])
            R_direct_coor[iR, 2] = int(line[2])
            data_size = int(line[3])
            
            if nspin != 4:
                data = np.zeros((data_size,), dtype=float)
            else:
                data = np.zeros((data_size,), dtype=complex)

            indices = np.zeros(data_size, dtype=int)
            indptr = np.zeros((basis_num+1,), dtype=int)

            if data_size != 0:
                if nspin != 4:
                    line = fread.readline().split()
                    for index in range(data_size):
                        data[index] = float(line[index]) * unit
                else:
                    line = re.findall('[(](.*?)[)]', fread.readline())
                    for index in range(data_size):
                        value = line[index].split(',')
                        data[index] = complex(float(value[0]), float(value[1])) * unit

                line = fread.readline().split()
                for index in range(data_size):
                    indices[index] = int(line[index])

                line = fread.readline().split()
                for index in range(basis_num+1):
                    indptr[index] = int(line[index])
        
            SR[iR,:,:] =  csr_matrix((data, indices, indptr),shape=(basis_num, basis_num)).todense()

    m_SR = multiXR('S')
    m_SR.set_XR(R_num, R_direct_coor, basis_num, SR)

    return m_SR

