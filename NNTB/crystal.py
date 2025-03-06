import numpy as np
from .util import *
from ase.data import atomic_numbers
from ase.io import read
import os
import torch
import torch_geometric
import torch_geometric.data
from NNTB.abacus import abacus_readHR, abacus_readSR
import ase.spacegroup
import concurrent.futures
import pickle  #TODO: Delete after debugging

"""
Copyright (c) 2024 Marcel S. Claro

GNU Lesser General Public License v3.0
"""


# A class for atom information
class Atom:
    """
    Represents an atom in a crystal structure.

    Attributes:
        unitcell_index (int): Index of the unit cell.
        element_index (int): Index of the element in the unit cell.
        element (str): Symbol of the element.
        atm_number (int): Atomic number of the element.
        position (np.ndarray): Vector position of the atom.
        R_vec (np.ndarray): R vector relative to the unit cell.

    Methods:
        __str__(): Returns a string representation of the atom.
    """
    def __init__(self,unitcell_index,element_index,element,position,R_vec=np.array([0,0,0], dtype=int)) -> None:
        self.unitcell_index = unitcell_index #unit cell index.
        self.element_index = element_index #element in the unit cell.
        self.element = element #Element Symbol.
        self.atm_number = atomic_numbers[element] #Atomic number to encode element
        self.position = position #Vector position.
        self.R_vec = R_vec #R vector relative to unit cell.
    def __str__(self):
        return f"Atom {self.unitcell_index} in the unit cell, element index: {self.element_index}, position: {self.position}"

# A class to store bond information
class Bond:
    """
    A class to represent a bond between two atoms in a supercell.

    Attributes
    ----------
    i : int
        Index of atom i in the supercell.
    j : int
        Index of atom j in the supercell.
    rij : array-like
        Vector from atom i to atom j.
    R : array-like
        Unit cell shift.
    norm : float, optional
        The norm of the vector rij. If not provided, it is calculated using np.linalg.norm(rij).

    Methods
    -------
    __str__():
        Returns a string representation of the bond.
    """
    def __init__(self, i, j, rij, R, norm=None) -> None:
        pass

        pass
    def __init__(self,i, j, rij, R, norm = None) -> None:
        self.i = i   #index of atom i in the supercell cell
        self.j = j  #index of atom i in the supercell cell
        self.rij = rij #vector r1-rj
        self.R = R #unit cell shift
        #The rij norm
        if not norm:
            self.norm = np.linalg.norm(rij)
        else:
            self.norm = norm
    def __str__(self):
        return f"Bond between {self.i} and {self.j}, conecting vector(x,y,z) = {self.rij}"
    
# A class for all the whole crystal properties and information
class Crystal:
    """
    Crystal class for representing and processing crystal structures reading from ABACUS DFT files.
    Args:
        filepath (str): The file path from where the crystal is read.
        prefix (str): A prefix for all files (.cif, .csr Hamiltonian/Overlap).
        tbsystem (object): An object containing the tight-binding system parameters.
        periodic (bool, optional): If True, extend the unit cell for periodic structures. Defaults to True.
        undirected (bool, optional): If True, ensures the graph is undirected. Defaults to True.
        grouptolerance (float, optional): Tolerance for space group determination. Defaults to 1e-1.
    Attributes:
        filepath (str): Path to the directory containing the crystal files.
        prefix (str): Prefix for all files (.cif, .csr Hamiltonian/Overlap).
        ASEstructure (ASE Atoms object): ASE object for further processing.
        unitcell (list): List of unit cell atoms.
        supercell (list): Big cell for searching all neighbors.
        used_atoms_index (list): Index of all used (with bonds) atoms in the extended cell.
        used_R (list): List of unique R vectors used in the extended cell.
        unique_bonds (list): List of unique (unidirectional) determined bonds to be included in the TB.
        lattice_vectors (numpy array): Unit cell lattice vectors.
        dimensions (int): Number of model dimensions (usually 3, rarely 2).
        neighbour_cells (int): Number of neighbor unit cells to construct supercell.
        spin (bool): Indicates if full spin model is used.
        unique_elements (numpy array): Unique chemical elements in the lattice.
        orbitals (list): List of orbital strings.
        found_abacus (bool): Indicates if Abacus files were found.
        found_SO (bool): Indicates if spin-orbit coupling files were found.
        pointgroup (list): List of point group transformations.
        graph (torch_geometric.data.Data): Graph representation of the crystal structure.
    Methods:
        __init__(self, filepath, prefix, tbsystem, periodic=True, undirected=True, grouptolerance=1e-1):
            Initializes the Crystal object and processes the crystal structure.
        plotBonds(self):
            Plots the bonds in the crystal structure using matplotlib.
    """
    def __init__(self,filepath,prefix,tbsystem, periodic = True, undirected = True, grouptolerance = 1e-1, supercell_dim=[1,1,1]):
        """
        Initializes the Crystal object.
        Args:
            filepath (str): The file path from where the crystal is read.
            prefix (str): A prefix for all files (.cif, .csr Hamiltonian/Overlap).
            tbsystem (object): An object containing the tight-binding system parameters.
            periodic (bool, optional): If True, extend the unit cell for periodic structures. Defaults to True.
            undirected (bool, optional): If True, ensures the graph is undirected. Defaults to True.
            grouptolerance (float, optional): Tolerance for space group determination. Defaults to 1e-1.
            supercell_dim (list, optional): Dimensions to scale the supercell. Defaults to [1,1,1].
        """
        torch.set_default_dtype(torch.float32)
        self.filepath = filepath  #filepath from where the crystal is read
        self.prefix = prefix  #A prefix for all files(.cif, .csr Hamiltonian/Overlap)
        self.undirected = undirected
        
        structure = read(self.filepath+self.prefix+'.cif') #Read cif structure using ASE
        structure = structure * (supercell_dim[0], supercell_dim[1], supercell_dim[2]) 

        self.ASEstructure = structure #keep ASE object for further processing
        self.unitcell = []  #list with unit cell atoms
        self.supercell = [] #Big cell for search for all neighbours
        self.used_atoms_index  = []  #Index of all used (with bonds) atoms in the extended cell
        self.used_R = [np.array([0,0,0], dtype=int)]
        self.unique_bonds = []  #A list of unique (unidirectional) determined bonds to be included in the TB
        self.lattice_vectors = structure.cell  #unit cell lattice vectors
        self.dimensions=tbsystem.dimensions,   #number of model dimensions (usually 3 - rarely 2)
        self.neighbour_cells=tbsystem.neighbour_cells  #number of neighbour unit cells to constuct supercell
        self.spin = tbsystem.spin #If full spin model
        
        #determine the number of different chemical elements in the lattice and the number of atom for each element
        self.unique_elements = np.unique(structure.get_chemical_symbols())
        self.orbitals = tbsystem.orbitalstrings
        
        self.found_abacus = False
        self.found_SO = False
        
        #It is not used so far
        #self.atomsperelement = []
        #for element in self.unique_elements:
        #    self.atomsperelement.append(len([atom for atom in structure if atom.symbol == element]))

        #Creates the list of point group transformations
        #self.pointgroup_small = [rotation for rotation,trans in zip(ase.spacegroup.get_spacegroup(self.ASEstructure,symprec=10**(-10)).rotations,ase.spacegroup.get_spacegroup(self.ASEstructure,symprec=10**(-10)).translations)]
        #TODO only improper?
        self.pointgroup = [rotation for rotation,trans in zip(ase.spacegroup.get_spacegroup(self.ASEstructure,grouptolerance).rotations,ase.spacegroup.get_spacegroup(self.ASEstructure,symprec=10**(-1)).translations)] # if np.all(trans == 0.0)
        assert len(self.pointgroup) >=1, "Error finding the space goup"

        #create unit cell list
        for n, atom in enumerate(structure):
            index_in_unique_elements = next((i for i, x in enumerate(self.unique_elements) if x == atom.symbol), None)
            self.unitcell.append(Atom(n, index_in_unique_elements,atom.symbol,structure.get_positions()[n]))
            

        
        if periodic:
        #if periodic extend the unit cell for periodic structures
            def generate_combinations(k, n, current_combination=[]):
                if k == 0:
                    return [current_combination.copy()]  # Copy the current combination to avoid reference issues
                else:
                    combinations = []
                    for i in range(-n, n + 1):
                        combinations.extend(generate_combinations(k - 1, n, current_combination + [i]))
                    return sorted(combinations, key=np.linalg.norm)
            R_combinations = np.array(generate_combinations(tbsystem.dimensions, tbsystem.neighbour_cells),dtype=int)

            for vec_R in R_combinations:
                for n, atom in enumerate(structure):
                    displacement = vec_R[0]*self.lattice_vectors[0]+vec_R[1]*self.lattice_vectors[1]+vec_R[2]*self.lattice_vectors[2]
                    index_in_unique_elements = next((i for i, x in enumerate(self.unique_elements) if x == atom.symbol), None)
                    newposition = structure.get_positions()[n] + displacement
                    if(all(-tbsystem.neighbour_cutoff <= coord <= lattice_vector + tbsystem.neighbour_cutoff for coord, lattice_vector in zip(newposition, self.lattice_vectors.diagonal()))):
                        self.supercell.append(Atom(n, index_in_unique_elements,atom.symbol,newposition,np.array(vec_R,dtype=int)))
        else:
            #otherwise supercell is equal to unit cell
            self.supercell = self.unitcell

        # Generate pairs and calculate vector difference, if inside the cufoff it adds to list of bonds

        #Parallel processing of the bonds
        def process_atom(i):
            local_used_atoms_index = [i]
            local_unique_bonds = [Bond(i, i, np.array([0, 0, 0]), np.array([0, 0, 0], dtype=int))]  # self interaction
            local_used_R = []

            for j in range(i + 1, len(self.supercell)):
                atom1 = self.supercell[i]
                atom2 = self.supercell[j]
                vector_difference = atom1.position - atom2.position
                norm = np.linalg.norm(vector_difference)
                if norm <= tbsystem.neighbour_cutoff:
                    local_unique_bonds.append(Bond(i, j, vector_difference, atom2.R_vec, norm))
                    local_used_atoms_index.append(j)
                    if not any(np.array_equal(atom2.R_vec, used_R_vec) for used_R_vec in self.used_R):
                        local_used_R.append(atom2.R_vec)

            return local_used_atoms_index, local_unique_bonds, local_used_R

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_atom, range(len(self.unitcell))))

        for local_used_atoms_index, local_unique_bonds, local_used_R in results:
            self.used_atoms_index.extend(local_used_atoms_index)
            self.unique_bonds.extend(local_unique_bonds)
            self.used_R.extend(local_used_R)

        self.used_atoms_index = np.unique(self.used_atoms_index)
        #self.used_atoms_index.sort()  #Not necessary

        #Here I check if there is a DFT log file and try to read forces
        if os.path.exists(self.filepath+self.prefix+'.log'):
            force_vectors = []
            with open(self.filepath+self.prefix+'.log', 'r') as file:
                inside_marker = False
                jumped_one = False
                found_forces = False
                for line in file:
                    line = line.strip()
                    if "TOTAL-FORCE (eV/Angstrom)" in line:
                        inside_marker = True
                        found_forces = True
                        continue
                    if inside_marker and ("----------" in line):
                        if jumped_one:
                            break
                        else:
                            jumped_one = True
                            continue
                    if inside_marker and line:
                        parts = line.split()
                        if len(parts) == 4:
                            vector = [float(parts[1]), float(parts[2]), float(parts[3])]
                            force_vectors.append(vector)
                if not found_forces:
                    force_vectors = None
        else:
            force_vectors = None


        #To store the hamiltonian:
        loadedR_vec_lst = []
        loadedHR_lst = []
        loadedSR_lst = []
        
        # Construct file paths
        hr_file_path = self.filepath + self.prefix + '.HR.csr'
        sr_file_path = self.filepath + self.prefix + '.SR.csr'
        hr_SO_file_path = self.filepath + self.prefix + '.HR_SO.csr'
        sr_SO_file_path = self.filepath + self.prefix + '.SR_SO.csr'
        
        if os.path.exists(hr_file_path) and os.path.exists(sr_file_path):
            self.found_abacus = True
            self.found_SO = False
        elif os.path.exists(hr_SO_file_path) and os.path.exists(sr_SO_file_path):
            self.found_abacus = True
            self.found_SO = True
        else:
            print(f"Abacus Files do not exist at {self.filepath}: cif file only graph generated!")
        
        if self.spin:
            if self.found_abacus:
                assert self.found_SO, "Spin is set to True in the TBsystem but abacus files found has not SOC"
        else:
            if self.found_abacus:
                assert not self.found_SO, "Spin is set to False in the TBsystem but abacus files found has SOC"
            
        #This is the matrix template which order the rows an columns of the hamiltonian
        self.sys_template = [(idx,orb)  for idx, atom in enumerate(self.unitcell) for orb in orbitals_from_str_yzx(self.orbitals,self.spin)] #All atoms have the same number of orbitals, it simplifies the hopping matrix shape

        #TODO: Need to check if the unit cell in abacus is the same (atom order e.g.) -> At 27/03/24 It seems right with version 3.5.4 
        #Orbitals from LCAO method can be more complex or in a different order
        abacus_template = [(idx,orb)  for idx, atom in enumerate(self.unitcell) for orb in orbitals_from_str_abacus(tbsystem.orbitalabacus[atom.element],self.spin)]
        

        #What is read from abacus files
        if self.found_abacus:
            if self.found_SO:
                abacus_HR = abacus_readHR(4,hr_SO_file_path,'Ry')
                abacus_SR = abacus_readSR(4,sr_SO_file_path)
            else:
                abacus_HR = abacus_readHR(1,hr_file_path,'Ry')
                abacus_SR = abacus_readSR(1,sr_file_path)

            # Find the correspondence between abacus and our system orbitals 
            indices_matrix = [abacus_template.index(label) for label in self.sys_template]
            
            #Get the hamiltonian in our format cutting extra orbitals
            for n, R in enumerate(abacus_HR.R_direct_coor):
                if any(np.array_equal(R,used) for used in self.used_R ):
                    assert np.array_equal(R, abacus_SR.R_direct_coor[n]), "SR and HR order are not the same"

                    loadedR_vec_lst.append(R)  # Add R in the list

                    symmetric_HR = abacus_HR.XR[n]
                    shortsymmetric_HR = symmetric_HR[indices_matrix][:, indices_matrix]
                    if self.found_SO:
                        loadedHR_lst.append(shortsymmetric_HR)
                    else:
                        loadedHR_lst.append(np.real(shortsymmetric_HR))

                    symmetric_SR = abacus_SR.XR[n]
                    shortsymmetric_SR = symmetric_SR[indices_matrix][:, indices_matrix]
                    if self.found_SO:
                        loadedSR_lst.append(shortsymmetric_SR)
                    else:
                        loadedSR_lst.append(np.real(shortsymmetric_SR))

            #In this part I extract the hopping/overlap matrices for each bond
            hoppings = []  #hopping matrices
            overlaps = [] #overlap matrices

        edge_src = [] #Edge source node
        edge_dst = []  #Edge destination node
        edge_shift = [] #R vector
        rij = []  #Vector
        norm = []  #Norm of the edge
        reversed = [] #indicates it is reversed one for consistency of unidirected graph
        selfenergy = [] #indicates it is onsite
        
        #TODO delete DEBUG
        self.loadedR_vec_lst = loadedR_vec_lst
        self.loadedHR_lst = loadedHR_lst
        self.loadedSR_lst = loadedSR_lst
        
        self.zerohop_flag = False


        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_bond, self.unique_bonds, timeout=None, chunksize=500))


        for result in results:
            local_hoppings, local_overlaps, local_reversed, local_edge_src, local_edge_dst, local_rij, local_norm, local_edge_shift, local_selfenergy = result
            if self.found_abacus:
                hoppings.extend(local_hoppings)
                overlaps.extend(local_overlaps)
            reversed.extend(local_reversed)
            edge_src.extend(local_edge_src)
            edge_dst.extend(local_edge_dst)
            rij.extend(local_rij)
            norm.extend(local_norm)
            edge_shift.extend(local_edge_shift)
            selfenergy.extend(local_selfenergy)
    
        if self.zerohop_flag: print("Warning: Neighbours defined in the cutoff are out of Abacus matrices range, adding zero hopping/overlap")
        
        z_attr = tbsystem.type_atomicZ[np.array([tbsystem.element_encoding[atom.element] for atom in self.unitcell])]
        elementonehot = tbsystem.type_onehot[[tbsystem.element_encoding[atom.element] for atom in self.unitcell]]

        #TODO: Here I convert everthing from numpy to torch.tensor, is it slow? Can I replace the numpy in the whole code to torch?     
        pos = torch.tensor(np.array([atom.position for atom in self.unitcell]),dtype=torch.float32,requires_grad=False)
        edge_index = torch.stack([torch.LongTensor(np.array(edge_src)), torch.LongTensor(np.array(edge_dst))], dim=0)
        edge_vec = torch.tensor(np.array(rij),dtype=torch.float32,requires_grad=False)
        
        
        edge_dist = torch.tensor(np.array(norm),dtype=torch.float32,requires_grad=False) #carry on distance to make the base change in the module if necessary
        edge_shift = torch.tensor(np.array(edge_shift),dtype=torch.int,requires_grad=False)
        
        if self.found_abacus:
            if self.found_SO:
                hopping=torch.tensor(np.array(hoppings),dtype=torch.complex64,requires_grad=False) #hopping matrix
                overlap=torch.tensor(np.array(overlaps),dtype=torch.complex64,requires_grad=False) #overlap matrix
            else:
                hopping=torch.tensor(np.array(hoppings),dtype=torch.float32,requires_grad=False) #hopping matrix
                overlap=torch.tensor(np.array(overlaps),dtype=torch.float32,requires_grad=False) #overlap matrix               
        
        if force_vectors:  
            forces=torch.tensor(np.array(force_vectors),dtype=torch.float32,requires_grad=False) #Force in that atom
        else:
            forces = None

        #   Extract the indices of the self-loops
        shift_mask = (edge_shift == torch.tensor([0, 0, 0], dtype=torch.int32)).all(dim=1) # Create a mask for edges where shift == [0, 0, 0]
        self_edges = edge_index[:, shift_mask] # Get edge indices where the condition is met
        self_loop_mask = self_edges[0] == self_edges[1] # Find the self-loops (edges where source and target nodes are the same)
        self_idx = torch.where(shift_mask)[0][self_loop_mask].tolist()


        if self.found_abacus:
            self.graph = torch_geometric.data.Data(
                pos=pos, #node(atom) position
                elementonehot=elementonehot,  # A one-hot encoder for atom type
                orbitals=self.orbitals, #orbital strings
                hopping=hopping, #hopping matrix
                overlap=overlap, #overlap matrix        
                forces=forces, #Force in that atom
                reversed = reversed, #Signal if the edge was reversed to make the graph unidirected                 edge_dist_gauss = edge_dist_gauss, #Edge distance in gaussian base
                edge_index=edge_index, #list of all edges (node_i,node_j)
                edge_vec=edge_vec, #the bonding vector
                edge_dist=edge_dist, #bond lenght 
                edge_shift = edge_shift, #unitcell where node_j is R=[int,int,int]
                z_attr=z_attr, #atomic z descriptor
                selfenergy=torch.tensor(selfenergy, dtype=torch.bool,requires_grad=False), #if a edge correspond to a selfenergy Hamiltonian (i=j)
                group_transform = self.pointgroup,
                spin = self.found_SO,
                lattice_vectors = self.lattice_vectors,
                self_idx = self_idx

            )
        else:
            self.graph = torch_geometric.data.Data(
                pos=pos, #node(atom) position
                elementonehot=elementonehot,  # A one-hot encoder for atom type
                orbitals=self.orbitals, #orbital strings
                reversed = reversed, #Signal if the edge was reversed to make the graph unidirected                 edge_dist_gauss = edge_dist_gauss, #Edge distance in gaussian base
                edge_index=edge_index, #list of all edges (node_i,node_j)
                edge_vec=edge_vec, #the bonding vector
                edge_dist=edge_dist, #bond lenght 
                edge_shift = edge_shift, #unitcell where node_j is R=[int,int,int]
                z_attr=z_attr, #atomic z descriptor
                selfenergy=torch.tensor(selfenergy, dtype=torch.bool,requires_grad=False), #if a edge correspond to a selfenergy Hamiltonian (i=j)
                group_transform = self.pointgroup,
                lattice_vectors = self.lattice_vectors,
                self_idx = self_idx,
                )
        

    def plotBonds(self):
        """
        Plots the bonds and atoms in a 3D space using matplotlib.
        This method visualizes the unique bonds and used atoms in the supercell
        by plotting them in a 3D scatter plot. Bonds are represented as lines
        between atoms, and atoms are represented as points.
        The method performs the following steps:
        1. Collects the positions of atoms involved in bonds and used atoms.
        2. Converts the positions to numpy arrays for plotting.
        3. Creates a 3D plot using matplotlib.
        4. Plots the atoms as scatter points.
        5. Plots the bonds as lines between the scatter points.
        6. Formats the axes for better visualization.
        7. Displays the plot.
        Note:
            This method requires `matplotlib` and `mpl_toolkits.mplot3d` for plotting.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        node_xyz = []
        edge_xyz = []
        
        for bond in self.unique_bonds:
            edge_xyz.append((self.supercell[bond.i].position, self.supercell[bond.j].position))
            
        for atom in self.used_atoms_index:
            node_xyz.append(self.supercell[atom].position)
            
        node_xyz = np.array(node_xyz)
        edge_xyz = np.array(edge_xyz)

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w")

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")


        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        
        _format_axes(ax)
        fig.tight_layout()
        plt.show()

    def process_bond(self,bond):

        #TODO: Delete after debugging
        pickle.dumps(bond)  # Try for the bond object
        pickle.dumps(self.supercell)  # Test for other related objects
        pickle.dumps(self.orbitals)  # Test for other related objects
        pickle.dumps(self.found_abacus)  # Test for other related objects
        pickle.dumps(self.zerohop_flag)  # Test for other related objects
        pickle.dumps(self.loadedHR_lst)  # Test for other related objects
        pickle.dumps(self.loadedSR_lst)  # Test for other related objects
        pickle.dumps(self.sys_template)  # Test for other related objects


        local_hoppings = []
        local_overlaps = []
        local_reversed = []
        local_edge_src = []
        local_edge_dst = []
        local_rij = []
        local_norm = []
        local_edge_shift = []
        local_selfenergy = []

        i = self.supercell[bond.i]
        j = self.supercell[bond.j]

        if i == j:
            local_selfenergy.append(True)
        else:
            local_selfenergy.append(False)

        R = np.array(bond.R, dtype=int)

        if self.found_abacus:
            Rindex = next((i for i, vec in enumerate(self.loadedR_vec_lst) if np.array_equal(vec, R)), None)

            if Rindex is None:
                self.zerohop_flag = True
                orb_j = len(orbitals_from_str_yzx(self.orbitals, self.spin))
                orb_i = len(orbitals_from_str_yzx(self.orbitals, self.spin))
                local_hoppings.append(np.zeros((orb_i, orb_j)))
                local_overlaps.append(np.zeros((orb_i, orb_j)))

                if i != j and self.undirected:
                    local_hoppings.append(np.zeros((orb_i, orb_j)))
                    local_overlaps.append(np.zeros((orb_i, orb_j)))
            else:
                hoppingtemplate_row = [(i.unitcell_index, orb) for orb in orbitals_from_str_yzx(self.orbitals, self.spin)]
                hoppingtemplate_col = [(j.unitcell_index, orb) for orb in orbitals_from_str_yzx(self.orbitals, self.spin)]
                hoppingindices_row = [self.sys_template.index(label) for label in hoppingtemplate_row]
                hoppingindices_col = [self.sys_template.index(label) for label in hoppingtemplate_col]

                local_hoppings.append(self.loadedHR_lst[Rindex][hoppingindices_row][:, hoppingindices_col])
                local_overlaps.append(self.loadedSR_lst[Rindex][hoppingindices_row][:, hoppingindices_col])

                if i != j and self.undirected:
                    local_hoppings.append(np.transpose(self.loadedHR_lst[Rindex][hoppingindices_row][:, hoppingindices_col]))
                    local_overlaps.append(np.transpose(self.loadedSR_lst[Rindex][hoppingindices_row][:, hoppingindices_col]))

        local_reversed.append(False)
        local_edge_src.append(j.unitcell_index)
        local_edge_dst.append(i.unitcell_index)
        local_rij.append(bond.rij)
        local_norm.append(bond.norm)
        local_edge_shift.append(np.array(bond.R, dtype=int))

        if i != j and self.undirected:
            local_reversed.append(True)
            local_edge_src.append(i.unitcell_index)
            local_edge_dst.append(j.unitcell_index)
            local_rij.append(-bond.rij)
            local_norm.append(bond.norm)
            local_edge_shift.append(np.array(bond.R, dtype=int))
            local_selfenergy.append(False)

        return (local_hoppings, local_overlaps, local_reversed, local_edge_src, local_edge_dst, local_rij, local_norm, local_edge_shift, local_selfenergy)

    

    def plotBondsPlotly(self):
        """
        Plots the atomic bonds using Plotly in a 3D scatter plot.

        This method visualizes the atomic structure by plotting nodes (atoms) and edges (bonds) 
        in a 3D space using Plotly. The nodes are represented as blue markers, and the edges 
        are represented as gray lines connecting the nodes.

        The method performs the following steps:
        1. Extracts the positions of atoms and bonds from the `supercell` attribute.
        2. Converts the positions into numpy arrays for easier manipulation.
        3. Creates a Plotly figure and adds traces for nodes and edges.
        4. Configures the layout of the plot, including axis labels and margins.
        5. Displays the plot.

        Note:
            This method requires the `plotly` and `numpy` libraries.

        Raises:
            ImportError: If the `plotly` or `numpy` libraries are not installed.

        """
        import plotly.graph_objects as go
        node_xyz = []
        edge_xyz = []

        for bond in self.unique_bonds:
            edge_xyz.append((self.supercell[bond.i].position, self.supercell[bond.j].position))

        for atom in self.used_atoms_index:
            node_xyz.append(self.supercell[atom].position)

        node_xyz = np.array(node_xyz)
        edge_xyz = np.array(edge_xyz)

        # Create the Plotly figure
        fig = go.Figure()

        # Plot the nodes
        fig.add_trace(go.Scatter3d(
            x=node_xyz[:, 0],
            y=node_xyz[:, 1],
            z=node_xyz[:, 2],
            mode='markers',
            marker=dict(size=8, color='blue', line=dict(width=2, color='white')),
            name='Nodes'
        ))

        # Plot the edges
        for vizedge in edge_xyz:
            fig.add_trace(go.Scatter3d(
                x=vizedge[:, 0],
                y=vizedge[:, 1],
                z=vizedge[:, 2],
                mode='lines',
                line=dict(color='gray', width=2),
                name='Bonds'
            ))

        # Set axes labels and title
        fig.update_layout(
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
            ),
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=0)
        )

        # Show the figure
        fig.show()