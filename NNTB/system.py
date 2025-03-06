import numpy as np
from e3nn.o3 import Irreps, spherical_harmonics
from .util import *
from ase.data import atomic_numbers
import torch
from torch_scatter import scatter
from .balanced_irreps import BalancedIrreps
from .edgesegnn import edgeSEGNN
from .edgesegnnO2 import edgeSEGNNonSO2
from .crystal import Crystal
from .hamiltonian_irrep import HamiltonianIrrepTransformer
from .tightbinding import bandfromgraph_torch
import os
from ase.dft.kpoints import monkhorst_pack
import multiprocessing as mp


class TBSystem:
    """
    A class to represent a Tight-Binding System (TBSystem).
    Attributes
    ----------
    spin : bool
        Indicates if spin is considered in the system.
    element_encoding : Dict[str, int]
        Dictionary encoding elements to integers.
    element_decoding : Dict[int, str]
        Dictionary decoding integers to elements.
    max_atoms : int
        Maximum number of atoms in the system.
    orbitalstrings : str
        Orbital strings for the system.
    orbitalabacus : Dict[str, str]
        Orbital abacus for the system.
    dimensions : int
        Number of dimensions in the system.
    neighbour_cells : int
        Number of neighbor cells considered.
    neighbour_cutoff : float
        Cutoff distance for neighbors.
    gauss_width : float
        Gaussian width for distance basis.
    dist_onehotsize : int
        Size of the one-hot vector for distance.
    MLP_layers : int
        Number of layers in the MLP.
    weight_hidden : int
        Number of hidden units in the weight network.
    model : str
        Model type used in the system.
    weightNetType : str
        Type of weight network.
    KANgrid : int
        Size of the KAN grid.
    MPgrid : Tuple[int, int, int]
        Monkhorst-Pack grid dimensions.
    monkhorst_pack : torch.Tensor
        Monkhorst-Pack grid for k-points.
    distance_encoding : str
        Encoding type for distances.
    lmax_irrep_Y : int
        Maximum irreducible representation for Y.
    lmax_irrep_hidden : int
        Maximum irreducible representation for hidden layers.
    hidden_features : int
        Number of hidden features.
    convolution_layers : int
        Number of convolution layers.
    norm : str
        Normalization type.
    hidden_irreps : Optional
        Hidden irreducible representations.
    type_onehot : torch.Tensor
        One-hot vectors for each element.
    type_atomicZ : torch.Tensor
        Atomic numbers one-hot encoding.
    hamiltoniantransformer : HamiltonianIrrepTransformer
        Transformer for Hamiltonian irreducible representations.
    node_input_irreps : Irreps
        Irreducible representations for node input.
    hamiltonian_irreps : Irreps
        Irreducible representations for Hamiltonian.
    SOC_irreps : Irreps
        Irreducible representations for SOC.
    edge_attr_irreps : Irreps
        Irreducible representations for edge attributes.
    node_attr_irreps : Irreps
        Irreducible representations for node attributes.
    additional_message_irreps : Irreps
        Additional message irreducible representations.
    force_irreps : Irreps
        Irreducible representations for force.
    edge_size_irreps : Irreps
        Irreducible representations for edge size.
    asymmetry_irreps : Irreps
        Irreducible representations for asymmetry.
    band_kpoints : torch.Tensor
        Tensor of Monkhorst-Pack k-points.
    Methods
    -------
    processGraph(graph, name='default'):
        Processes a graph based on the model type.
    processGraphFolder(folderin, folderout='./', full_process=True):
        Processes all graphs in a folder.
    gettorchModel(type="Hamiltonian"):
        Initializes and returns a torch model.
    print_param(file=None):
        Prints or saves the parameters of the TBSystem.
    """
    def __init__(self,
                 element_encoding = None,
                 orbitalstrings: str = None, orbitalabacus = None,
                 spin = False,
                 neighbour_cutoff = 8,
                 max_atoms = 10,
                 dimensions=3,
                 neighbour_cells=2,
                 model = 'edgesegnn',
                 gauss_width = 250.0,
                 dist_onehotsize = 128,
                 lmax_irrep_Y = 4,
                 lmax_irrep_hidden = 4,
                 hidden_features = 128,
                 hidden_irreps = None,
                 weightNetType = 'MLP',
                 MLP_layers = 3,
                 weight_hidden = 64,
                 KANgrid = 5,
                 convolution_layers = 4,
                 norm = "batch",
                 MPgrid = (4,4,4),
                 distance_encoding = 'gaussian_custom',
                 max_domain_size=10.0
                 ) -> None:
        self.spin = spin  # Indicates if spin is considered in the system
        self.element_encoding = element_encoding  # Dictionary encoding elements to integers
        self.element_decoding = { value: key for key, value in self.element_encoding.items()}  # Dictionary decoding integers to elements
        self.max_atoms = max_atoms  # Maximum number of atoms in the system
        self.orbitalstrings = orbitalstrings  # Orbital strings for the system
        self.orbitalabacus = orbitalabacus  # Orbital abacus for the system
        self.dimensions = dimensions  # Number of dimensions in the system
        self.neighbour_cells = neighbour_cells  # Number of neighbor cells considered
        self.neighbour_cutoff = neighbour_cutoff  # Cutoff distance for neighbors
        self.gauss_width = gauss_width  # Gaussian width for distance basis
        self.dist_onehotsize = dist_onehotsize  # Size of the one-hot vector for distance
        self.MLP_layers = MLP_layers  # Number of layers in the MLP
        self.weight_hidden = weight_hidden  # Number of hidden units in the weight network
        self.model = model  # Model type used in the system
        self.weightNetType = weightNetType  # Type of weight network
        self.KANgrid = KANgrid  # Size of the KAN grid
        self.monkhorst_pack = MPgrid  # Monkhorst-Pack grid for k-points
        self.band_kpoints = torch.tensor(monkhorst_pack(MPgrid), dtype=torch.float32)  # Tensor of Monkhorst-Pack k-points
        self.distance_encoding = distance_encoding  # Encoding type for distances
        self.max_domain_size = max_domain_size  # Maximum domain size for grid partitioning
        
        assert self.max_atoms > len(self.element_encoding), f"{len(self.element_encoding)} atoms are encoded but the number of max atoms is {self.max_atoms}"

        #Extend element encoding for max_atoms (for extendable model)
        dummy_i = len(self.element_encoding)
        while len(self.element_encoding) < self.max_atoms:
            self.element_encoding[f'DU{dummy_i}']= dummy_i
            dummy_i += 1
            
        #The onehot vectors for each element    
        self.type_onehot = torch.eye(len(self.element_encoding),dtype=torch.float32)
        
        # Fill the tensors with atomic numbers one-hot encoding
        self.type_atomicZ = torch.tensor([atomic_numbers.get(element, 0)/294.0 for element in self.element_encoding],dtype=torch.float32).view(-1, 1)


        self.lmax_irrep_Y = lmax_irrep_Y
        self.lmax_irrep_hidden = lmax_irrep_hidden
        self.hidden_features = hidden_features
        self.convolution_layers=convolution_layers
        self.norm=norm
        
        self.hamiltoniantransformer = HamiltonianIrrepTransformer(self.orbitalstrings,spin=self.spin)

        self.node_input_irreps = Irreps(f"{max_atoms}x0e").simplify() #+8x0e
        self.hamiltonian_irreps = self.hamiltoniantransformer.ham_irrep
        self.SOC_irreps = self.hamiltoniantransformer.soc_irrep
        self.edge_attr_irreps = Irreps.spherical_harmonics(lmax_irrep_Y)
        self.node_attr_irreps = Irreps.spherical_harmonics(lmax_irrep_Y)
        self.additional_message_irreps = Irreps("1x0e")
        self.force_irreps = Irreps("1x1o")
        self.edge_size_irreps = Irreps(f"{self.dist_onehotsize}x0e")

        self.orbitals = orbitals_from_str_yzx(self.orbitalstrings,spin=self.spin)
        self.num_orbitals = len(self.orbitals)

        
        if hidden_irreps == None:
            self.hidden_irreps = BalancedIrreps(self.lmax_irrep_hidden, self.hidden_features, False)
        else:
            self.hidden_irreps = hidden_irreps
        print(f"Hidden Irreducible Representation = {self.hidden_irreps}")
       
    def processGraph(self, graph, name='default'):
        edge_vec = graph.edge_vec
        edge_dist = graph.edge_dist
        elementonehot = graph.elementonehot
        graph.edge_attr = spherical_harmonics(self.edge_attr_irreps, edge_vec, normalize=True, normalization='integral')
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean")
        graph.x = elementonehot
        graph.additional_message_features = torch.unsqueeze(edge_dist, dim=1)

        if self.model == 'oldsegnn':
            def rbf(r):
                mu = np.linspace(0, self.neighbour_cutoff, self.dist_onehotsize).reshape(1, -1)
                return np.exp(-self.gauss_width * (r[:, np.newaxis] - mu) ** 2)
            graph.edge_gauss_width = torch.tensor(rbf(np.array(edge_dist)), dtype=torch.float32)
            graph.hopping_irrep = self.hamiltoniantransformer.to_irrep_Hamiltonian(graph.hopping)
            graph.overlap_irrep = self.hamiltoniantransformer.to_irrep_Overlap(graph.overlap)
        
        elif self.model == 'edgesegnn':
            if hasattr(graph, 'hopping') and hasattr(graph, 'overlap'):
                if self.spin:
                    # SOC is considered, only self-energy SOC is considered (inter-atomic SOC did not work well in the tested examples...)
                    graph.hopping_irrep, soc_irrep = self.hamiltoniantransformer.to_irrep_Hamiltonian(graph.hopping, selfenergy=graph.selfenergy)
                    graph.soc_irrep = self.hamiltoniantransformer.fullsoc_to_nodeonly(soc_irrep, graph)
                else:
                    if graph.spin == True:
                        raise ValueError("Spin is considered in the graph, but the system does not support it. Set spin=True in the system or reprocess the graph with spin=False.")
                    graph.hopping_irrep = self.hamiltoniantransformer.to_irrep_Hamiltonian(graph.hopping)
                graph.overlap_irrep = self.hamiltoniantransformer.to_irrep_Overlap(graph.overlap)
                graph.band = bandfromgraph_torch(graph, self)
            else:
                graph.spin = self.spin
        
        elif self.model == 'edgesegnno2':
            
            rot_mat = edge_rot_mat(edge_vec)
            graph.D = []
            top_l = max(self.hamiltonian_irreps.lmax, self.hidden_irreps.lmax)
            for l in range(1, top_l + 1):
                graph.D.append(Irrep(l, 1).D_from_matrix(rot_mat))
            
            if hasattr(graph, 'hopping') and hasattr(graph, 'overlap'):
                graph.hopping_irrep = self.hamiltoniantransformer.to_irrep_Hamiltonian(graph.hopping)
                graph.overlap_irrep = self.hamiltoniantransformer.to_irrep_Overlap(graph.overlap)
        
        graph.preprocessed = True
        graph.model = self.model
        graph.name = name
        return graph

    def processGraphFolder(self, folderin, folderout='./', full_process=True):
        os.makedirs(f"{folderout}", exist_ok=True)
        for filename in os.listdir(folderin):
            if filename.endswith('.cif') and os.path.isfile(os.path.join(folderin, filename)):
                prefix = filename[:-4]
                print(f"Processing: {prefix}")
                crystal = Crystal(folderin, prefix, self, undirected=True)
                graph = crystal.graph
                
                if full_process:
                    graph = self.processGraph(graph, name=prefix)
                else:
                    graph.preprocessed = False
                    graph.model = None
                    graph.name = prefix

                # Save the graph
                torch.save(graph, f"{folderout}/{prefix}.ptg" if full_process else f"{folderout}/{prefix}.tg")

    
    def gettorchModel(self,type="Hamiltonian"):
        #initialize a model

        if type == "HamiltonianSOC" or type == "SOC":
            assert self.spin == True, "SOC can only be used if spin = True"

        if self.model == 'edgesegnn' :

            if type == "Hamiltonian" or type == "Overlap":
                outputType = 'edge2'
                edgeoutput_irrep = self.hamiltonian_irreps
                nodeoutput_irrep = None
            elif type == "SOC":
                outputType = 'node'
                edgeoutput_irrep = None
                nodeoutput_irrep = self.SOC_irreps
            elif type == "HamiltonianSOC":
                outputType = 'edge2_plus_node'
                edgeoutput_irrep = self.hamiltonian_irreps
                nodeoutput_irrep = self.SOC_irreps
            elif type == "Force":
                outputType = 'node'
                edgeoutput_irrep = None
                nodeoutput_irrep = self.force_irreps

            else:
                raise ValueError(f"{type} is not a valid model type. Only Force, Hamiltonian, Overlap, or SOC are valid.")
                
            model = edgeSEGNN(self.node_input_irreps,
                    self.hidden_irreps,
                    edgeoutput_irrep,
                    nodeoutput_irrep,
                    self.edge_attr_irreps,
                    self.node_attr_irreps,
                    self.edge_size_irreps,
                    self.neighbour_cutoff,
                    self.gauss_width,
                    ConvLayers=self.convolution_layers,
                    MLP_layers=self.MLP_layers,
                    weight_hidden=self.weight_hidden,
                    weightNetType=self.weightNetType,
                    KANgrid_size = self.KANgrid,
                    norm=self.norm,
                    additional_message_irreps=self.additional_message_irreps,
                    outputType = outputType,
                    distance_encoding=self.distance_encoding)
            
        elif self.model == 'edgesegnno2' :
            model = edgeSEGNNonSO2(self.node_input_irreps,
                    self.hidden_irreps,
                    self.hamiltonian_irreps,
                    self.edge_size_irreps,
                    self.neighbour_cutoff,
                    self.gauss_width,
                    ConvLayers=self.convolution_layers,
                    MLP_layers=self.MLP_layers,
                    weight_hidden=self.weight_hidden,
                    weightNetType=self.weightNetType,
                    KANgrid_size = self.KANgrid,
                    norm=self.norm,
                    resnet_edge = True,
                    resnet_node = True,
                    additional_message_irreps=self.additional_message_irreps)
        else:
            pass
            

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("The model you built has %d parameters." % params)
        
        return model
               

    def orbital_index(self, graph, element = 'all', orbital = 'all', spin = 'all', atom_num = 'all'):
        elemlst=[self.element_decoding[i] for x in graph.elementonehot.to('cpu') for i, value in enumerate(self.type_onehot) if torch.equal(value, x)]
        ham_template = [(idx, orb) for idx, atom in enumerate(elemlst) for orb in self.orbitals]

        atom_idx = []
        if atom_num == 'all':
            atom_idx += [i for i in range(len(ham_template))]
        else:
            assert type(atom_num) == int or type(atom_num) == list, f"{atom_num} is not a valid atom number. Choose an integer or a list of integers."
            if isinstance(atom_num, int):
                assert atom_num < graph.x.size(0), f"atom_num {atom_num} is out of range. Maximum allowed is {graph.x.size(0) - 1}."
            elif isinstance(atom_num, list):
                assert max(atom_num) < graph.x.size(0), f"Maximum atom_num {max(atom_num)} is out of range. Maximum allowed is {graph.x.size(0) - 1}."
            atom_idx += [i for atom in atom_num for i in range(atom*len(self.orbitals),atom*len(self.orbitals)+len(self.orbitals))]

        elem_idx = []
        assert element in self.element_decoding, f"{element} is not a valid element. Choose from {self.element_decoding.keys()}"
        if element == 'all':
            elem_idx += [i for i in range(len(ham_template))]
        else:
            elem_idx += [i for i, orb in enumerate(self.orbitals*len(graph.x)) if elemlst[orb[0]] == element]

        orb_idx = []
        if orbital == 's':
            l=0
        elif orbital == 'p':
            l=1
        elif orbital == 'd':
            l=2
        elif orbital == 'f':
            l=3
        else:
            raise ValueError(f"{orbital} is not a valid orbital. Only s, p, d, or f are valid.")

        if orbital == 'all':
            orb_idx += [i for i in range(len(ham_template))]
        else:
            orb_idx += [i for i, orb in enumerate(self.orbitals*len(graph.x)) if orb[2] == l]

        spin_idx = []
        if self.spin == False:
            spin_idx += [i for i in range(len(ham_template))]
        else:
            if spin == 'all' or spin == 'up':
                spin_idx += [i for i, orb in enumerate(self.orbitals*len(graph.x)) if orb[3] == -1]
            if spin == 'all' or spin == 'down':
                spin_idx += [i for i, orb in enumerate(self.orbitals*len(graph.x)) if orb[3] == 1]
        
        final_idx = list(set(elem_idx) & set(orb_idx) & set(spin_idx) & set(atom_idx))
        return final_idx

        
      
    def print_param(self, file=None):
        text = f"""
    Dimension: {self.dimensions}D
    Max_Atoms: {self.max_atoms}
    Neighbour Cells: {self.neighbour_cells}
    Neighbour Cutoff: {self.neighbour_cutoff}
    Width Gaussian: {self.gauss_width}
    Dist Onehot Size: {self.dist_onehotsize}
    MLP Layers: {self.MLP_layers}
    Weight Hidden: {self.weight_hidden}
    Model: {self.model}
    Weight Net Type: {self.weightNetType}
    KAN Grid: {self.KANgrid}
    Convolution Layers: {self.convolution_layers}
    Normalization: {self.norm}
    Spin: {self.spin}
    Element Encoding: {self.element_encoding}
    Element Decoding: {self.element_decoding}
    Orbital Strings: {self.orbitalstrings}
    Orbital Abacus: {self.orbitalabacus}
    Lmax Irrep Y: {self.lmax_irrep_Y}
    Lmax Irrep Hidden: {self.lmax_irrep_hidden}
    Hidden Features: {self.hidden_features}
    Hidden Irreps: {self.hidden_irreps}
    Type Atomic Z: {self.type_atomicZ}
    Node Input Irreps: {self.node_input_irreps}
    Hamiltonian Irreps: {self.hamiltonian_irreps}
    SOC Irreps: {self.SOC_irreps}
    Edge Attr Irreps: {self.edge_attr_irreps}
    Node Attr Irreps: {self.node_attr_irreps}
    Additional Message Irreps: {self.additional_message_irreps}
    Force Irreps: {self.force_irreps}
    Edge Size Irreps: {self.edge_size_irreps}
    Asymmetry Irreps: {self.asymmetry_irreps}
    Monkhorst Pack: {self.monkhorst_pack}
    Distance Encoding: {self.distance_encoding}
    Band Kpoints: {self.band_kpoints}
    """
        if file:
            with open(file, 'w') as f:
                f.write(text)
        else:
            print(text)


