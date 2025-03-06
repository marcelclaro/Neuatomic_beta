# Neuatomic 

Neuatomic is a project that uses equivariant message passing graph networks to create tight-binding models for heterostructures. It uses as input hamiltonians from NAO-DFT calculations.

It is related to the projects:
- https://github.com/RobDHess/Steerable-E3-GNN (SEGNN)
- https://github.com/mzjb/DeepH-pack
- https://github.com/QuantumLab-ZY/HamGNN
- https://github.com/mir-group/nequip

## Features
- Integration with ABACUS
- Experimental support for Kolmogorov-Arnold Networks and projection on SO(2)
- Basic calculations: Bandstructure, DOS, and dielectric constant 

## Dependencies
All dependencies are listed in the `env` file. Make sure to install them before running the code.
For example:
* [PyTorch](https://pytorch.org/) 
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
* [E3NN](https://e3nn.org/) 

## Usage
Documentation is  still a work in progress.

The `Example` jupyter notebook has a minimal example of how to use the code.
It requires the sample data available at: https://github.com/marcelclaro/Neuatomic_beta/releases

Brief description of the files:

- `system.py`: Parameters container
- `crystal.py`: Generic crystal data structure, .cif and DFT data reader, and graph constructor
- `hamiltonian_irrep.py`: Convert a Hamiltonian matrix to flat vector of irreducible representations.
- `trainwithband.py`: Training algorithm
- `edgesegnn.py` (or `edgesegnnO2.py` for SO(2) projection version): Equivariant message passing network based on SEGNN, modified to output edge attribute in addition of node attributes.
- `tensorproducts.py` (or `tensorproductsO2.py` for SO(2) projection version): Tensor product PyTorch modules
- `tightbinding.py`: Basic calculations (Bandstructure, DOS, and dielectric constant) from a graph. Native in PyTorch, use dense matrices and full diagonalization, suitable for small graphs.
- `tightbinding_scipy.py`: Same as above but native on sparse SciPy matrices and Chebyshev polinomial approximations, suitable for larger systems.

## License
This project is licensed under the GNU Lesser General Public License v3.0 as described in the LICENSE file.
As it reuses other codes, specific licenses for each code are also described in the LICENSE file.


by Marcel Santos Claro from University of Santiago de Compostela

Group: https://www.usc.es/ciqus/es/grupos/cond-matter-chem 
