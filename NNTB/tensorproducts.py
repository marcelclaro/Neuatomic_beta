import torch
import torch.nn as nn

from e3nn.o3 import Irreps, FullyConnectedTensorProduct, Irrep
from e3nn.nn import Gate, NormActivation

from math import sqrt

from .KAN.bsrbf_kan import BSRBF_KAN
from .KAN.fast_kan import FastKAN
from .KAN.efficient_kan import EfficientKAN

import torch.nn.functional as F

class ScaledSiLU(torch.nn.Module):
    def __init__(self, scale=2.0):  # Adjust scale as needed
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.silu(self.scale * x)

"""
Copyright (c) 2024 Marcel S. Claro

GNU Lesser General Public License v3.0
"""

"""
Based on https://github.com/RobDHess/Steerable-E3-GNN
Modified version where weights are not shared and are obtained from a Neural network
"""

#TODO: tp_rescale can lead to NaN understand and solve it (Common on the embedding block, why?)

class O3TensorProductWeighted(nn.Module):
    """ A class to perform tensor product operations with Swish gating and weighting for O(3) equivariant neural networks.

    Parameters
    ----------
    irreps_in1 : o3.Irreps
        Input irreps.
    irreps_out : o3.Irreps
        Output irreps.
    irreps_in2 : o3.Irreps, optional
        Second input irreps. Defaults to scalar irreps if not provided.
    irreps_weight: o3.Irreps, optional
        Weight MLP input irreps. If None, internal weights are used.
    bias : bool, optional
        If true, adds a bias term. Default is True.
    irrep_normalization : str, optional
        Normalization of irreps. Default is None.
    path_normalization : str, optional
        Normalization of paths. Default is None.
    weightNetType : str, optional
        Type of network for weights. Options are 'BSRBF-KAN', 'FAST-KAN', 'EfficientKAN', 'MLP'. Default is 'MLP'.
    weight_hidden : int, optional
        Number of hidden units in the weight network. Default is 64.
    weight_MLPlayers : int, optional
        Number of layers in the weight MLP. Default is 2.
    KANgrid_size : int, optional
        Grid size for KAN networks. Default is 30.
    KANspline_order : int, optional
        Spline order for KAN networks. Default is 3.
    KANbase_activation : callable, optional
        Base activation function for KAN networks. Default is torch.nn.SiLU.
    KANgrid_range : list, optional
        Grid range for KAN networks. Default is [-0.5, 8.5].
    tp_rescale : bool, optional
        If true, rescales the tensor product. Default is True.
    """

    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, irreps_weight = None,
                 bias = True,irrep_normalization = None,path_normalization = None,
                 weightNetType = 'MLP', weight_hidden=64,
                 weight_MLPlayers=2, 
                 KANgrid_size = 30,KANspline_order = 3,KANbase_activation = torch.nn.SiLU, KANgrid_range=[-0.5, 8.5],
                 tp_rescale=True) -> None:
        super().__init__()

        self.bias = bias
        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2 if not defined (simple scalar by default)
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2
        self.tp_rescale = tp_rescale

        # Build the layers for tensor product
        if irreps_weight == None:  #internal weigths if there is no weight net
            self.ext_weight = False
            self.tp = FullyConnectedTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out, shared_weights=True, irrep_normalization=irrep_normalization,path_normalization=path_normalization)
        
        else: #Case weight has his own network, weights are not shared
            self.ext_weight = True
            self.tp = FullyConnectedTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out, internal_weights = False, shared_weights=False, irrep_normalization=irrep_normalization,path_normalization=path_normalization)
            
        # Creates network for the weigths

        if self.ext_weight:
            assert weightNetType in ['BSRBF-KAN','FAST-KAN','EfficientKAN','MLP']
            if weightNetType == 'BSRBF-KAN':
                self.weightNN = BSRBF_KAN([irreps_weight.dim,weight_hidden,self.tp.weight_numel],grid_size = KANgrid_size,spline_order = KANspline_order,base_activation = KANbase_activation,grid_range=KANgrid_range)
            elif weightNetType == 'FAST-KAN':
                self.weightNN = FastKAN([irreps_weight.dim,weight_hidden,self.tp.weight_numel],num_grids = KANgrid_size,base_activation = KANbase_activation,grid_min=KANgrid_range[0],grid_max=KANgrid_range[1])
            elif weightNetType == 'EfficientKAN':
                self.weightNN = EfficientKAN([irreps_weight.dim,self.tp.weight_numel],grid_size = KANgrid_size,base_activation = KANbase_activation,grid_range=KANgrid_range)
            else:
                weightlayers = []
                weightlayers.append(nn.Linear(irreps_weight.dim, weight_hidden))
                weightlayers.append(nn.ReLU())
                for _ in range(weight_MLPlayers-1):
                    weightlayers.append(nn.Linear(weight_hidden, weight_hidden))
                    weightlayers.append(nn.ReLU())
                weightlayers.append(nn.Linear(weight_hidden, self.tp.weight_numel))

                self.weightNN = nn.Sequential(*weightlayers)              


        
        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()
        # Store tuples of slices and corresponding biases in a list
        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = irreps_out.slices()[slice_idx]
                out_bias = torch.zeros(self.irreps_out_dims[slice_idx], dtype=self.tp.weight.dtype)
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]

        # Initialize the correction factors
        self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        self.tensor_product_init()
        # Adapt parameters so they can be applied using vector operations.
        self.vectorise()

    def tensor_product_init(self) -> None:
        with torch.no_grad():
            if self.ext_weight:
                # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
                slices_fan_in = {}  # fan_in per slice
                for instr in self.tp.instructions:
                    slice_idx = instr[2]
                    fan_in = self.irreps_in1[instr[0]].mul * self.irreps_in2[instr[1]].mul
                    slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                                fan_in if slice_idx in slices_fan_in.keys() else fan_in)
                    if self.tp_rescale:
                        sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
                    else:
                        sqrt_k = 1.
                    self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

                # Initialize the biases
                for (out_slice_idx, out_slice, out_bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
                    sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx] if out_slice_idx in slices_fan_in.keys() else 1.0 )
                    out_bias.uniform_(-sqrt_k, sqrt_k)
            else:
                # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
                slices_fan_in = {}  # fan_in per slice
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    slice_idx = instr[2]
                    mul_1, mul_2, mul_out = weight.shape
                    fan_in = mul_1 * mul_2
                    slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                                fan_in if slice_idx in slices_fan_in.keys() else fan_in)
                # Do the initialization of the weights in each instruction
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.tp_rescale:
                        sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
                    else:
                        sqrt_k = 1.
                    weight.data.uniform_(-sqrt_k, sqrt_k)
                    self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

                # Initialize the biases
                for (out_slice_idx, out_slice, out_bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
                    sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx] if out_slice_idx in slices_fan_in.keys() else 1.0 )
                    out_bias.uniform_(-sqrt_k, sqrt_k)

    def vectorise(self):
        """ Adapts the bias parameter and the sqrt_k corrections so they can be applied using vectorised operations """

        # Vectorise the bias parameters
        if len(self.biases) > 0:
            with torch.no_grad():
                self.biases = torch.cat(self.biases, dim=0)
            self.biases = nn.Parameter(self.biases)

            # Compute broadcast indices.
            bias_idx = torch.LongTensor()
            for slice_idx in range(len(self.irreps_out_orders)):
                if self.irreps_out_orders[slice_idx] == 0:
                    out_slice = self.irreps_out.slices()[slice_idx]
                    bias_idx = torch.cat((bias_idx, torch.arange(out_slice.start, out_slice.stop).long()), dim=0)

            self.register_buffer("bias_idx", bias_idx, persistent=False)
        else:
            self.biases = None

        # Now onto the sqrt_k correction
        sqrt_k_correction = torch.zeros(self.irreps_out.dim)
        for instr in self.tp.instructions:
            slice_idx = instr[2]
            slice, sqrt_k = self.slices_sqrt_k[slice_idx]
            sqrt_k_correction[slice] = sqrt_k

        # Make sure bias_idx and sqrt_k_correction are on same device as module
        self.register_buffer("sqrt_k_correction", sqrt_k_correction, persistent=False)

    def forward_tp_rescale_bias(self, data_in1, data_in2,weight = None) -> torch.Tensor:
        if data_in2 == None:
            data_in2 = torch.ones_like(data_in1[:, 0:1])

        if self.ext_weight:
            assert weight != None, "Weighted tensor product had no weight"
            data_out = self.tp(data_in1, data_in2,self.weightNN(weight))
        else:
            assert weight == None, "Weighted tensor product had weight but was not initialized to support it"
            data_out = self.tp(data_in1, data_in2)

        # Apply corrections
        if self.tp_rescale:
            data_out /= self.sqrt_k_correction

        # Add the biases
        if self.biases is not None and self.bias:
            data_out[:, self.bias_idx] += self.biases
        return data_out

    def forward(self, data_in1, data_in2,weight = None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2,weight)
        return data_out


class O3TensorProductSwishGateWeighted(O3TensorProductWeighted):
    """
    A class to perform tensor product operations with Swish gating and weighting for O(3) equivariant neural networks.
    Methods
    -------
    path_exists(irreps_in1, irreps_in2, ir_out):
        Static method to check if a path exists between input irreducible representations and an output irreducible representation.
    __init__(irreps_in1, irreps_out, irreps_in2=None, irreps_weight=None, bias=True, irrep_normalization=None, path_normalization=None, nonlinearity_type='gate', weightNetType='BSRBF-KAN', weight_MLPlayers=2, weight_hidden=64, KANgrid_size=30, KANspline_order=3, KANbase_activation=torch.nn.SiLU, KANgrid_range=[-0.5, 8.5], tp_rescale=True):
        Initializes the O3TensorProductSwishGateWeighted class with the given parameters.
    forward(data_in1, data_in2, weight=None) -> torch.Tensor:
        Forward pass of the tensor product operation with gating and weighting.
    """
    @staticmethod
    def path_exists(irreps_in1, irreps_in2, ir_out):
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()
        ir_out = Irrep(ir_out)

        for _, ir1 in irreps_in1:
            for _, ir2 in irreps_in2:
                if ir_out in ir1 * ir2:
                    return True
        
        return False
    
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None,irreps_weight = None,
                 bias = True,irrep_normalization = None,path_normalization = None,
                 nonlinearity_type = 'gate',
                 weightNetType = 'BSRBF-KAN',
                 weight_MLPlayers=2, weight_hidden=64,
                 KANgrid_size = 30,KANspline_order = 3,KANbase_activation = torch.nn.SiLU, KANgrid_range=[-0.5, 8.5],
                 tp_rescale=True) -> None:

        # nequip style gate parity check
        nonlinearity_scalars = {1: nn.SiLU(), -1: torch.tanh}
        nonlinearity_gates = {1: nn.SiLU(), -1: torch.tanh}

        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_out
                if ir.l == 0
                and self.path_exists(irreps_in1,irreps_in2,ir)
            ]
        )

        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_out
                if ir.l > 0
                and self.path_exists(irreps_in1,irreps_in2,ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = (
                "0e"
                if self.path_exists(irreps_in1, irreps_in2, "0e")
                else "0o"
            )
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])


            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    nonlinearity_scalars[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[nonlinearity_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            conv_irreps_out = irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=conv_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=nonlinearity_scalars[1],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )


        # Build the layers
        super(O3TensorProductSwishGateWeighted, self).__init__(irreps_in1, conv_irreps_out, irreps_in2,irreps_weight = irreps_weight,
                                                                bias = bias,irrep_normalization = irrep_normalization,path_normalization = path_normalization,
                                                                weightNetType = weightNetType,
                                                                weight_MLPlayers=weight_MLPlayers,weight_hidden=weight_hidden,
                                                                KANgrid_size = KANgrid_size,KANspline_order = KANspline_order,KANbase_activation = KANbase_activation, KANgrid_range=KANgrid_range,
                                                                tp_rescale=tp_rescale)
        
        self.gate = equivariant_nonlin


    def forward(self, data_in1, data_in2,weight=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2,weight)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out
