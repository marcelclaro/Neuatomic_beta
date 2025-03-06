import torch
import torch.nn as nn

from e3nn.o3 import Irreps, Linear, spherical_harmonics, Irrep
from e3nn.nn import Gate, NormActivation

from math import sqrt
from e3nn.o3._wigner import wigner_3j
import math

from .KAN.bsrbf_kan import BSRBF_KAN
from .KAN.fast_kan import FastKAN

"""
Copyright (c) 2024 Marcel S. Claro

GNU Lesser General Public License v3.0
"""

#Based on https://github.com/RobDHess/Steerable-E3-GNN

#TODO: tp_rescale can lead to NaN, understand and solve it (Common on the embedding  block, why?)

class O3onO2TensorProductWeighted(nn.Module):
    """ A class to perform tensor product operations weighting for O(3) equivariant neural networks using O(2) space projection.

    Parameters
    ----------
    irreps_in1 : o3.Irreps
        Input irreps.
    irreps_out : o3.Irreps
        Output irreps.
    irreps_weight : o3.Irreps, optional
        Weight MLP input irreps.
    trasformationO2 : str, optional
        Type of transformation for O2. Options are 'both', 'in', 'out', 'none'. Default is 'both'.
    bias : bool, optional
        Whether to include bias. Default is False.
    irrep_normalization : str, optional
        Normalization type for irreps. Options are 'component', 'norm', 'none'. Default is None.
    path_normalization : str, optional
        Normalization type for paths. Options are 'element', 'path', 'none'. Default is None.
    weightNetType : str, optional
        Type of network for weights. Options are 'BSRBF-KAN', 'FAST-KAN', 'MLP'. Default is 'MLP'.
    weight_hidden : int, optional
        Number of hidden units in the weight MLP. Default is 64.
    weight_MLPlayers : int, optional
        Number of layers in the weight MLP. Default is 2.
    KANgrid_size : int, optional
        Grid size for KAN. Default is 30.
    KANspline_order : int, optional
        Spline order for KAN. Default is 3.
    KANbase_activation : callable, optional
        Base activation function for KAN. Default is torch.nn.SiLU.
    KANgrid_range : list, optional
        Grid range for KAN. Default is [0, 8.0].
    """

    def __init__(self, irreps_in1, irreps_out, irreps_weight = None, trasformationO2 = 'both',
                 bias = False,irrep_normalization = None,path_normalization = None,
                 weightNetType = 'MLP', weight_hidden=64,
                 weight_MLPlayers=2, 
                 KANgrid_size = 30,KANspline_order = 3,KANbase_activation = torch.nn.SiLU, KANgrid_range=[0, 8.0]) -> None:
        super().__init__()

        self.__name__ = "O3onO2TensorProductWeighted"
        self.irreps_in1 = irreps_in1 #input irreducible representations
        self.irreps_out = irreps_out #output irreducible representations
        self.trasformationO2 = trasformationO2
        assert self.trasformationO2 in ['both', 'in', 'out', 'none'], "trasformationO2 should be 'both', 'in', 'out' or 'none'"
        self.bias = bias
        if self.bias:
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
                    out_bias = torch.zeros(self.irreps_out_dims[slice_idx], dtype=torch.float32) #TODO dtype is right?
                    self.biases += [out_bias]
                    self.biases_slices += [out_slice]
                    self.biases_slice_idx += [slice_idx]

        #Contruct the indexes for the tensor product
        self.out_hmo_idx = [] #indexes of m=0 for output
        self.in_hmo_idx = [] #indexes of m=0 for input

        self.out_posm_idx = [] #index for positive m for output
        self.out_negm_idx = [] #index for negative m for output
        self.in_negm_idx= [] #index for negative m for input
        self.in_posm_idx= [] #index for positive m for input

        #slices of each representation
        sliceout = irreps_out.slices() 
        slicein = irreps_in1.slices()

        lmax = 10 #Max l on spherical harmonics of the vector
        self.Irrep_HS = Irreps.spherical_harmonics(lmax)


        #Calculate the normalization factor lf is not [1.0, 0.0, 1.0 ,0.0, ... 1.0 ...]
        norm_HS = []
        strider = 0
        HSvalue = spherical_harmonics(self.Irrep_HS, torch.tensor([0.0,1.0,0.0]), normalize=True, normalization='integral')
        for l in range(lmax+1):
            norm_HS += [HSvalue[strider+l]]
            strider += (2*l+1)

        

        #here we go for each output value
        for o_idx, (_,irr_o) in enumerate(self.irreps_out):
            for i_idx, (_,irr_i) in enumerate(self.irreps_in1): #sum on input irreps
                for _, (_,irr_f) in enumerate(self.Irrep_HS): #in an unlimited (max of 20)
                    if(irr_o in irr_i * irr_f):   #only if there is a valid path - using e3nn (e.i. |li-lf|  <= lo <= li+lf and po=pi*pf)
                        #m=0 is in the position l for each representation, stride 2*l+1 wich is the dimension of the representation
                        self.out_hmo_idx.append(range(sliceout[o_idx].start+irr_o.l,sliceout[o_idx].stop,2*irr_o.l+1)) 
                        self.in_hmo_idx.append(range(slicein[i_idx].start+irr_i.l,slicein[i_idx].stop,2*irr_i.l+1))
                        
                        #As above, m!=0 is in the position l(+ or - m) for each representation, stride 2*l+1 wich is the dimension of the representation
                        for mo in range(1,min(irr_o.l,irr_i.l)+1):
                            self.out_posm_idx.append(range(sliceout[o_idx].start+irr_o.l+mo,sliceout[o_idx].stop,2*irr_o.l+1))
                            self.out_negm_idx.append(range(sliceout[o_idx].start+irr_o.l-mo,sliceout[o_idx].stop,2*irr_o.l+1))
                            self.in_posm_idx.append(range(slicein[i_idx].start+irr_i.l+mo,slicein[i_idx].stop,2*irr_i.l+1))
                            self.in_negm_idx.append(range(slicein[i_idx].start+irr_i.l-mo,slicein[i_idx].stop,2*irr_i.l+1))
                        
                        break #break if find the first valid SH because of bijection (https://doi.org/10.48550/arXiv.2302.03655)

        
        totalweights = 0
        valids = {}
        for o_idx, (mul_o,irr_o) in enumerate(self.irreps_out):  
            for i_idx, (mul_i,irr_i) in enumerate(self.irreps_in1): #sum on input irreps
                for f_idx, (_,irr_f) in enumerate(self.Irrep_HS): #in an unlimited (max of 20)
                    if(irr_o in irr_i * irr_f):   #only if there is a valid path - using e3nn (e.i. |li-lf|  <= lo <= li+lf and po=pi*pf)
                        totalweights += mul_o * mul_i
                        if (o_idx, i_idx) in valids:
                            valids[(o_idx, i_idx)].append(f_idx)
                        else:
                            valids[(o_idx,i_idx)] =  [f_idx]

        self.validweigths = 0
        
        #Normalization default initialization
        normalization_coefficients = []
        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"


        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        #Weight initialization factor.
        weightfactor = []
        
        for (o_idx,i_idx), f_idx_lst in valids.items():
            self.validweigths += irreps_out[o_idx][0] * irreps_in1[i_idx][0]

            #Normalization coeff calculation
            if irrep_normalization == "component":
                alpha = irreps_out[o_idx][1].dim
            if irrep_normalization == "norm":
                alpha = irreps_in1[i_idx][1].dim * sum([self.Irrep_HS[f_idx][1].dim for f_idx in f_idx_lst])
            if irrep_normalization == "none":
                alpha = 1

            if path_normalization == "element":
                x = sum(irreps_in1[i_idx_b].mul * len(f_idx_lst_b) for (o_idx_b,i_idx_b), f_idx_lst_b in valids.items() if o_idx_b == o_idx)
            if path_normalization == "path":
                x = self.irreps_in1[i_idx].mul * len(f_idx_lst)
                x *= len([i_idx_b for (_,i_idx_b) in valids.keys() if i_idx == i_idx_b])
            if path_normalization == "none":
                x = 1

            if x > 0.0:
                alpha /= x


            normalization_coefficients += [sqrt(alpha)]
            #Weight distribution depends on multiplicity (Xavier init)
            a = (6 / (irreps_out[o_idx][0] * irreps_in1[i_idx][0]) * len(f_idx_lst))**0.5
            weightfactor += [1.0]
       
        #print(f"Valid weights = {100*self.validweigths/totalweights}%")


        self.hzero = []
        self.hpos = []
        self.hneg = []

        for irr_o_irr_i, lf in valids.items():
            cpos = []
            cneg = []
            czero = 0

            l_out = irreps_out[irr_o_irr_i[0]][1].l
            l_in = irreps_in1[irr_o_irr_i[1]][1].l

            for f in lf:
                l_f = self.Irrep_HS[f][1].l
                wig = wigner_3j(l_in, l_f, l_out)
                czero +=  norm_HS[l_f]  * wig[l_in, l_f, l_out] #(-1)**(-l_in + l_f) * math.sqrt(2 * l_out + 1) * 

            max_m = min(l_out, l_in)
            for m in range(1, max_m +1):
                totalpos = 0
                totalneg = 0

                for f in lf:
                    l_f = self.Irrep_HS[f][1].l
                    wig = wigner_3j(l_in, l_f, l_out)
                    factor =  norm_HS[l_f]  # (-1)**(-l_in + l_f - m) * math.sqrt(2 * l_out + 1) * 
                    totalpos +=  factor * wig[l_in+m, l_f, l_out+m] #factor * 
                    totalneg +=  factor * wig[l_in+m, l_f, l_out-m] #factor * 
                    
                cpos.append(totalpos)
                cneg.append(totalneg)

            self.hzero.append(czero)
            self.hpos.append(cpos)
            self.hneg.append(cneg)


        self.paths_num = sum([len(valids.values())])

        #normalization...
        for n, coeff in enumerate(normalization_coefficients):
            self.hzero[n] *= coeff
            self.hpos[n] = [hpos * coeff for hpos in self.hpos[n]]
            self.hneg[n] = [hneg * coeff for hneg in self.hneg[n]]
     
        # Creates a MLP network for the weigths
        if irreps_weight != None:
            self.sharedweight = False
            assert weightNetType in ['BSRBF-KAN','FAST-KAN','MLP'], "weightNetType is not a valid one (use 'BSRBF-KAN','FAST-KAN' or'MLP')"
            if weightNetType == 'BSRBF-KAN':
                self.weightNN = BSRBF_KAN([1,weight_hidden,self.validweigths],grid_size = KANgrid_size,spline_order = KANspline_order,base_activation = KANbase_activation,grid_range=KANgrid_range)
            if weightNetType == 'FAST-KAN':
                self.weightNN = FastKAN([1,weight_hidden,self.validweigths],num_grids = KANgrid_size,spline_order = KANspline_order,base_activation = KANbase_activation,grid_min=KANgrid_range[0],grid_max=KANgrid_range[1])
            else:
                weightlayers = []
                weightlayers.append(nn.Linear(irreps_weight.dim, weight_hidden))
                weightlayers.append(nn.ReLU())
                for _ in range(weight_MLPlayers-1):
                    weightlayers.append(nn.Linear(weight_hidden, weight_hidden))
                    weightlayers.append(nn.ReLU())
                weightlayers.append(nn.Linear(weight_hidden, self.validweigths))

                self.weightNN = nn.Sequential(*weightlayers)    
        else:
            self.sharedweight = True
            self.s_weight = torch.nn.Parameter(torch.zeros(self.validweigths))
            with torch.no_grad():
                #Here weigths are initialized depending on path multiplicity
                #TODO It would be more elegant if use valids, but works well too...
                strider = 0
                for idx, _ in enumerate(self.hzero): #goes through all the output representations
                    out_idx = self.out_hmo_idx[idx]
                    in_idx = self.in_hmo_idx[idx] #input
                    mat_size = len(out_idx) * len(in_idx) #The number of weights depends on the multiplicity of each representation (given by the len of idx's)
                    torch.nn.init.uniform_(self.s_weight[strider:strider+mat_size], -weightfactor[idx], weightfactor[idx])
                    strider += mat_size
        
        if self.bias:
            # Initialize the biases
            bias_var = 1.0 #bias variance
            for  out_bias in  self.biases:
                out_bias.uniform_(-bias_var, bias_var)

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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.Irrep_HS.simplify()} "
            f"-> {self.irreps_out.simplify()} | {self.paths_num} paths | {self.validweigths} weights)"
        ) 
    
    def tensorproduct(self, data_in , D , weight=None):
        #Rotation of the input by rotation matrix

        data_in = data_in.clone()

        if self.trasformationO2 != 'none': 
            assert len(D) >= self.irreps_in1.lmax and len(D) >= self.irreps_out.lmax, "The rotation matrix D should have lmax(dim=0) bigger than lmax of input and output irrep" 

        if self.trasformationO2 == 'in' or self.trasformationO2 == 'both':
            strider = 0
            # TODO now it only works with batch
            for mul, ir in self.irreps_in1:
                l = ir.l
                dim = 2 * l + 1  # Dimension of the SO(3) irrep
                
                if l != 0:
                    for n in range(mul):
                        # Directly use einsum without unsqueeze/squeeze
                        data_in[:, strider:strider + dim] = torch.einsum('bij,bj->bi',
                                                                        D[l-1],  # Rotation matrix (D[l-1])
                                                                        data_in[:, strider:strider + dim])
                        strider += dim
                else:
                    strider += mul
                       
        data_out = torch.zeros((data_in.shape[0], self.irreps_out.dim),device=data_in.device) #initialize result vector, Assuming data_in1 is of shape (batch_size, dim_in)

        #get weights from the NN and the slices for each m
        if self.sharedweight: #if shared weight
            fullweight = self.s_weight.to(data_in.device)  #TODO manage better the device thing

        else: #use NN, depends on weight input
            fullweight = self.weightNN(weight)


        z = "" if self.sharedweight else "b"

        strider = 0
        another_idx = 0
        for idx, hzero in enumerate(self.hzero):  # goes through all the output representations
            out_idx = self.out_hmo_idx[idx]
            in_idx = self.in_hmo_idx[idx]  # input
            mat_size = len(out_idx) * len(in_idx)  # The number of weights depends on the multiplicity of each representation (given by the len of idx's)

            # Calculate using Einstein summation (einsum), summing over the specified indices
            if hzero != 0:
                weight_reshaped = fullweight[strider:strider+mat_size].reshape(len(out_idx), len(in_idx)) if self.sharedweight else fullweight[:,strider:strider+mat_size].reshape(-1,len(out_idx), len(in_idx))
                data_out[:, out_idx] += torch.einsum(f'{z}ij,bj->bi', hzero * weight_reshaped, data_in[:, in_idx])

            # Same for m != 0, but now it depends on +m and -m
            for ms in range(len(self.hpos[idx])):
                out_pos_idx = self.out_posm_idx[another_idx]
                out_neg_idx = self.out_negm_idx[another_idx]
                in_pos_idx = self.in_posm_idx[another_idx]
                in_neg_idx = self.in_negm_idx[another_idx]

                if self.hpos[idx][ms] != 0 or self.hneg[idx][ms] != 0:
                    weight_reshaped = fullweight[strider:strider+mat_size].reshape(len(out_pos_idx), len(in_pos_idx)) if self.sharedweight else fullweight[:,strider:strider+mat_size].reshape(-1,len(out_pos_idx), len(in_pos_idx))
                    
                    # For positive m component
                    data_out[:, out_pos_idx] += torch.einsum(f'{z}ij,bj->bi', self.hpos[idx][ms] * weight_reshaped, data_in[:, in_pos_idx]) - \
                                                torch.einsum(f'{z}ij,bj->bi', self.hneg[idx][ms] * weight_reshaped, data_in[:, in_neg_idx])
                    
                    # For negative m component
                    data_out[:, out_neg_idx] += torch.einsum(f'{z}ij,bj->bi', self.hneg[idx][ms] * weight_reshaped, data_in[:, in_pos_idx]) + \
                                                torch.einsum(f'{z}ij,bj->bi', self.hpos[idx][ms] * weight_reshaped, data_in[:, in_neg_idx])
                another_idx += 1

            strider += mat_size                 

        if self.trasformationO2 == 'out' or self.trasformationO2 == 'both':
            strider = 0
            for idx, (mul, ir) in enumerate(self.irreps_out):
                l = ir.l
                dim = 2 * l + 1  # Dimension of the SO(3) irrep
                
                if l != 0:
                    for n in range(mul):
                        # Directly use einsum without unsqueeze/squeeze
                        data_out[:, strider:strider + dim] = torch.einsum('bij,bj->bi',
                                                                        D[l-1].transpose(1, 2),  # Transpose the rotation matrix
                                                                        data_out[:, strider:strider + dim])
                        strider += dim
                else:
                    strider += mul

        # Add the biases to the scalar 
        if self.bias and (self.biases is not None):
            data_out[:, self.bias_idx] += self.biases
        
        return data_out

    def forward(self, data_in1, D,weight=None) -> torch.Tensor:
        data_out = self.tensorproduct(data_in1, D,weight)
        return data_out


class O3onO2TensorProductSwishGateWeighted(O3onO2TensorProductWeighted):
    """
    A class to perform tensor product operations with Swish gate activation on O3 projecting on O2 groups.
    Methods
    -------
    path_exists(irreps_in1, irreps_in2, ir_out):
        Static method to check if a path exists between input irreducible representations and output irreducible representation.
    __init__(irreps_in1, irreps_out, irreps_weight=None, trasformationO2='both', bias=True, irrep_normalization=None, path_normalization=None, nonlinearity_type='gate', weightNetType='MLP', weight_hidden=64, weight_MLPlayers=2, KANgrid_size=30, KANspline_order=3, KANbase_activation=torch.nn.SiLU, KANgrid_range=[0, 8.0]):
        Initializes the tensor product layer with specified parameters.
    forward(data_in1, D, weight=None):
        Forward pass of the tensor product layer.
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
    
    
    def __init__(self, irreps_in1, irreps_out, irreps_weight = None, trasformationO2 = 'both',
                 bias = True,irrep_normalization = None,path_normalization = None,
                 nonlinearity_type = 'gate',
                 weightNetType = 'MLP', weight_hidden=64,
                 weight_MLPlayers=2, 
                 KANgrid_size = 30,KANspline_order = 3,KANbase_activation = torch.nn.SiLU, KANgrid_range=[0, 8.0]) -> None:
        
        nonlinearity_scalars = {1: nn.SiLU(), -1: torch.tanh}
        nonlinearity_gates = {1: nn.SiLU(), -1: torch.tanh}

        lmax = 10 #Max l on spherical harmonics of the vector
        Irrep_HS = Irreps.spherical_harmonics(lmax)

        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_out
                if ir.l == 0
                and self.path_exists(irreps_in1,Irrep_HS,ir)
            ]
        )

        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_out
                if ir.l > 0
                and self.path_exists(irreps_in1,Irrep_HS,ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = (
                "0e"
                if self.path_exists(irreps_in1, Irrep_HS, "0e")
                else "0o"
            )
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])



            # TO DO, it's not that safe to directly use the
            # dictionary
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
        super(O3onO2TensorProductSwishGateWeighted, self).__init__(irreps_in1, conv_irreps_out,irreps_weight = irreps_weight, trasformationO2 = trasformationO2,
                                                                   bias=bias,irrep_normalization = irrep_normalization,path_normalization =path_normalization,
                                                                   weightNetType = weightNetType, weight_hidden=weight_hidden,
                                                                   weight_MLPlayers=weight_MLPlayers, 
                                                                   KANgrid_size = KANgrid_size,KANspline_order = KANspline_order,KANbase_activation = KANbase_activation, KANgrid_range=KANgrid_range)
        
        self.gate = equivariant_nonlin

    def forward(self, data_in1, D,weight=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.tensorproduct(data_in1, D,weight)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out


class O3LinearGated(Linear):
    """
    A linear layer with gated nonlinearity for O(3) equivariant neural networks.
    This class extends the standard linear layer by adding a gated nonlinearity
    mechanism. It supports two types of nonlinearities: "gate" and "norm".
    Attributes:
        irreps_in (Irreps): Input irreducible representations.
        irreps_out (Irreps): Output irreducible representations.
        nonlinearity_type (str): Type of nonlinearity to apply. Default is "gate".
        gate (nn.Module): The nonlinearity module applied after the linear transformation.
    Methods:
        forward(data_in1: torch.Tensor) -> torch.Tensor:
            Applies the linear transformation followed by the gated nonlinearity.
    """
    def __init__(self, irreps_in, irreps_out,nonlinearity_type = "gate") -> None:
        nonlinearity_scalars = {1: nn.SiLU(), -1: torch.tanh}
        nonlinearity_gates = {1: nn.SiLU(), -1: torch.tanh}
        
        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_out
                if ir.l == 0
                and ir in irreps_in
            ]
        )

        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_out
                if ir.l > 0
                and ir in irreps_in
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = "0e"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])



            # TO DO, it's not that safe to directly use the
            # dictionary
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
        super(O3LinearGated, self).__init__(irreps_in,conv_irreps_out,biases=True)

        self.gate = equivariant_nonlin


    def forward(self, data_in1) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = super(O3LinearGated, self).forward(data_in1)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out
