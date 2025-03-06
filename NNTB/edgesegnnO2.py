import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps, Linear
from .tensorproductsonSO2 import O3onO2TensorProductWeighted, O3onO2TensorProductSwishGateWeighted, O3LinearGated


"""
Copyright (c) 2024 Marcel S. Claro

GNU Lesser General Public License v3.0
"""

class edgeSEGNNonSO2(nn.Module):

    """
    Steerable E(3) equivariant message passing network projected on O(2) space.
    Args:
        node_input_irreps (Irreps): Input irreps for nodes.
        hidden_irreps (Irreps): Hidden irreps.
        neighbour_cutoff (float, optional): Neighbour cutoff distance. Default is 8.0.
        gauss_width (float, optional): Gaussian width for edge distance. Default is 250.0.
        ConvLayers (int, optional): Number of convolution layers. Default is 7.
        MLP_layers (int, optional): Number of MLP layers. Default is 4.
        weight_hidden (int, optional): Number of hidden units in weight network. Default is 128.
        node_embedding_hidden (int, optional): Number of hidden units in node embedding. Default is 32.
        node_embedding_layers (int, optional): Number of layers in node embedding. Default is 2.
        resnet_edge (bool, optional): Whether to use ResNet for edges. Default is False.
        resnet_node (bool, optional): Whether to use ResNet for nodes. Default is True.
        norm (optional): Normalization method. Default is None.
        additional_message_irreps (optional): Additional message irreps. Default is None.
        weightNetType (str, optional): Type of weight network. Default is 'MLP'.
        KANgrid_size (int, optional): Grid size for KAN. Default is 8.
    Attributes:
        hidden_scalars (int): Number of hidden scalars.
        hidden_dim (int): Dimension of hidden irreps.
        node_embedding_layer (nn.Sequential): Node embedding layers.
        weightNetType (str): Type of weight network.
        MLP_layers (int): Number of MLP layers.
        weight_hidden (int): Number of hidden units in weight network.
        edge_size_irreps (Irreps): Irreps for edge size.
        edge_embedding_layer (O3onO2TensorProductWeighted): Edge embedding layer.
        offsite_irreps (Irreps): Offsite irreps.
        onsite_irreps (Irreps): Onsite irreps.
        neighbour_cutoff (float): Neighbour cutoff distance.
        gauss_width (float): Gaussian width for edge distance.
        layers (nn.ModuleList): List of message passing layers.
        edge_final (O3onO2TensorProductWeighted): Final edge layer.
        node_final_linear1 (O3LinearGated): First final linear layer for nodes.
        node_final_linear2 (Linear): Second final linear layer for nodes.
    Methods:
        forward(graph):
            Perform a forward pass of the network.
    """
    """Steerable E(3) equivariant message passing network projected on O(2) space"""

    def __init__(
        self,
        node_input_irreps,
        hidden_irreps,
        offsite_irreps,
        edge_size_irreps,
        neighbour_cutoff=8.0,
        gauss_width=250.0,
        ConvLayers=7,
        MLP_layers=4,
        weight_hidden=128,
        node_embedding_hidden=32,
        node_embedding_layers=2,
        resnet_edge = False,
        resnet_node = True,
        norm=None,
        additional_message_irreps=None,
        weightNetType = 'MLP',
        KANgrid_size = 8,
    ):
        super().__init__()

        self.hidden_scalars = hidden_irreps[0].mul #assume first irrep is scalar
        self.hidden_dim = hidden_irreps.dim

        node_embedding = []
        node_embedding.append(nn.Linear(node_input_irreps.dim, node_embedding_hidden)) #TODO check if node_input_irreps is all scalar???
        node_embedding.append(nn.ReLU())
        for _ in range(node_embedding_layers-1):
            node_embedding.append(nn.Linear(node_embedding_hidden, node_embedding_hidden))
            node_embedding.append(nn.ReLU())
        node_embedding.append(nn.Linear(node_embedding_hidden, self.hidden_scalars))

        self.node_embedding_layer = nn.Sequential(*node_embedding)
        total_params = sum(p.numel() for p in self.node_embedding_layer.parameters())
        print(f"Total number of weights: {total_params}")

        self.weightNetType = weightNetType
        self.MLP_layers = MLP_layers
        self.weight_hidden = weight_hidden

        if self.weightNetType in ['BSRBF-KAN','FAST-KAN','EfficientKAN']:
            self.edge_size_irreps = Irreps("1x0e")
        else:
            assert self.weightNetType == "MLP", "weightNetType was not MLP or valid KAN (e.g. FAST-KAN) "
            self.edge_size_irreps = edge_size_irreps
        doublehidden_irreps  = (hidden_irreps+hidden_irreps)

        #starting embedding (no weight)
        self.edge_embedding_layer = O3onO2TensorProductWeighted(
            doublehidden_irreps , hidden_irreps
        )       

        #on-site can be simplified by the selection rules, but not now
        self.offsite_irreps = offsite_irreps
        self.onsite_irreps = offsite_irreps

        self.neighbour_cutoff = neighbour_cutoff  # Initial value, you can adjust
        self.gauss_width = gauss_width
        
        # Message passing layers.
        layers = []
        for i in range(ConvLayers):
            layers.append(
                SEGNNLayeronO2(
                    hidden_irreps,
                    self.edge_size_irreps,
                    MLP_layers=MLP_layers,
                    weight_hidden=weight_hidden,
                    weightNetType = weightNetType,
                    norm=norm,
                    resnet_edge=resnet_edge,
                    resnet_node=resnet_node,
                    additional_message_irreps=additional_message_irreps,
                    KANgrid_range=[-0.5, neighbour_cutoff+0.5],
                    KANgrid_size = KANgrid_size
                )
            )
        self.layers = nn.ModuleList(layers)


        self.edge_final = O3onO2TensorProductWeighted(
            hidden_irreps, self.offsite_irreps,irreps_weight=self.edge_size_irreps,
            weight_MLPlayers=MLP_layers,weight_hidden=weight_hidden,
            weightNetType = self.weightNetType, KANgrid_range=[-0.5, neighbour_cutoff+0.5]
        )
        
        #equivariance should come from Graph convolution...
        #Now linear
        #TODO Should use non-linearities?
        self.node_final_linear1 = O3LinearGated( #O3LinearGated
            hidden_irreps, hidden_irreps
        )
        self.node_final_linear2 = Linear(
            hidden_irreps, self.onsite_irreps,shared_weights=True,biases=False,
        )
        
        total_params = sum(p.numel() for p in self.node_final_linear1.parameters())
        print(f"Total number of weights node_final_linear1: {total_params}")
        total_params = sum(p.numel() for p in self.node_final_linear2.parameters())
        print(f"Total number of weights node_final_linear2: {total_params}")
              
   
    def forward(self, graph):
        """SEGNN forward pass"""
        element, edge_index, selfenergy,rotD, batch = (
            graph.elementonehot,
            graph.edge_index,
            graph.selfenergy,
            graph.D,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None          



        #initialize x (scalars) with element descriptor
        x = torch.zeros((element.size(0),self.hidden_dim),dtype=torch.float32,device=edge_index.device) #TODO dtype, device
        x[:,:self.hidden_scalars] = self.node_embedding_layer(element)
        
        if self.weightNetType in ['BSRBF-KAN','FAST-KAN','EfficientKAN']:
            edge_dist_gauss = additional_message_features.clone()
        else:
            mu = torch.linspace(0, self.neighbour_cutoff, self.edge_size_irreps.dim,device=x.device).unsqueeze(0)
            edge_dist_gauss = torch.exp(-self.gauss_width * (additional_message_features - mu) ** 2)
        
        #print(f"x1 = {x}")
        
        #initialize edges
        edge = self.edge_embedding_layer(torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1),rotD,edge_dist_gauss) #Din=2*hidden, Dout=hidden

        #print(f"edge1 = {edge}")

        # Pass messages
        for layer in self.layers:
            x,edge = layer(
                x,edge, edge_index,rotD, batch, edge_dist_gauss, additional_message_features
            )

        #print(f"x2 = {x}")
        #print(f"edge2 = {edge}")

        #Pooling
        edge = self.edge_final(edge,rotD,edge_dist_gauss) #Din=hidden, Dout=hamiltonian
        


        # Cannot use node_attr because there is no Din, Dout
        #Here only option is to use linear + gate(maybe norm-activation) + linear

        #x = self.node_final_linear1(x)
        x = self.node_final_linear2(x)  

        for i,bool in enumerate(selfenergy):
            if bool.item():
                edge[i] = x[edge_index[0,i]]    
           
        return edge
           

class SEGNNLayeronO2(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        hidden_irreps,
        edge_size_irreps,
        weightNetType = "MLP",
        MLP_layers=2,
        weight_hidden=64,
        norm=None,
        resnet_edge = True,
        resnet_node = True,
        additional_message_irreps=None,
        KANgrid_range = [-0.5,8.5],
        KANgrid_size = 8,
    ):
        super().__init__(node_dim=-2, aggr="add") #,flow="target_to_source"
        self.hidden_irreps = hidden_irreps
        
        self.resnet_edge = resnet_edge,
        self.resnet_node = resnet_node,

        #Do not simplify (order is relevant)
        doublehidden_irreps = (hidden_irreps+hidden_irreps)  
        message_input_irreps = (additional_message_irreps + hidden_irreps + hidden_irreps + hidden_irreps)

        self.message_layer_1 = O3onO2TensorProductSwishGateWeighted(
            message_input_irreps, hidden_irreps, trasformationO2='in'
        )
        self.message_layer_2 = O3onO2TensorProductSwishGateWeighted(
            hidden_irreps, hidden_irreps, trasformationO2='out'
        )

        self.node_update_layer1 = Linear(   #TODO add biases?
            doublehidden_irreps,hidden_irreps
        )

        self.node_update_layer2 = Linear( #
            hidden_irreps,hidden_irreps
        )
        
        self.edge_update_layer_1 = O3onO2TensorProductSwishGateWeighted(
            doublehidden_irreps, hidden_irreps,irreps_weight=edge_size_irreps, trasformationO2='in', 
            weightNetType = weightNetType,
            weight_MLPlayers=MLP_layers,weight_hidden=weight_hidden,
            KANgrid_range=KANgrid_range,KANgrid_size=KANgrid_size
        )
        self.edge_update_layer_2 = O3onO2TensorProductSwishGateWeighted(
            hidden_irreps, hidden_irreps,irreps_weight=edge_size_irreps, trasformationO2='out',
            weightNetType = weightNetType,
            weight_MLPlayers=MLP_layers,weight_hidden=weight_hidden,
            KANgrid_range=KANgrid_range,KANgrid_size=KANgrid_size
        )


        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            #self.feature_norm = InstanceNorm(self.hidden_irreps)
            self.feature_norm = BatchNorm(self.hidden_irreps, instance=True)
            self.message_norm = BatchNorm(self.hidden_irreps, instance=True)

    def forward(
        self,
        x,
        edge,
        edge_index,
        rotD,
        batch,
        edge_dist_gauss,
        additional_message_features
    ):
        """Propagate messages along edges and update nodes"""
        x = self.propagate(
            edge_index,
            x=x,
            edge=edge,
            rotD=rotD,
            edge_dist_gauss=edge_dist_gauss,
            additional_message_features=additional_message_features
        )
        
        
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x)
        
                
        """update edges"""
        edge = self.edge_updater(
            edge_index,
            x=x,
            edge=edge,
            rotD=rotD,
            edge_dist_gauss=edge_dist_gauss
        )
        
        
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                edge = self.feature_norm(edge)
            elif self.norm == "instance":
                edge = self.feature_norm(edge)
        


        return x, edge

    def message(self, x_i, x_j,rotD,edge_dist_gauss, edge, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j,edge), dim=-1)
        else:
            input = torch.cat((additional_message_features, x_i, x_j, edge ), dim=-1)
        message = self.message_layer_1(input,rotD) #Here the weight can include elements since x was already updated, think about...
        message = self.message_layer_2(message,rotD)

        if self.message_norm:
            message = self.message_norm(message)

        # I can include another non-linearity here but is supose to be non-linear already
        return message

    def update(self, message, x):
        """Update node features"""
        input = torch.cat((x, message), dim=-1)
        update = self.node_update_layer1(input)
        update = self.node_update_layer2(update)
        if self.resnet_node: 
            x += update  # Residual connection
        else:
            x = update
        return x
    
    def edge_update(self, x_i, x_j,rotD,edge,edge_dist_gauss):
        """Update edge features"""
        #here I have the vectorial information
        input = torch.cat((x_i, x_j), dim=-1)
        update = self.edge_update_layer_1(input,rotD,edge_dist_gauss)
        update = self.edge_update_layer_2(update,rotD,edge_dist_gauss)
        if self.resnet_edge: 
            edge += update  # Residual connection
        else:
            edge = update
        return edge
