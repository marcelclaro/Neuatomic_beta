import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps
from .tensorproducts import O3TensorProductWeighted, O3TensorProductSwishGateWeighted
from e3nn.math import soft_one_hot_linspace

"""
Copyright (c) 2024 Marcel S. Claro

GNU Lesser General Public License v3.0
"""

"""
Based on https://github.com/RobDHess/Steerable-E3-GNN
Fully modified version where edges values can be also trained
"""


class edgeSEGNN(nn.Module):
    """
    Steerable E(3) equivariant message passing network.
    Parameters
    ----------
    node_input_irreps : Irreps
        Irreducible representations for node input features.
    hidden_irreps : Irreps
        Irreducible representations for hidden layers.
    offsite_irreps : Irreps
        Irreducible representations for offsite interactions.
    edge_attr_irreps : Irreps
        Irreducible representations for edge attributes.
    node_attr_irreps : Irreps
        Irreducible representations for node attributes.
    edge_size_irreps : Irreps
        Irreducible representations for edge sizes.
    neighbour_cutoff : float, optional
        Cutoff distance for considering neighbors (default is 8.0).
    gauss_width : float, optional
        Width of the Gaussian for distance encoding (default is 250.0).
    ConvLayers : int, optional
        Number of convolutional layers (default is 4).
    MLP_layers : int, optional
        Number of layers in the MLP (default is 2).
    weight_hidden : int, optional
        Number of hidden units in the weight network (default is 128).
    norm : optional
        Normalization method (default is None).
    additional_message_irreps : optional
        Additional irreducible representations for message passing (default is None).
    resnet_node : bool, optional
        Whether to use ResNet connections for nodes (default is True).
    resnet_edge : bool, optional
        Whether to use ResNet connections for edges (default is True).
    weightNetType : str, optional
        Type of weight network to use (default is 'MLP').
    KANgrid_size : int, optional
        Size of the KAN grid (default is 8).
    outputType : str, optional
        Type of output, either 'edge' or 'node' (default is 'edge').
    distance_encoding : str, optional
        Method for distance encoding (default is 'gaussian_custom').
    distance_adaptive : bool, optional
        Whether the distance encoding is adaptive (default is False).
    Methods
    -------
    adaptive(on=True)
        Enable or disable adaptive Gaussian width.
    forward(graph)
        Perform a forward pass through the network.
    """    

    def __init__(
        self,
        node_input_irreps,
        hidden_irreps,
        edgeoutput_irreps,
        nodeoutput_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        edge_size_irreps,
        neighbour_cutoff=8.0,
        gauss_width=250.0,
        ConvLayers=4,
        MLP_layers=2,
        weight_hidden=128,
        norm=None,
        additional_message_irreps=None,
        resnet_node = True,
        resnet_edge = True,
        weightNetType = 'MLP',
        KANgrid_size = 8,
        outputType = 'edge2',
        distance_encoding = [('gaussian_custom',112),('bessel',16)],
        distance_adaptive = False,
    ):
        super().__init__()

        assert outputType in ['edge1','edge2','node','edge1_plus_node','edge2_plus_node'], "outputType must be either 'edge1(2)' or 'node' or 'edge1(2)_plus_node'"
        self.outputType = outputType


        self.node_embedding_layer = O3TensorProductWeighted(
            node_input_irreps, hidden_irreps, node_attr_irreps, tp_rescale=False
        )

        self.MLP_layers = MLP_layers
        self.weight_hidden = weight_hidden
        
        self.weightNetType = weightNetType

        if self.weightNetType in ['BSRBF-KAN','FAST-KAN','EfficientKAN']:
            self.edge_size_irreps = Irreps("1x0e")
        else:
            assert self.weightNetType == "MLP", "weightNetType was not MLP or valid KAN (e.g. FAST-KAN) "
            self.edge_size_irreps = edge_size_irreps
        edge_input_irreps = (self.edge_size_irreps+2*hidden_irreps)

        self.edge_embedding_layer = O3TensorProductWeighted(
            edge_input_irreps, hidden_irreps, edge_attr_irreps, tp_rescale=False
        )       


        self.edgeoutput_irreps = edgeoutput_irreps
        self.nodeoutput_irreps = nodeoutput_irreps

        if outputType == 'edge1' or outputType == 'edge2':
            self.totalnodeoutput_irreps = edgeoutput_irreps
        elif outputType == 'node':
            self.totalnodeoutput_irreps = nodeoutput_irreps
        elif outputType == 'edge1_plus_node':
            self.totalnodeoutput_irreps = nodeoutput_irreps
        elif outputType == 'edge2_plus_node':
            self.totalnodeoutput_irreps = edgeoutput_irreps+nodeoutput_irreps        
        else:
            raise ValueError("outputType must be either 'edge' or 'node' or 'edge_plus_node'")

        self.neighbour_cutoff = neighbour_cutoff  # Initial value, you can adjust
        self.gauss_width = gauss_width
        self.distance_encoding = distance_encoding

        #backwards compatibility
        if isinstance(self.distance_encoding, str):
            self.distance_encoding = [(self.distance_encoding, self.edge_size_irreps.dim)]

        assert sum([d[1] for d in self.distance_encoding]) == self.edge_size_irreps.dim, "The sum of the distance_encoding_base must be equal to the edge_size_irreps.dim"

        # Check if 'gaussian_custom' is in the distance encoding methods
        distance_encoding_methods = [d[0] for d in self.distance_encoding]
        if 'gaussian_custom' in distance_encoding_methods:
            gaussian_custom_index = distance_encoding_methods.index('gaussian_custom')
            gaussian_custom_base = self.distance_encoding[gaussian_custom_index][1]
            self.gaussian_width = torch.full((1,gaussian_custom_base),self.gauss_width,requires_grad=distance_adaptive)
            self.gaussian_mu = torch.linspace(0, self.neighbour_cutoff, gaussian_custom_base,requires_grad=distance_adaptive).unsqueeze(0)
        
        # Message passing layers.
        layers = []
        for i in range(ConvLayers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    self.edge_size_irreps,
                    weightNetType = weightNetType,
                    MLP_layers=MLP_layers,
                    weight_hidden=weight_hidden,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                    resnet_node=resnet_node,
                    resnet_edge=resnet_edge,
                    KANgrid_range=[-0.5, neighbour_cutoff+0.5],
                    KANgrid_size = KANgrid_size
                )
            )
        self.layers = nn.ModuleList(layers)


        if self.outputType == 'edge1' or self.outputType == 'edge2' or self.outputType == 'edge2_plus_node' or self.outputType == 'edge1_plus_node':
            # self.edge_pre_pool1 = O3TensorProductSwishGateWeighted(
            #     hidden_irreps, hidden_irreps, edge_attr_irreps,irreps_weight=self.edge_size_irreps,
            #     weight_MLPlayers=MLP_layers,weight_hidden=weight_hidden,
            #     weightNetType = weightNetType, KANgrid_range=[-0.5, neighbour_cutoff+0.5],KANgrid_size=KANgrid_size
            # )
            self.edge_pre_pool2 = O3TensorProductWeighted(
                hidden_irreps, self.edgeoutput_irreps, edge_attr_irreps,irreps_weight=self.edge_size_irreps,
                weightNetType = weightNetType, weight_hidden=weight_hidden,
                weight_MLPlayers=MLP_layers,KANgrid_range=[-0.5, neighbour_cutoff+0.5],KANgrid_size=KANgrid_size
            )
        
        if self.outputType == 'node' or self.outputType == 'edge2' or self.outputType == 'edge2_plus_node' or self.outputType == 'edge1_plus_node':

            # self.node_pre_pool1 = O3TensorProductSwishGateWeighted(
            #     hidden_irreps, hidden_irreps, node_attr_irreps
            # )
            self.node_pre_pool2 = O3TensorProductWeighted(
                hidden_irreps, self.totalnodeoutput_irreps , node_attr_irreps
            )
              
    def adaptive(self,on = True):
        if 'gaussian_custom' in [d[0] for d in self.distance_encoding]:
            self.gaussian_width.requires_grad_(on) 
            self.gaussian_mu.requires_grad_(on)


    def forward(self, graph):
        """SEGNN forward pass"""
        x, pos, edge_index,edge_vec, edge_attr, node_attr,selfenergy, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_vec,
            graph.edge_attr,
            graph.node_attr,
            graph.selfenergy,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

            
        if self.weightNetType in ['BSRBF-KAN','FAST-KAN','EfficientKAN']:
            edge_dist_gauss = additional_message_features
        else:
            base_lst = []
            for base in self.distance_encoding:
                if base[0] == 'gaussian_custom':
                    self.gaussian_width = self.gaussian_width.to(device=additional_message_features.device)
                    self.gaussian_mu = self.gaussian_mu.to(device=additional_message_features.device)
                    base_lst.append(torch.exp(-self.gaussian_width * (additional_message_features - self.gaussian_mu) ** 2))
                else:
                    base_lst.append(soft_one_hot_linspace(additional_message_features,-0.001,self.neighbour_cutoff,base[1],base[0],cutoff=False).squeeze(1))
            edge_dist_gauss = torch.cat(base_lst,dim=1)

        node = self.node_embedding_layer(x, node_attr)
        edge = self.edge_embedding_layer(torch.cat((edge_dist_gauss,node[edge_index[0]], node[edge_index[1]]), dim=1),edge_attr)

       
        # Pass messages
        for layer in self.layers:
            node,edge = layer(
                node,edge, edge_index, edge_attr, node_attr, batch, additional_message_features, edge_dist_gauss
            )
        

        if self.outputType == 'edge1' or self.outputType == 'edge2':
            # Pre pool
            #edge = self.edge_pre_pool1(edge, edge_attr,edge_dist_gauss)
            edge = self.edge_pre_pool2(edge, edge_attr,edge_dist_gauss)
            
            if self.outputType == 'edge2':
                #node = self.node_pre_pool1(node, node_attr)
                node = self.node_pre_pool2(node, node_attr)
                for i,bool in enumerate(selfenergy):
                    if bool.item():
                        edge[i] = node[edge_index[0,i]]    
            return edge
        elif self.outputType == 'node':
            #node = self.node_pre_pool1(node, node_attr)
            node = self.node_pre_pool2(node, node_attr)
            return node
        elif self.outputType == 'edge1_plus_node':
            # Pre pool
            #edge = self.edge_pre_pool1(edge, edge_attr,edge_dist_gauss)
            edge = self.edge_pre_pool2(edge, edge_attr,edge_dist_gauss)
            #node = self.node_pre_pool1(node, node_attr)
            node = self.node_pre_pool2(node, node_attr)
            return edge,node
        elif self.outputType == 'edge2_plus_node':
            # Pre pool
            #edge = self.edge_pre_pool1(edge, edge_attr,edge_dist_gauss)
            edge = self.edge_pre_pool2(edge, edge_attr,edge_dist_gauss)
            #node = self.node_pre_pool1(node, node_attr)
            node = self.node_pre_pool2(node, node_attr)
            
            node_edgepart, node_selfpart = torch.split(node, [self.edgeoutput_irreps.dim, self.nodeoutput_irreps.dim], dim=1)
            
            for i,bool in enumerate(selfenergy):
                if bool.item():
                    edge[i] = node_edgepart[edge_index[0,i]]    
            
            return edge, node_selfpart  
           

class SEGNNLayer(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        edge_size_irreps,
        weightNetType = "MLP",
        MLP_layers=2,
        weight_hidden=64,
        norm=None,
        additional_message_irreps=None,
        resnet_node = True,
        resnet_edge = True,
        KANgrid_range = [-0.5,8.5],
        KANgrid_size = 8,
    ):
        super().__init__(node_dim=-2, aggr="add") #,flow="target_to_source"
        self.hidden_irreps = hidden_irreps
        self.resnet_node = resnet_node
        self.resnet_edge = resnet_edge

        #Do not simplify (order is relevant)
        message_input_irreps = (additional_message_irreps + 2 * input_irreps + hidden_irreps)
        update_input_irreps = (input_irreps + hidden_irreps)
        edge_update_input_irreps = (2 * input_irreps)

        self.message_layer_1 = O3TensorProductSwishGateWeighted(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGateWeighted(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.node_update_layer_1 = O3TensorProductSwishGateWeighted(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.node_update_layer_2 = O3TensorProductWeighted(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )
        
        self.edge_update_layer_1 = O3TensorProductSwishGateWeighted(
            edge_update_input_irreps, hidden_irreps, edge_attr_irreps,irreps_weight=edge_size_irreps,
            weightNetType = weightNetType,
            weight_MLPlayers=MLP_layers,weight_hidden=weight_hidden,
            KANgrid_range=KANgrid_range,KANgrid_size=KANgrid_size
        )
        self.edge_update_layer_2 = O3TensorProductSwishGateWeighted(
            hidden_irreps, hidden_irreps, edge_attr_irreps,irreps_weight=edge_size_irreps,
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
        edge_attr,
        node_attr,
        batch,
        additional_message_features,
        edge_dist_gauss
    ):
        """Propagate messages along edges and update nodes"""
        x = self.propagate(
            edge_index,
            x=x,
            edge=edge,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
            edge_dist_gauss=edge_dist_gauss
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
            node_attr=node_attr,
            edge_attr=edge_attr,
            edge_dist_gauss=edge_dist_gauss
        )
        
        
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                edge = self.feature_norm(edge)
            elif self.norm == "instance":
                edge = self.feature_norm(edge)
        


        return x, edge

    def message(self, x_i, x_j, edge_attr , edge, additional_message_features,edge_dist_gauss):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j,edge), dim=-1)
        else:
            input = torch.cat((additional_message_features, x_i, x_j, edge ), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update node features"""
        input = torch.cat((x, message), dim=-1)
        update = self.node_update_layer_1(input, node_attr)
        update = self.node_update_layer_2(update, node_attr)
        if self.resnet_node:
            x += update  # Residual connection
        else:
            x = update
        return x
    
    def edge_update(self, x_i, x_j, edge, edge_attr,edge_dist_gauss):
        """Update edge features"""
        input = torch.cat((x_i, x_j), dim=-1)
        update = self.edge_update_layer_1(input, edge_attr,edge_dist_gauss)
        update = self.edge_update_layer_2(update, edge_attr,edge_dist_gauss)
        if self.resnet_edge:
            edge += update  # Residual connection
        else:
            edge = update
        return edge
