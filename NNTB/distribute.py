import numpy as np
import torch
import torch_geometric
import copy

def subgraph_w_edge_mask(fullset, subset):
    r"""Returns the induced subgraph given by the node indices
    :obj:`subset`.

    Args:
        subset (LongTensor or BoolTensor): The nodes to keep.
    """
    if 'edge_index' in fullset:
        edge_index, _, edge_mask = torch_geometric.utils.subgraph(
            subset,
            fullset.edge_index,
            relabel_nodes=True,
            num_nodes=fullset.num_nodes,
            return_edge_mask=True,
        )
    else:
        edge_index = None
        edge_mask = torch.ones(
            fullset.num_edges,
            dtype=torch.bool,
            device=subset.device,
        )

    data = copy.copy(fullset)

    for key, value in fullset:
        if key == 'edge_index':
            data.edge_index = edge_index
        elif key == 'num_nodes':
            if subset.dtype == torch.bool:
                data.num_nodes = int(subset.sum())
            else:
                data.num_nodes = subset.size(0)
        elif fullset.is_node_attr(key):
            cat_dim = fullset.__cat_dim__(key, value)
            data[key] = torch_geometric.utils.select(value, subset, dim=cat_dim)
        elif fullset.is_edge_attr(key):
            cat_dim = fullset.__cat_dim__(key, value)
            data[key] = torch_geometric.utils.select(value, edge_mask, dim=cat_dim)

    return data, edge_mask


def evaluate(system, model, data):
    # Parameters for grid partitioning
    skin = system.neighbour_cutoff + 0.001  # Skin thickness
    voxel_size = system.max_domain_size  # Size of each 3D region

    # Extract node coordinates and edges
    coordinates = data.pos  # Node positions

    # Compute voxel grid
    grid_indices = (coordinates  // voxel_size)

    # Create subgraphs and edge lists
    subgraphs = []
    unique_voxels = torch.unique(grid_indices, dim=0)
    edge_maps = []

    for voxel in unique_voxels:
        # Get nodes within the voxel, including ghost nodes
        mask = torch.all(
            (coordinates >= voxel * voxel_size - skin) &
            (coordinates < (voxel + 1) * voxel_size + skin),
            axis=1
        )
        nodes = torch.where(mask)[0]

        # Convert to PyTorch Geometric Data format
        subgraph, edge_mask = subgraph_w_edge_mask(data,nodes)
        subgraph.original_edge_indices = edge_mask.nonzero(as_tuple=False).squeeze()
        subgraphs.append(subgraph)

        
    # Evaluate model on each subgraph
    subgraph_results = []
    for subgraph in subgraphs:
        # Ensure subgraph is on the appropriate device
        #subgraph = subgraph.to(model.device)
        with torch.no_grad():
            subgraph_results.append(model(subgraph))

    #Sequential version:
    # all_values = torch.cat([subgraph.original_edge_indices for subgraph in subgraphs]) # Concatenate all vectors into a single tensor
    # unique_values, counts = torch.unique(all_values, return_counts=True) # Identify unique values and their counts
    # repeated_edges = unique_values[counts > 1] # Find repeated values (counts > 1)

    # # Compile results in the order of the original graph edges
    # compiled_results = torch.zeros((data.edge_index.size(1),subgraph_results[0].size(1)), dtype=subgraph_results[0].dtype)
    # for subgraph, result in zip(subgraphs,subgraph_results):
    #     for n, edge_idx in enumerate( subgraph.original_edge_indices):
    #         if edge_idx in repeated_edges:
    #             compiled_results[edge_idx] += result[n]
    #         else:
    #             compiled_results[edge_idx] = result[n]

    # # Average the results for repeated edges
    # for n, edge_idx in enumerate(repeated_edges):
    #     compiled_results[edge_idx] /= counts[n]

    # Step 1: Concatenate all edge indices and identify unique values and their counts
    all_values = torch.cat([subgraph.original_edge_indices for subgraph in subgraphs])  # [N]
    unique_values, counts = torch.unique(all_values, return_counts=True)  # [U]

    # Step 2: Initialize the compiled results tensor
    compiled_results = torch.zeros(
        (data.edge_index.size(1), subgraph_results[0].size(1)), 
        dtype=subgraph_results[0].dtype
    )  # [E, F]

    # Step 3: Accumulate results for all subgraphs
    # Flatten subgraph results into a single tensor corresponding to all_values
    flattened_results = torch.cat(subgraph_results, dim=0)  # [N, F]

    # Use scatter_add to efficiently accumulate results for each edge index
    compiled_results.index_add_(0, all_values, flattened_results)

    # Step 4: Normalize repeated edges by their counts
    repeated_mask = counts > 1  # [U]
    repeated_edges = unique_values[repeated_mask]  # [R]
    repeated_counts = counts[repeated_mask].float().unsqueeze(1)  # [R, 1]

    # Scatter divide to average results for repeated edges
    compiled_results[repeated_edges] /= repeated_counts

    return compiled_results