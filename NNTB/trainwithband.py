from .util import *
from ase.data import atomic_numbers
import torch
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
from torch.utils.tensorboard import SummaryWriter
import datetime

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'band':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

#New version of training that account for Hamiltonian group transformations
def trainModel(system,model,data_lst = None, datafolder = None,validation_folder = None,
                    epochs = 500, learn_rate = 0.005,amsgrad=True, weight_decay=0e-5, device = 'cpu',
                    scheduler_patience = 30, scheduler_factor = 0.8, optm_multiplier = 5,
                    batchsize = 1, runname = 'run',SymmGauge = True, EnergyGauge = True , modeltype = 'Hamiltonian',
                    validation_batch = 5,
                    bandloss = True, bandloss_beta = 0.005, bandloss_patience = 400, bandloss_max_lr = 0.001, soc_beta = 0.05,
                    adaptive_patience = -1, verbose = True, gradientlog = False,
                    distributed = False,rank = 0,checkpointpath=  '"./model.checkpoint"' ):
    """
    Trains a neural network model for a given system using specified parameters.
    Args:
        system: The system object containing necessary configurations and methods.
        model: The neural network model to be trained.
        data_lst (list, optional): List of training data graphs. Defaults to None.
        datafolder (str, optional): Path to the folder containing training data files. Defaults to None.
        validation_folder (str, optional): Path to the folder containing validation data files. Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 500.
        learn_rate (float, optional): Learning rate for the optimizer. Defaults to 0.005.
        amsgrad (bool, optional): Whether to use the AMSGrad variant of Adam. Defaults to True.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 0e-5.
        device (str, optional): Device to use for training ('cpu' or 'cuda'). Defaults to 'cpu'.
        scheduler_patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced. Defaults to 30.
        scheduler_factor (float, optional): Factor by which the learning rate will be reduced. Defaults to 0.8.
        optm_multiplier (int, optional): Multiplier for optimization steps. Defaults to 5.
        batchsize (int, optional): Batch size for training. Defaults to 1.
        runname (str, optional): Name for the training run. Defaults to 'run'.
        SymmGauge (bool, optional): Whether to use symmetry gauge transformations. Defaults to True.
        EnergyGauge (bool, optional): Whether to use energy gauge transformations. Defaults to True.
        modeltype (str, optional): Type of model ('Hamiltonian', 'Overlap', 'Force', 'SOC'). Defaults to 'Hamiltonian'.
        asymmetry_alpha (float, optional): Weight for asymmetry loss (higher keeps the symmetry). Defaults to 1e-3.
        asymmetry_patience (int, optional): Number of epochs to apply asymmetry loss. Defaults to 1000.
        validation_batch (int, optional): Number of validation samples per validation step. Defaults to 5.
        bandloss (bool, optional): Whether to use band loss. Defaults to True.
        bandloss_beta (float, optional): Weight for band loss. Defaults to 0.005.
        bandloss_patience (int, optional): Number of epochs before applying band loss. Defaults to 400.
        bandloss_max_lr (float, optional): Maximum learning rate for applying band loss. Defaults to 0.001.
        soc_beta (float, optional): Weight for SOC loss. Defaults to 0.05.
        adaptive_patience (int, optional): Number of epochs before activating adaptive distance one-hot. Defaults to -1.
        verbose (bool, optional): Whether to print detailed logs. Defaults to True.
        gradientlog (bool, optional): Whether to log gradients to TensorBoard. Defaults to False.
        distributed (bool, optional): Whether to use distributed training. Defaults to False.
        rank (int, optional): Rank of the current process in distributed training. Defaults to 0.
        checkpointpath (str, optional): Path to save the model checkpoint. Defaults to './model.checkpoint'.
    Raises:
        AssertionError: If batch size is less than 1.
        Exception: If an invalid model type is specified.
    Returns:
        None
    """

    if distributed:
        #TODO Manage memory here
        print("CAUTION: Distributed case not implemented yet!")
        pass
    else:
        #Set up device for training
        if device == 'cuda':
            system.device_train = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            system.device_train = 'cpu'
            #Move model to device
        model = model.to(system.device_train)

    #Starting Tensorboard to track training
    log_dir = 'runs'
    run_name = runname + datetime.datetime.now().strftime("%d-%m_%H-%M")
    writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

    #Start an optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=learn_rate,amsgrad=amsgrad, weight_decay=weight_decay)
    
    #Loss function TODO: Any option here?
    lossfunc = torch.nn.MSELoss()
    asymm_lossfunc = torch.nn.MSELoss()
    band_lossfunc = torch.nn.MSELoss()
    

    #Kpoints list for band calculation
    band_kpoints = system.band_kpoints.to(device=system.device_train)

    assert batchsize >= 1, "Batch size should be >= 1" #Check batchsize
    #assert (data_lst is not None) and (datafolder is not None), "Both data_lst and datafolder were specified, choose one"
    #assert (data_lst is None) and (datafolder is None), "No data specified"

    #If no data list is provide
    if data_lst == None:
        assert datafolder != None, "No data_lst or data_folder specified"
        data_lst = []

    print("Loading files...")
    #Load all .ptg and .tg in the folder
    if datafolder:
        file_names = [file for file in os.listdir(datafolder) if file.endswith('.ptg') or file.endswith('.tg')]
        for file_name in file_names:
            graph = torch.load(os.path.join(datafolder, file_name))
            if not isinstance(graph, MyData):
                graph = MyData.from_dict(graph)
            data_lst.append(graph)

    # Load validation data if validation folder is specified
    validation_data_lst = []
    if validation_folder:
        validation_file_names = [file for file in os.listdir(validation_folder) if file.endswith('.ptg') or file.endswith('.tg')]
        for file_name in validation_file_names:
            graph = torch.load(os.path.join(validation_folder, file_name))
            if not isinstance(graph, MyData):
                graph = MyData.from_dict(graph)
            validation_data_lst.append(graph)

        
    #Process graphs if they are not fully processed
    print(f"Processing train graphs... Total={len(data_lst)}")
    for i,graph in enumerate(data_lst):
        print(f"File: {i+1}:", end=' ')
        if graph.preprocessed:
            print("Already processed") if verbose else ""
        else:
            print("Processing") if verbose else ""
            system.processGraph(graph)
    print(f"Processing validation graphs... Total={len(validation_data_lst)}")
    for i,graph in enumerate(validation_data_lst):
        print(f"File: {i+1}:", end=' ')
        if graph.preprocessed:
            print("Already processed") if verbose else ""
        else:
            print("Processing") if verbose else ""
            system.processGraph(graph)
    
    print("Start training...")

    # Early stopping parameters
    #TODO: parameter in the call
    early_stopping_patience = 1000
    early_stopping_counter = 0
    best_val_loss = float('inf')
    

    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)



    #The epoch loop
    for epoch in range(epochs):
        model.train()     #set model to train     
        optimizer.zero_grad() #zero gradient
        if batchsize > 1:  #if batchsize more than 1
            if batchsize == len(data_lst):
                random_indices = torch.arange(0,batchsize).tolist() #If batch size equal to data gets a random order of all data
            else:
                random_indices = [0]+torch.randint(0, len(data_lst), (batchsize-1,), dtype=torch.long).tolist() #otherwise [0] will be always the reference for group trnasformations

            data = Batch.from_data_list([data_lst[i] for i in random_indices])  # creates a batch
        else:
            r_num = random.randint(0, len(data_lst)-1)  #if not minibatchs gets a random data item and print the name
            data = data_lst[r_num]
            print(f"Crystal:  {data.name}") if verbose else ""          
        data.to(system.device_train) #move data to device

        #activate adaptive distance one-hot
        if epoch == adaptive_patience:
            model.adaptive(True)

        if modeltype == 'HamiltonianSOC':
            out, out_soc = model(data) # evaluates model
        else:
            out = model(data)

        """
        In a periodic crystal the Hamiltonian can be transformed acording to the crystal symmetry group g, so that H' != gH but the result still valid (i.e. bands and properties are the same)
        So, for the training is necessary to consider the target H is in an arbitrary symmetry gauge g.
        The same for base energy, DFT calculations can have different E=0 level...
        """
        if SymmGauge and batchsize == 1 and modeltype != 'SOC':  #If using hamiltonian group invariance
            #TODO for SOC transformation is never necessary as SOC are invariant
            minimum = (0,float('inf'))
            with torch.no_grad():
                for id,group in enumerate(data.group_transform):
                    if modeltype == 'Hamiltonian' or modeltype == 'Overlap' or modeltype == 'HamiltonianSOC':
                        Dmat = system.hamiltonian_irreps.D_from_matrix(torch.Tensor(group)).to(system.device_train)
                    elif modeltype == 'Force':
                        Dmat = Irreps("1x1o").D_from_matrix(torch.Tensor(group)).to(system.device_train)
                    #depending if is for hopping or overlap models
                    if modeltype == 'Hamiltonian':
                        temphopping = torch.matmul(Dmat,data.hopping_irrep.t()).squeeze(0).t()
                        if EnergyGauge:
                            tempoverlap = torch.matmul(Dmat,data.overlap_irrep.t()).squeeze(0).t()
                            mu = torch.sum( (out - temphopping) * tempoverlap ) / torch.sum(tempoverlap * tempoverlap)
                            losstemp = lossfunc(out,temphopping + mu*tempoverlap)
                        else:
                            losstemp = lossfunc(out,temphopping)
                    elif modeltype == 'HamiltonianSOC':
                        temphopping = torch.matmul(Dmat,data.hopping_irrep.t()).squeeze(0).t()
                        if EnergyGauge:
                            tempoverlap = torch.matmul(Dmat,data.overlap_irrep.t()).squeeze(0).t()
                            mu = torch.sum( (out - temphopping) * tempoverlap ) / torch.sum(tempoverlap * tempoverlap)
                            losstemp = lossfunc(out,temphopping + mu*tempoverlap)
                        else:
                            losstemp = lossfunc(out,temphopping)
                    elif modeltype == 'Overlap':
                        tempoverlap = torch.matmul(Dmat,data.overlap_irrep.t()).squeeze(0).t()
                        losstemp = lossfunc(out,tempoverlap)
                    elif modeltype == 'Force':
                        tempforces = torch.matmul(Dmat,data.forces.t()).squeeze(0).t()
                        losstemp = lossfunc(out,tempforces)
                    else:
                        raise Exception("Modeltype invalid")
                    if losstemp < minimum[1]:
                        minimum = (id,losstemp)
            
            if data.group_transform[minimum[0]] != torch.eye(3,dtype=torch.int64):
                transformeddata = data.clone()  #TODO not necessary if not transfomed, same for the SOC case(invariant). Optmize!
                print(f"Graph transformed by {minimum[0]}")
                if modeltype == 'Hamiltonian' or modeltype == 'Overlap' or modeltype == 'HamiltonianSOC':
                    Dmat = system.hamiltonian_irreps.D_from_matrix(torch.tensor(transformeddata.group_transform[minimum[0]],dtype=torch.float32)).to(system.device_train)
                elif modeltype == 'Force':
                    Dmat = Irreps("1x1o").D_from_matrix(torch.tensor(transformeddata.group_transform[minimum[0]],dtype=torch.float32)).to(system.device_train)
                
                if modeltype == 'Hamiltonian':
                    transformeddata.hopping_irrep = torch.matmul(Dmat,transformeddata.hopping_irrep.t()).squeeze(0).t()  #Squeeze?                                  
                    if EnergyGauge:
                        transformeddata.overlap_irrep = torch.matmul(Dmat,transformeddata.overlap_irrep.t()).squeeze(0).t()
                        mu = torch.sum( (out - transformeddata.hopping_irrep) * transformeddata.overlap_irrep ) / torch.sum(transformeddata.overlap_irrep * transformeddata.overlap_irrep)
                        loss = lossfunc(out,transformeddata.hopping_irrep + mu*transformeddata.overlap_irrep)
                    else:
                        loss = lossfunc(out, transformeddata.hopping_irrep)
                elif modeltype == 'HamiltonianSOC':
                    transformeddata.hopping_irrep = torch.matmul(Dmat,transformeddata.hopping_irrep.t()).squeeze(0).t()  #Squeeze?                                  
                    if EnergyGauge:
                        transformeddata.overlap_irrep = torch.matmul(Dmat,transformeddata.overlap_irrep.t()).squeeze(0).t()
                        mu = torch.sum( (out - transformeddata.hopping_irrep) * transformeddata.overlap_irrep ) / torch.sum(transformeddata.overlap_irrep * transformeddata.overlap_irrep)
                        loss = lossfunc(out,transformeddata.hopping_irrep + mu*transformeddata.overlap_irrep)
                        loss += soc_beta * lossfunc(out_soc,transformeddata.soc_irrep)
                    else:
                        loss = lossfunc(out, transformeddata.hopping_irrep)
                        loss += soc_beta * lossfunc(out_soc,transformeddata.soc_irrep)
                elif modeltype == 'Overlap':
                    transformeddata.overlap_irrep = torch.matmul(Dmat,transformeddata.overlap_irrep.t()).squeeze(0).t()
                    loss = lossfunc(out, transformeddata.overlap_irrep)
                elif modeltype == 'Force':
                    transformeddata.forces = torch.matmul(Dmat,transformeddata.forces.t()).squeeze(0).t()
                    loss = lossfunc(out, transformeddata.forces)
                else:
                    raise Exception("Modeltype invalid")            
        elif SymmGauge and batchsize > 1 and modeltype != 'SOC':  #If using hamiltonian group invariance
            #TODO for SOC transformation is never necessary as SOC are invariant
            transformations = []
            for index in range(1,data._num_graphs):  #For each graph in the batch (except the first which is the reference) calculate the transformation with the lower loss
                minimum = (index,0,1.0)
                graphslice = data.get_example(index)
                if modeltype == 'HamiltonianSOC':
                    outhop, _ = model(graphslice)
                else:
                    output = model(graphslice)
                with torch.no_grad():
                    for id,group in enumerate(graphslice.group_transform):
                        if modeltype == 'Hamiltonian' or modeltype == 'Overlap':
                            Dmat = system.hamiltonian_irreps.D_from_matrix(torch.Tensor(group)).to(system.device_train)
                        elif modeltype == 'Force':
                            Dmat = Irreps("1x1o").D_from_matrix(torch.Tensor(group)).to(system.device_train)
                        #depending if is for hopping or overlap models
                        if modeltype == 'Hamiltonian':
                            temphopping = torch.matmul(Dmat,graphslice.hopping_irrep.t()).squeeze(0).t()
                            if EnergyGauge:
                                tempoverlap = torch.matmul(Dmat,graphslice.overlap_irrep.t()).squeeze(0).t()
                                mu = torch.sum( (output - temphopping) * tempoverlap ) / torch.sum(tempoverlap * tempoverlap)
                                losstemp = lossfunc(output,temphopping + mu*tempoverlap)
                            else:
                                losstemp = lossfunc(output,temphopping)
                        elif modeltype == 'HamiltonianSOC':
                            temphopping = torch.matmul(Dmat,graphslice.hopping_irrep.t()).squeeze(0).t()
                            if EnergyGauge:
                                tempoverlap = torch.matmul(Dmat,graphslice.overlap_irrep.t()).squeeze(0).t()
                                mu = torch.sum( (outhop - temphopping) * tempoverlap ) / torch.sum(tempoverlap * tempoverlap)
                                losstemp = lossfunc(outhop,temphopping + mu*tempoverlap)
                            else:
                                losstemp = lossfunc(outhop,temphopping)
                        elif modeltype == 'Overlap':
                            tempoverlap = torch.matmul(Dmat,graphslice.overlap_irrep.t()).squeeze(0).t()
                            losstemp = lossfunc(output,tempoverlap)
                        elif modeltype == 'Force':
                            tempforces = torch.matmul(Dmat,graphslice.forces.t()).squeeze(0).t()
                            losstemp = lossfunc(output,tempforces)
                        else:
                            raise Exception("Modeltype invalid")
                        if losstemp < minimum[2]:
                            minimum = (index,id,losstemp)
                transformations.append(minimum)
            transformeddata = data.clone()  #TODO not necessary if not transfomed. Optmize!
            for transf in transformations:
                print(f"Graph:{transf[0]} transformed by {transf[1]}")
                if modeltype == 'Hamiltonian' or modeltype == 'Overlap':
                    Dmat = system.hamiltonian_irreps.D_from_matrix(torch.tensor(transformeddata.group_transform[transf[0]][transf[1]],dtype=torch.float32)).to(system.device_train)
                elif modeltype == 'Force':
                    Dmat = Irreps("1x1o").D_from_matrix(torch.tensor(transformeddata.group_transform[transf[0]][transf[1]],dtype=torch.float32)).to(system.device_train)
                
                if modeltype == 'Hamiltonian':
                    transformeddata.hopping_irrep[transformeddata._slice_dict['hopping_irrep'][transf[0]]:transformeddata._slice_dict['hopping_irrep'][transf[0]+1]] = torch.matmul(Dmat,transformeddata.hopping_irrep[transformeddata._slice_dict['hopping_irrep'][transf[0]]:transformeddata._slice_dict['hopping_irrep'][transf[0]+1]].t()).squeeze(0).t()                                    
                    if EnergyGauge:
                        transformeddata.overlap_irrep[transformeddata._slice_dict['overlap_irrep'][transf[0]]:transformeddata._slice_dict['overlap_irrep'][transf[0]+1]] = torch.matmul(Dmat,transformeddata.overlap_irrep[transformeddata._slice_dict['overlap_irrep'][transf[0]]:transformeddata._slice_dict['overlap_irrep'][transf[0]+1]].t()).squeeze(0).t()
                        mu = torch.sum( (out - transformeddata.hopping_irrep) * transformeddata.overlap_irrep ) / torch.sum(transformeddata.overlap_irrep * transformeddata.overlap_irrep)
                        loss = lossfunc(out,transformeddata.hopping_irrep + mu*transformeddata.overlap_irrep)
                    else:
                        loss = lossfunc(out, transformeddata.hopping_irrep)
                elif modeltype == 'HamiltonianSOC':
                    transformeddata.hopping_irrep[transformeddata._slice_dict['hopping_irrep'][transf[0]]:transformeddata._slice_dict['hopping_irrep'][transf[0]+1]] = torch.matmul(Dmat,transformeddata.hopping_irrep[transformeddata._slice_dict['hopping_irrep'][transf[0]]:transformeddata._slice_dict['hopping_irrep'][transf[0]+1]].t()).squeeze(0).t()                                    
                    if EnergyGauge:
                        transformeddata.overlap_irrep[transformeddata._slice_dict['overlap_irrep'][transf[0]]:transformeddata._slice_dict['overlap_irrep'][transf[0]+1]] = torch.matmul(Dmat,transformeddata.overlap_irrep[transformeddata._slice_dict['overlap_irrep'][transf[0]]:transformeddata._slice_dict['overlap_irrep'][transf[0]+1]].t()).squeeze(0).t()
                        mu = torch.sum( (out - transformeddata.hopping_irrep) * transformeddata.overlap_irrep ) / torch.sum(transformeddata.overlap_irrep * transformeddata.overlap_irrep)
                        loss = lossfunc(out,transformeddata.hopping_irrep + mu*transformeddata.overlap_irrep)
                        loss += soc_beta * lossfunc(out_soc,transformeddata.soc_irrep)
                    else:
                        loss = lossfunc(out, transformeddata.hopping_irrep)
                        loss += soc_beta * lossfunc(out_soc,transformeddata.soc_irrep)
                elif modeltype == 'Overlap':
                    transformeddata.overlap_irrep[transformeddata._slice_dict['overlap_irrep'][transf[0]]:transformeddata._slice_dict['overlap_irrep'][transf[0]+1]] = torch.matmul(Dmat,transformeddata.overlap_irrep[transformeddata._slice_dict['overlap_irrep'][transf[0]]:transformeddata._slice_dict['overlap_irrep'][transf[0]+1]].t()).squeeze(0).t()
                    loss = lossfunc(out, transformeddata.overlap_irrep)
                elif modeltype == 'Force':
                    transformeddata.forces[transformeddata._slice_dict['forces'][transf[0]]:transformeddata._slice_dict['forces'][transf[0]+1]] = torch.matmul(Dmat,transformeddata.overlap_irrep[transformeddata._slice_dict['forces'][transf[0]]:transformeddata._slice_dict['forces'][transf[0]+1]].t()).squeeze(0).t()
                    loss = lossfunc(out, transformeddata.forces)
                else:
                    raise Exception("Modeltype invalid")
        else:
            if modeltype == 'Hamiltonian':
                if EnergyGauge:
                    mu = torch.sum( (out - data.hopping_irrep) * data.overlap_irrep ) / torch.sum(data.overlap_irrep * data.overlap_irrep)
                    losstemp = lossfunc(out,data.hopping_irrep + mu*data.overlap_irrep)
                else:
                    loss = lossfunc(out, data.hopping_irrep)
            elif modeltype == 'HamiltonianSOC':
                if EnergyGauge:
                    mu = torch.sum( (out - data.hopping_irrep) * data.overlap_irrep ) / torch.sum(data.overlap_irrep * data.overlap_irrep)
                    losstemp = lossfunc(out,data.hopping_irrep + mu*data.overlap_irrep)
                    loss += soc_beta * lossfunc(out_soc,data.soc_irrep)
                else:
                    loss = lossfunc(out, data.hopping_irrep)
                    loss += soc_beta * lossfunc(out_soc,data.soc_irrep)
            elif modeltype == 'Overlap':
                loss = lossfunc(out, data.overlap_irrep)
            elif modeltype == 'Force':
                loss = lossfunc(out, data.forces)
            elif modeltype == 'SOC':
                loss = lossfunc(out, data.soc_irrep) #torch.cat((data.soc_irrep,data.hopping_irrep), dim=1)
            else:
                raise Exception("Modeltype invalid")

        if modeltype in ['Hamiltonian','HamiltonianSOC','Overlap','SOC'] and bandloss and bandloss_patience < epoch and bandloss_max_lr >= optimizer.param_groups[0]['lr'] and batchsize == 1:
            if not hasattr(data, 'band'):
                print("Warning: data.band does not exist. Skipping band loss calculation... (Tip: Re-process the graph)")
                continue
            if data.spin:
                hamtype = torch.complex64
            else:
                hamtype = torch.float32
                
            if modeltype == 'Hamiltonian':
                if data.spin:
                    hamiltonian_model = system.hamiltoniantransformer.from_irrep_Hamiltonian(out,data.soc_irrep,selfenergy=data.selfenergy)
                else:
                    hamiltonian_model = system.hamiltoniantransformer.from_irrep_Hamiltonian(out)
                overlap_model = data.overlap
            elif modeltype == 'HamiltonianSOC':
                if data.spin:
                    fullsoc = system.hamiltoniantransformer.nodeonlysoc_to_full(out_soc,data.self_idx,data.selfenergy.size(0))
                    hamiltonian_model = system.hamiltoniantransformer.from_irrep_Hamiltonian(out,fullsoc,selfenergy=data.selfenergy)
                else:
                    raise Exception("Trying to train SOC using band but graph has spin=False")
                overlap_model = data.overlap
            elif modeltype == 'Overlap':
                hamiltonian_model = data.hopping
                overlap_model = system.hamiltoniantransformer.from_irrep_Overlap(out)
            elif modeltype == 'SOC':
                if data.spin:
                    hamiltonian_model = system.hamiltoniantransformer.from_irrep_Hamiltonian(data.hopping_irrep,out,selfenergy=data.selfenergy)
                else:
                    raise Exception("Trying to train SOC using band but graph has spin=False")
                overlap_model = data.overlap
            else:
                raise Exception("Modeltype invalid")            
            
            
            #Get the element from the one-hot attribute
            elemlst=[system.element_decoding[i] for x in data.elementonehot.to('cpu') for i, value in enumerate(system.type_onehot) if torch.equal(value, x)]
            orbitals = orbitals_from_str_yzx(system.orbitalstrings,system.spin)
            sys_template = [(idx,orb)  for idx, atom in enumerate(elemlst) for orb in orbitals]
            basis_len= len(sys_template) 

            #Hamiltonia H(R) in my usual format
            R_vec_lst,HR_lst,SR_lst = [torch.tensor([0,0,0],dtype=torch.float32,device=system.device_train)],[torch.zeros((basis_len, basis_len), dtype=hamtype,device=system.device_train)],[torch.zeros((basis_len, basis_len), dtype=hamtype,device=system.device_train)]
            
            Rs = data.edge_shift.to(dtype=torch.float32,device=system.device_train)

            for Rvec in Rs:
                Ridx = next((i for i, vec in enumerate(R_vec_lst) if torch.equal(vec, Rvec)), None)
                Ridxneg = next((i for i, vec in enumerate(R_vec_lst) if torch.equal(vec, -Rvec)), None)
                if Ridx == None and Ridxneg == None:
                    R_vec_lst.append(Rvec)  # Add R in the list
                    R_vec_lst.append(-Rvec)  # Add R in the list
                    HR_lst.append(torch.zeros((basis_len, basis_len), dtype=hamtype,device=system.device_train))
                    HR_lst.append(torch.zeros((basis_len, basis_len), dtype=hamtype,device=system.device_train))
                    SR_lst.append(torch.zeros((basis_len, basis_len), dtype=hamtype,device=system.device_train))
                    SR_lst.append(torch.zeros((basis_len, basis_len), dtype=hamtype,device=system.device_train))


            for index,reversed,Rvec,ham,overlap in zip(data.edge_index.t(),data.reversed,data.edge_shift.to(torch.float32),hamiltonian_model,overlap_model):
                if not reversed:
                    Ridx = next((i for i, vec in enumerate(R_vec_lst) if torch.equal(vec, Rvec)), None)

                    # hoppingtemplate_row = [(index[1],orb) for orb in system.orbitals]  Old version
                    # hoppingtemplate_col = [(index[0],orb) for orb in system.orbitals]
                    # hoppingindices_row = [sys_template.index(label) for label in hoppingtemplate_row]
                    # hoppingindices_col = [sys_template.index(label) for label in hoppingtemplate_col]
                    hoppingindices_row = torch.arange(index[1]*system.num_orbitals,(index[1]+1)*system.num_orbitals)
                    hoppingindices_col = torch.arange(index[0]*system.num_orbitals,(index[0]+1)*system.num_orbitals) 
                    row_ix, col_ix = torch.meshgrid(torch.tensor(hoppingindices_row,dtype=torch.int32,device=system.device_train), torch.tensor(hoppingindices_col,dtype=torch.int32,device=system.device_train), indexing='ij')       
                    
                    HR_lst[Ridx][row_ix, col_ix] = ham.clone()
                    SR_lst[Ridx][row_ix, col_ix] = overlap.clone()
                    if torch.all(Rvec == 0) and index[1] != index[0]:               
                        HR_lst[Ridx][col_ix, row_ix] = ham.clone()
                        SR_lst[Ridx][col_ix, row_ix] = overlap.clone()       
                    
            eigenvalues = [] #Band eigenvalues
            for k in band_kpoints: #for each k-point append list of eigenvalues
                # The H(R) -> H(k) transformation
                sum_H = sum([torch.exp(2j * torch.pi * torch.dot(k, R))*HR for R, HR in zip(R_vec_lst, HR_lst)])

                if SR_lst: #case overlap matrix is not identity
                    # The S(R) -> S(k) transformation
                    sum_S = sum([torch.exp(2j * torch.pi * torch.dot(k, R))*SR for R, SR in zip(R_vec_lst, SR_lst)])

                    #Approximation for the general eigenvalue system using binomial series: sum_S ~= I
                    dS = sum_S - torch.eye(sum_S.size(0),device=system.device_train)
                    S_1_2 = (torch.eye(dS.size(0),device=system.device_train)  # Identity matrix
                    + (1/2.0) * dS
                    - (1/8.0) * dS @ dS
                    + (1/16.0) * dS @ dS @ dS
                    - (5/128.0) * dS @ dS @ dS @ dS)
                    # + (7/256.0) * dS @ dS @ dS @ dS @ dS
                    # - (21/1024.0) * dS @ dS @ dS @ dS @ dS @ dS
                    # + (33/2048.0) * dS @ dS @ dS @ dS @ dS @ dS @ dS)

                    S_1_2_inv = torch.linalg.pinv(S_1_2)

                    C = S_1_2_inv @ sum_H @ S_1_2_inv
                    
                    eigenval = torch.linalg.eigvalsh(C)
                    eigenvalues.append(eigenval)

                else:
                    eigenval = torch.linalg.eigvalsh(sum_H)
                    eigenvalues.append(eigenval) #solve system
            eigenvalues = torch.stack(eigenvalues,dim=0).to(device=data.band.device)
            
            print(f"Training with band -  beta={bandloss_beta}")
            loss += bandloss_beta * band_lossfunc(eigenvalues, data.band)

        print(f"Epoch {epoch+1}, Total Loss: {loss.item()}")
        writer.add_scalar("Loss/train", loss.item(), epoch)
        loss.backward()
        optimizer.step()

        # Log gradients and parameters to TensorBoard
        if gradientlog:
            total_grad_norm = 0
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}.value', param, epoch) 
                # Only log gradients if they exist (i.e., param.grad is not None)
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, epoch)
                    param_norm = param.grad.data.norm(2).item()  # Calculate the gradient norm
                else:
                    param_norm = 0  # If no gradient exists, log zero norm or handle accordingly

                writer.add_scalar(f'GradNorm/{name}', param_norm, epoch)
                total_grad_norm += param_norm
                
            for name, param in model.named_parameters():
                # Only log gradients if they exist (i.e., param.grad is not None)
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()  # Calculate the gradient norm
                else:
                    param_norm = 0  # If no gradient exists, log zero norm or handle accordingly

                writer.add_scalar(f'Grad %/{name}', 100*param_norm/total_grad_norm, epoch)
            
            writer.add_scalar('TotalGradNorm', total_grad_norm, epoch)

        #TODO I dont need to rotate it?
        if epoch % optm_multiplier == 0:
            # Calculate validation loss
            if validation_data_lst:
                #model.eval()   TODO: Sometime I get NaN or Inf, is that due to Batchnorm? Outside the training range?
                with torch.no_grad():
                    validation_loss = 0
                    if len(validation_data_lst) > validation_batch:
                        data_indices = torch.randint(0, len(validation_data_lst)-1, (validation_batch,), dtype=torch.long).tolist()
                    else:
                        data_indices = list(range(0,len(validation_data_lst)))

                    for validation_data in [validation_data_lst[i] for i in data_indices]:
                        validation_data.to(system.device_train)
                        if modeltype == 'HamiltonianSOC':
                            validation_out, validation_out_soc = model(validation_data)
                        else:
                            validation_out = model(validation_data)
                        if SymmGauge and modeltype != 'SOC':
                            minimum = (0, 1.0)
                            for id, group in enumerate(validation_data.group_transform):
                                if modeltype == 'Hamiltonian' or modeltype == 'Overlap' or modeltype == 'HamiltonianSOC':
                                    Dmat = system.hamiltonian_irreps.D_from_matrix(torch.Tensor(group)).to(system.device_train)
                                elif modeltype == 'Force':
                                    Dmat = Irreps("1x1o").D_from_matrix(torch.Tensor(group)).to(system.device_train)

                                if modeltype == 'Hamiltonian':
                                    temphopping = torch.matmul(Dmat,validation_data.hopping_irrep.t()).squeeze(0).t()
                                    if EnergyGauge:
                                        tempoverlap = torch.matmul(Dmat,validation_data.overlap_irrep.t()).squeeze(0).t()
                                        mu = torch.sum( (validation_out - temphopping) * tempoverlap ) / torch.sum(tempoverlap * tempoverlap)
                                        losstemp = lossfunc(validation_out,temphopping + mu*tempoverlap)
                                    else:
                                        losstemp = lossfunc(validation_out,temphopping)
                                elif modeltype == 'HamiltonianSOC':
                                    temphopping = torch.matmul(Dmat,validation_data.hopping_irrep.t()).squeeze(0).t()
                                    if EnergyGauge:
                                        tempoverlap = torch.matmul(Dmat,validation_data.overlap_irrep.t()).squeeze(0).t()
                                        mu = torch.sum( (validation_out - temphopping) * tempoverlap ) / torch.sum(tempoverlap * tempoverlap)
                                        losstemp = lossfunc(validation_out,temphopping + mu*tempoverlap)
                                    else:
                                        losstemp = lossfunc(validation_out,temphopping)
                                elif modeltype == 'Overlap':
                                    tempoverlap = torch.matmul(Dmat,validation_data.overlap_irrep.t()).squeeze(0).t()
                                    losstemp = lossfunc(validation_out,tempoverlap)
                                elif modeltype == 'Force':
                                    tempforces = torch.matmul(Dmat,validation_data.forces.t()).squeeze(0).t()
                                    losstemp = lossfunc(validation_out,tempforces)
                                else:
                                    raise Exception("Modeltype invalid")
                                if losstemp < minimum[1]:
                                    minimum = (id,losstemp)
                            transformeddata = validation_data.clone()
                            if modeltype == 'Hamiltonian' or modeltype == 'Overlap' or modeltype == 'HamiltonianSOC':
                                Dmat = system.hamiltonian_irreps.D_from_matrix(torch.tensor(transformeddata.group_transform[minimum[0]], dtype=torch.float32)).to(system.device_train)
                            elif modeltype == 'Force':
                                Dmat = Irreps("1x1o").D_from_matrix(torch.tensor(transformeddata.group_transform[minimum[0]], dtype=torch.float32)).to(system.device_train)
                            
                            if modeltype == 'Hamiltonian':
                                transformeddata.hopping_irrep = torch.matmul(Dmat,transformeddata.hopping_irrep.t()).squeeze(0).t()                                    
                                if EnergyGauge:
                                    transformeddata.overlap_irrep = torch.matmul(Dmat,transformeddata.overlap_irrep.t()).squeeze(0).t()
                                    mu = torch.sum( (validation_out - transformeddata.hopping_irrep) * transformeddata.overlap_irrep ) / torch.sum(transformeddata.overlap_irrep * transformeddata.overlap_irrep)
                                    validation_loss += lossfunc(validation_out,transformeddata.hopping_irrep + mu*transformeddata.overlap_irrep)
                                else:
                                    validation_loss += lossfunc(validation_out, transformeddata.hopping_irrep)
                            elif modeltype == 'HamiltonianSOC':
                                transformeddata.hopping_irrep = torch.matmul(Dmat,transformeddata.hopping_irrep.t()).squeeze(0).t()                                    
                                if EnergyGauge:
                                    transformeddata.overlap_irrep = torch.matmul(Dmat,transformeddata.overlap_irrep.t()).squeeze(0).t()
                                    mu = torch.sum( (validation_out - transformeddata.hopping_irrep) * transformeddata.overlap_irrep ) / torch.sum(transformeddata.overlap_irrep * transformeddata.overlap_irrep)
                                    validation_loss += lossfunc(validation_out,transformeddata.hopping_irrep + mu*transformeddata.overlap_irrep)
                                    validation_loss += soc_beta * lossfunc(validation_out_soc,transformeddata.soc_irrep)
                                else:
                                    validation_loss += lossfunc(validation_out,transformeddata.hopping_irrep)
                                    validation_loss += soc_beta * lossfunc(validation_out_soc,transformeddata.soc_irrep)
                            elif modeltype == 'Overlap':
                                transformeddata.overlap_irrep = torch.matmul(Dmat,transformeddata.overlap_irrep.t()).squeeze(0).t()
                                validation_loss += lossfunc(validation_out, transformeddata.overlap_irrep)
                            elif modeltype == 'Force':
                                transformeddata.forces = torch.matmul(Dmat,transformeddata.forces.t()).squeeze(0).t()
                                validation_loss += lossfunc(validation_out, transformeddata.forces)
                            elif modeltype == 'SOC':
                                validation_loss += lossfunc(validation_out, transformeddata.soc_irrep)
                            else:
                                raise Exception("Modeltype invalid")
                        else:
                            if modeltype == 'Hamiltonian':
                                if EnergyGauge:
                                    mu = torch.sum( (validation_out - validation_data.hopping_irrep) * validation_data.overlap_irrep ) / torch.sum(validation_data.overlap_irrep * validation_data.overlap_irrep)
                                    validation_loss += lossfunc(validation_out,validation_data.hopping_irrep + mu*validation_data.overlap_irrep)
                                else:
                                    validation_loss += lossfunc(validation_out, validation_data.hopping_irrep)
                            elif modeltype == 'HamiltonianSOC':
                                if EnergyGauge:
                                    mu = torch.sum( (validation_out - validation_data.hopping_irrep) * validation_data.overlap_irrep ) / torch.sum(validation_data.overlap_irrep * validation_data.overlap_irrep)
                                    validation_loss += lossfunc(validation_out,validation_data.hopping_irrep + mu*validation_data.overlap_irrep)
                                    validation_loss += soc_beta * lossfunc(validation_out_soc,validation_data.soc_irrep)
                                else:
                                    validation_loss += lossfunc(validation_out,validation_data.hopping_irrep)
                                    validation_loss += soc_beta * lossfunc(validation_out_soc,validation_data.soc_irrep)
                            elif modeltype == 'Overlap':
                                validation_loss += lossfunc(validation_out, validation_data.overlap_irrep)
                            elif modeltype == 'Force':
                                validation_loss += lossfunc(validation_out, validation_data.forces)
                            elif modeltype == 'SOC':
                                validation_loss += lossfunc(validation_out, validation_data.soc_irrep)
                            else:
                                raise Exception("Modeltype invalid")

                    
                    validation_loss = validation_loss / float(len(data_indices))

                    # Print the loss at each epoch
                    print(f"Epoch {epoch+1}, validation_Loss: {validation_loss.item()}")
                    writer.add_scalar("ValidationLoss/train", validation_loss.item(), epoch)
                    writer.flush()
                    
                    # Step the scheduler with the validation loss
                    scheduler.step(validation_loss)

                    # Print the learning rate
                    for param_group in optimizer.param_groups:
                        print(f"Epoch {epoch+1}, Learning rate: {param_group['lr']}")
                        writer.add_scalar("Learning rate/train", param_group['lr'], epoch)

                    # Early stopping logic
                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        early_stopping_counter = 0
                        torch.save(model.state_dict(), f'./bestmodel-{runname}.pth')
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
    
        if epoch % 30 == 0:
            torch.save(model.state_dict(), f'./lastmodel-{runname}.pth')
    # add the hyperparameters and metrics to TensorBoard
    #TODO complete the hyperparameters
    writer.add_hparams(
        {
            "MLP_layers": system.MLP_layers,
            "weight_hidden": system.weight_hidden,
            "Irrepsize": system.hidden_irreps.dim,
            "Irrepmaxl": system.hidden_irreps.lmax,
            "convolution_layers": system.convolution_layers,
            "dist_onehotsize": system.dist_onehotsize,
            "gauss_width": system.gauss_width,
            "neighbour_cutoff": system.neighbour_cutoff
        },
        {
            "train_lastloss": loss.item(),
            "validation_lastloss": validation_loss.item(),
        },
    )        
    writer.flush()
    writer.close()                   
     