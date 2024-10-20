import constants
import math
import matplotlib.pyplot    as plt
import numpy                as np
import torch
import torch.optim          as optim
import util
import os

from data.data_util     import DataVisualizer
from debug              import Printer, print_to_log_file
from torch.utils.data   import DataLoader, random_split
from tqdm               import tqdm

def print_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Plot")
    plt.show()
        
def __prepare_data(data):
    if len(data) == 2:
        inputs, labels = data
        metadata = []
    else:
        inputs, labels, metadata = data

    inputs = inputs.to(util.get_device())
    if len(labels):
        labels = labels.to(util.get_device())

    return inputs, labels, metadata

def __get_data_splits(dataset, training_data_split):
        total_data      = len(dataset)
        training_split  = math.ceil( total_data * training_data_split)
        
        return random_split(dataset, 
                            [training_split, total_data - training_split],
                            generator=torch.Generator()
                                .manual_seed(constants.DATALOADER_SEED))

def train(model, 
          complete_dataset, 
          configuration,
          starting_epoch = 0):
    printer         = Printer()
    data_visualizer = DataVisualizer()
    
    batch_size      = configuration["batch_size"]
    num_epochs      = configuration["epochs"]
    logging_steps   = configuration["logging_steps"]

    cpu_count       = configuration["num_workers"]
    cpu_count       = (cpu_count if cpu_count >= 0 
                       else os.cpu_count() + cpu_count)

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(constants.DATALOADER_SEED)

    loss_weights = complete_dataset.loss_weights

    training_dataset, validation_dataset = __get_data_splits(
            complete_dataset,
            configuration["Data_Split"])
    

    training_dataloader = DataLoader(training_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     generator=data_loader_generator,
                                     num_workers=cpu_count,
                                     pin_memory=True, 
                                     pin_memory_device=str(util.get_device()))
    
    validation_dataloader = DataLoader(validation_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       generator=data_loader_generator,
                                       num_workers=cpu_count,
                                       pin_memory=True, 
                                       pin_memory_device=str(util.get_device()))
            
    training_losses     = []
    validation_losses   = []
            
    print_to_log_file(f"\nModel: {model.name}", constants.TRAINING_LOSS_LOG)

    model.to(util.get_device())
    
    # Training ================================================================
    for epoch_idx in tqdm(range(starting_epoch, num_epochs), 
                          total     = num_epochs,
                          desc      = "Epochs",
                          disable   = False,
                          colour    = "magenta"):
        model.train()

        print_to_log_file(f"Epoch: {epoch_idx+1}", constants.TRAINING_LOSS_LOG)

        running_losses = [0.] * len(model.optimizers)
        for training_step_idx, data in tqdm(enumerate(training_dataloader, 0), 
                                            total   = len(training_dataloader),
                                            desc    = "Training Steps",
                                            disable = False,
                                            colour  = "red"):
            
            for optimizer_idx, optimizer in enumerate(model.optimizers):
                # Smoother loss for monitoring, averaged over the epoch
                running_loss = running_losses[optimizer_idx]
                
                # Main training loop ==========================================
                optimizer.zero_grad()
                inputs, labels, _ = __prepare_data(data)

                loss, _ = model.training_step(inputs, 
                                              labels, 
                                              loss_weights,
                                              epoch_idx,
                                              training_step_idx,
                                              optimizer_idx)
                # Ignore irrelevant losses
                if loss.item() == 0:  
                    continue

                loss.backward()
                optimizer.step()

                # Loss monitoring =============================================
                training_losses.append(loss.item())
                
                running_loss                    += loss.item()
                running_losses[optimizer_idx]   = running_loss 

                if ((training_step_idx + 1) % logging_steps) == 0:
                    monitoring_loss = running_loss / (training_step_idx + 1)
                    print_to_log_file(
                        f"{optimizer_idx}: {monitoring_loss}", 
                        constants.TRAINING_LOSS_LOG)
                    
            # Post training step behaviour ====================================
            model.on_training_step_completed()

        # Post epoch behaviour ================================================
        if configuration["Save_after_epoch"]:
            util.save_checkpoint(model, epoch_idx)
        
        # Validation ==========================================================
        if len(validation_dataloader):
            validation_loss = __validate(model,
                                            validation_dataloader, 
                                            configuration["Validation"],
                                            batch_size)
            validation_losses.append(validation_loss)
            printer.print_log(f"Validation Loss: {validation_loss:.4f}")

    print_to_log_file("\n", constants.TRAINING_LOSS_LOG)
    
    # Post training evaluations =============================================== 
    model.eval()

    with torch.no_grad():
        for data in training_dataloader:
            inputs, _, metadata = __prepare_data(data)

            reconstructions = model(inputs)[0]
            
            for image_idx in range(len(inputs)):
                original_image  = inputs[image_idx]
                reconstruction  = reconstructions[image_idx]

                data_visualizer.create_image_tensor_tuple(
                    [original_image, reconstruction], 
                    title=metadata["filename"][image_idx])

                inputs  = inputs.to(util.get_device())
                data_visualizer.show_ensemble()


            break    

def __validate(model, dataloader, configuration, batch_size):    
    model.eval()

    running_loss = 0.0
    losses      = []
    metadatas   = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0), 
                            total   = len(dataloader),
                            desc    = "Validation"):
            
            if len(data) == 2:
                inputs, labels = data
                metadata = []
            else:
                inputs, labels, metadata = data

            inputs = inputs.to(util.get_device())
            if len(labels):
                labels = labels.to(util.get_device())

            loss, individual_losses = model.training_step(inputs, labels)
            
            running_loss    += loss.item()
            if individual_losses:
                losses          += [l.item() for l in individual_losses]
            metadatas       += metadata


    validation_loss = running_loss / (len(dataloader) * batch_size)
    
    if metadatas:
        z_scores        = __compute_z_score(losses)
        outliers        = []
        outlier_score   = configuration["z_score"]
        for i, z_score in enumerate(z_scores, 0):
            if z_score >= outlier_score:
                outliers.append(metadatas[i])

    print_to_log_file(validation_loss, constants.VALIDATION_LOSS_LOG)
    return validation_loss

def __compute_z_score(losses):
    losses  = np.array(losses)
    mu      = np.mean(losses)
    sigma   = np.std(losses)
    
    z_scores = (losses - mu) / sigma
        
    return z_scores

        