import constants
import os
import torch
import util

import matplotlib.pyplot    as plt
import numpy                as np

from data.data_util                 import DataVisualizer
from debug                          import (Printer,
                                            print_to_model_loss_log)
from torch.utils.data               import DataLoader
from torch.utils.data.distributed   import DistributedSampler
from tqdm                           import tqdm


def print_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Plot")
    plt.show()
        

def train(model, 
          model_state,
          dataset_wrapper, 
          configuration,
          starting_epoch = 0,
          is_distributed = False,
          global_rank = 0,
          local_rank = 0,
          local_amount = 1,
          gloabl_amount = 1):
    
    printer         = Printer()
    data_visualizer = DataVisualizer()
    save_counter    = 0

    # Training Parameters =====================================================
    num_epochs      = configuration["epochs"]
    logging_steps   = configuration["logging_steps"]

    # Loss Weights ============================================================
    loss_weights = {}
    loss_weights = dataset_wrapper.loss_weights

    # Data Loaders ============================================================
    training_dataloader, validation_dataloader = __prepare_data_loaders(
        dataset_wrapper,
        configuration,
        is_distributed)
            
    validation_losses   = []
            
    # Training ================================================================    
    total_training_step_idx = 0
    for epoch_idx in tqdm(
        range(starting_epoch, num_epochs),
        total     = num_epochs,
        initial   = starting_epoch,
        position  = local_rank,
        leave     = True, 
        desc      = f"[{local_rank}] Epochs",
        disable   = False,
        colour    = "magenta"):
        model.train()

        if global_rank == 0:
            print_to_model_loss_log(f"Epoch: {epoch_idx+1}",
                                    model, 
                                    constants.TRAINING_LOSS_LOG)

        for training_step_idx, data in tqdm(
            enumerate(training_dataloader, 0), 
            total   = len(training_dataloader),
            position= local_amount + local_rank,
            leave   = False,
            desc    = f"[{local_rank}] Training Steps",
            disable = False,
            colour  = "red"):
            inputs, labels, _ = __prepare_data(data)
            total_training_step_idx = (len(training_dataloader) * epoch_idx 
                                       + training_step_idx) * gloabl_amount
            
            
            # Main training loop over all optimizers ==========================
            for optimizer_idx, optimizer in enumerate(model_state.optimizers):                
                optimizer.zero_grad()   
                loss, _ = model(None,
                                True,
                                inputs, 
                                labels, 
                                loss_weights,
                                epoch_idx,
                                total_training_step_idx,
                                training_step_idx,
                                optimizer_idx,
                                global_rank,
                                local_rank)
                    
                loss.backward()
                optimizer.step()

            # Loss monitoring =================================================
            if ((training_step_idx + 1) % logging_steps == 0 
                and global_rank == 0):

                print_to_model_loss_log(model_state.loss_log, 
                                        model,
                                        constants.TRAINING_LOSS_LOG)
                    
            # Post training step behaviour ====================================
            model_state.on_training_step_completed()

        # Post epoch behaviour ================================================
        if (configuration["Save_after_epoch"] 
            and (epoch_idx + 1) % configuration["save_after_n"] == 0 
            and global_rank == 0):

            save_counter += 1
            util.save_checkpoint(model_state, 
                                 epoch_idx, 
                                 save_counter, 
                                 configuration["backup_after_n"])
        
        # Validation ==========================================================
        if validation_dataloader is not None and len(validation_dataloader):
            validation_loss = __validate(model,
                                            validation_dataloader, 
                                            configuration["Validation"])
            validation_losses.append(validation_loss)
            printer.print_log(f"Validation Loss: {validation_loss:.4f}")
    
    # Post training evaluations =============================================== 
    lp = LaplaceFilter()

    model.eval()
    model.on_loaded_as_pretrained()
    with torch.no_grad():
        for training_step_idx, data in tqdm(enumerate(training_dataloader, 0)):
            inputs, _, metadata = __prepare_data(data)

            reconstructions = model(inputs)[0]
            
            for image_idx in range(len(inputs)):
                original_image  = inputs[image_idx]
                reconstruction  = reconstructions[image_idx]

                # title = metadata["filename"][image_idx]
                title = "mnnist"

                data_visualizer.create_image_tensor_tuple(
                    [original_image, reconstruction], 
                    title=title)
                
                data_visualizer.create_image_tensor_tuple(
                    [lp(original_image), lp(reconstruction)], 
                    title=title)

                inputs  = inputs.to(util.get_device())
                data_visualizer.show_ensemble()


            break    

import torch.nn.functional                  as f
class LaplaceFilter():
    def __init__(self):
        self.kernel = torch.tensor([[0,  1, 0], 
                                    [1, -4, 1], 
                                    [0,  1, 0]],   
                                    dtype=torch.float32).view(1, 1, 3, 3).to(util.get_device())

    def __call__(self, x):
        return f.conv2d(x, self.kernel, padding=0)
    

class SobelFilter():
    def __init__(self):
        self.sobel_x_kernel = torch.tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=torch.float32)

        self.sobel_y_kernel = torch.tensor([[-1, -2, -1], 
                                    [ 0,  0,  0], 
                                    [ 1,  2,  1]], dtype=torch.float32)

        self.sobel_x_kernel = self.sobel_x_kernel.view(1, 1, 3, 3).to(util.get_device())
        self.sobel_y_kernel = self.sobel_y_kernel.view(1, 1, 3, 3).to(util.get_device())
        
        #self.sobel_x_kernel /= 6
        #self.sobel_y_kernel /= 6
    def __call__(self, x):
        return f.conv2d(f.conv2d(x, self.sobel_x_kernel, padding=0), 
                        self.sobel_y_kernel, padding=0)

def __validate(model, dataloader, configuration):    
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


    validation_loss = running_loss / (len(dataloader))
    
    if metadatas:
        z_scores        = __compute_z_score(losses)
        outliers        = []
        outlier_score   = configuration["z_score"]
        for i, z_score in enumerate(z_scores, 0):
            if z_score >= outlier_score:
                outliers.append(metadatas[i])

    print_to_model_loss_log(validation_loss, 
                            model,
                            constants.VALIDATION_LOSS_LOG)
    return validation_loss

def __compute_z_score(losses):
    losses  = np.array(losses)
    mu      = np.mean(losses)
    sigma   = np.std(losses)
    
    z_scores = (losses - mu) / sigma
        
    return z_scores

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

def __prepare_data_loaders(dataset_wrapper, 
                           configuration, 
                           is_distributed=False):
    training_dataset, validation_dataset = dataset_wrapper.get_splits(
        split=configuration["Data_Split"])
    
    batch_size              = configuration["batch_size"]
    cpu_count               = configuration["num_workers"]
    cpu_count               = (cpu_count if cpu_count >= 0 
                               else os.cpu_count() + cpu_count)

    data_loader_generator   = torch.Generator()
    data_loader_generator.manual_seed(constants.DATALOADER_SEED)
    
    if is_distributed:
        training_dataloader = DataLoader(
            training_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            generator=data_loader_generator,
            num_workers=cpu_count,
            persistent_workers=True,
            sampler=DistributedSampler(training_dataset, shuffle=True)
        )
    else: 
        training_dataloader = DataLoader(
            training_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            generator=data_loader_generator,
            num_workers=cpu_count,
            persistent_workers=True,
            pin_memory=True,
            pin_memory_device=str(util.get_device())
        )
    # validation_dataloader = DataLoader(
    #     validation_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False,
    #     generator=data_loader_generator,
    #     num_workers=cpu_count)
    
    return training_dataloader, None #validation_dataloader
