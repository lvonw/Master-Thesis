import constants
import matplotlib.pyplot    as plt
import numpy                as np
import torch
import torch.optim          as optim
import util
import os

from data.data_util     import DataVisualizer
from debug              import Printer, print_to_log_file
from torch.utils.data   import DataLoader
from tqdm               import tqdm


def print_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Plot")
    plt.show()

def show_log_loss():
    with open(constants.LOG_PATH_LOSS, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    
    print_loss_graph(numbers)

def show_tensors(dataloader, num_tensors=1):
    printer = Printer()
    printer.print_log("Preparing to show first image...")
    for images, _ in dataloader:
        image_tensor = images[0]
        
        DataVisualizer.show_image_tensor(image_tensor)
        num_tensors -= 1
        if num_tensors == 0:
            break
        
    printer.print_log("Finished.")


def train(model, training_dataset, validation_dataset, configuration):
    printer = Printer()
    
    batch_size = 128
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(constants.DATALOADER_SEED)

    training_dataloader = DataLoader(training_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     generator=data_loader_generator,
                                     num_workers=os.cpu_count() -2,
                                     pin_memory=True, 
                                     pin_memory_device=str(util.get_device()))
    
    validation_dataloader = DataLoader(validation_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       generator=data_loader_generator,
                                       num_workers=os.cpu_count() - 2,
                                       pin_memory=True, 
                                       pin_memory_device=str(util.get_device()))
            
    training_losses     = []
    validation_losses   = []
    
    optimizer       = optim.Adam(model.parameters(), lr=4.5e-6)
    num_epochs      = 0 # 100
    logging_steps   = 10

    #show_tensors(dataloader, 1)
    
    print_to_log_file(f"\nModel: {model.name}", constants.TRAINING_LOSS_LOG)

    model.to(util.get_device())
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()

        print_to_log_file(f"Epoch: {epoch+1}", constants.TRAINING_LOSS_LOG)

        running_loss = 0.0
        for i, data in tqdm(enumerate(training_dataloader, 0), 
                            total=len(training_dataloader),
                            desc="Training Steps",
                            disable=False):
            optimizer.zero_grad()
            
            if len(data) == 2:
                inputs, labels = data
            else:
                inputs, labels, _ = data

            inputs          = inputs.to(util.get_device())

            if len(labels):
                labels          = labels.to(util.get_device())

            loss, _ = model.training_step(inputs, labels)
            
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            running_loss += loss.item()

            if ((i+1) % logging_steps) == 0:
                printer.clear_all()
                print_to_log_file(running_loss/(i+1), 
                                  constants.TRAINING_LOSS_LOG)

        torch.save(model.state_dict(), constants.MODEL_PATH_TEST)

        validation_loss = __validate(model,
                                     validation_dataloader, 
                                     configuration,
                                     batch_size)
        validation_losses.append(validation_loss)
        printer.print_log(f"Validation Loss: {validation_loss:.4f}")

    # print_loss_graph(training_losses)

    model.eval()
    with torch.no_grad():
        printer.print_log("Preparing to show first image...")
        for images, _ in training_dataloader:
            image_tensor = images[0]
            
            DataVisualizer.show_image_tensor(image_tensor)
            images          = images.to(util.get_device())
            reconstruction = model(images)[0]
            reconstruction          = reconstruction.to("cpu")
            DataVisualizer.show_image_tensor(reconstruction)
            break
    
    print_to_log_file("\n", constants.TRAINING_LOSS_LOG)

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

            inputs                      = inputs.to(util.get_device())
            if len(labels):
                labels                  = labels.to(util.get_device())

            loss, individual_losses = model.training_step(inputs, labels)
            
            running_loss    += loss.item()
            losses          += [l.item() for l in individual_losses]
            metadatas       += metadata


    validation_loss = running_loss / (len(dataloader) * batch_size)
    
    if metadatas:
        z_scores        = __compute_z_score(losses)
        outliers        = []
        for i, z_score in enumerate(z_scores, 0):
            if z_score >= 3:
                outliers.append(metadatas[i])

    print_to_log_file(validation_loss, constants.VALIDATION_LOSS_LOG)
    return validation_loss

def __compute_z_score(losses):
    losses  = np.array(losses)
    mu      = np.mean(losses)
    sigma   = np.std(losses)
    
    z_scores = (losses - mu) / sigma
        
    return z_scores

        