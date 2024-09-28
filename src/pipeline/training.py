import constants
import matplotlib.pyplot    as plt
import numpy                as np
import torch
import torch.optim          as optim
import util

from tqdm   import tqdm
from debug  import Printer


def print_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Plot")
    plt.show()


def log_loss(loss):
    with open(constants.LOG_PATH_LOSS, 'a') as file:
        file.write(f"{loss}\n")

def show_tensors(dataloader):
    printer = Printer()
    printer.print_log("Preparing to show first image...")
    for images, _ in dataloader:
        image_tensor = images[0]
        break
    printer.print_log("Finished.")

    
    image_tensor = image_tensor.permute(1, 2, 0)

    image_numpy = image_tensor.numpy()
    
    # Plot the image using matplotlib
    plt.imshow(image_numpy)
    plt.title(f"Label")
    plt.axis('off')
    plt.show()


def train(model, dataloader, configuration):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    num_epochs = 10

    #show_first_tensor(dataloader)
    
    model.to(util.get_device())
    model.train()
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = 0.0
        epoch_loss = 0.0
        
        for i, data in tqdm(enumerate(dataloader, 0), 
                            total=len(dataloader),
                            desc="Training Steps"):
            optimizer.zero_grad()
            
            inputs, labels  = data
            inputs          = inputs.to(util.get_device())
            #labels          = labels.to(util.get_device())

            loss = model.training_step(inputs, labels)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            running_loss += loss.item()
            epoch_loss += loss.item()


            if ((i+1) % 10) == 0:
                log_loss(running_loss/i)

        torch.save(model.state_dict(), constants.MODEL_PATH_TEST)

    print_loss_graph(losses)