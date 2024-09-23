import util

from tqdm import tqdm

def train(model, dataloader, configuration):
    optimizer = None
    validation_graph    = []
    num_epochs = 0
    criterion = None

    for epoch in tqdm(range(num_epochs)):
        
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        
        for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            optimizer.zero_grad()
            
            inputs, labels  = data
            inputs          = inputs.to(util.get_device())
            labels          = labels.to(util.get_device())
            outputs         = model(inputs)

            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()