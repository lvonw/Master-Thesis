import constants
import os
import torch
import torch.nn         as nn

# TODO Need to make it so we dont constantly check this
def get_device(idle = False):
    if idle:
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)


def make_path(path_string):
    path_arr = path_string.split("/")
    return os.path.join(*path_arr)

def get_model_family(model):
    return model.model_family

def get_model_file_path(model):
    model_family = get_model_family(model)

    model_dir = os.path.join(constants.MODEL_PATH_MASTER,
                        model_family)
    os.makedirs(model_dir, exist_ok=True)
    
    return os.path.join(model_dir,
                        model.name + constants.MODEL_FILE_TYPE)
    
def save_model(model):
    if model.can_save:
        torch.save(model.state_dict(), get_model_file_path(model))

def load_model(model, strict=True):
    model_path = get_model_file_path(model)
    if not os.path.exists(model_path):
        return False

    model.load_state_dict(torch.load(model_path, 
                                     weights_only=False),
                          strict=strict)
    return True

def save_checkpoint(model, epoch_idx, save_counter, backup_at = 5):
    if not model.can_save:
        return
    
    optimizer_states = [opt.state_dict() for opt in model.optimizers]
    checkpoint = {
        "epoch": epoch_idx,
        "model_state": model.state_dict(),
        "optimizer_states": optimizer_states,
    }

    model_file_path = get_model_file_path(model)

    # Save to a backup at every n saves
    if (save_counter % backup_at == 0 
        and os.path.exists(model_file_path)):

        backup_path = os.path.join(
            constants.MODEL_PATH_MASTER,
            get_model_family(model),
            constants.MODEL_BACKUP_FOLDER)
        os.makedirs(backup_path, exist_ok=True)

        backup_file_path = os.path.join(
            backup_path,
            os.path.splitext(os.path.basename(model_file_path))[0]
            + "_backup" 
            + constants.MODEL_FILE_TYPE)
        
        if os.path.exists(backup_file_path):
            os.remove(backup_file_path)
        
        model_file_path = backup_file_path

    torch.save(checkpoint, model_file_path)

def load_checkpoint(model,
                    strict=True):
    model_path = get_model_file_path(model)
    if not os.path.exists(model_path):
        return 0

    checkpoint = torch.load(model_path, 
                            weights_only=False,
                            map_location="cuda")
    
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    
    optimizer_state_dicts = checkpoint["optimizer_states"]
    for optimizer, state_dict in zip(model.optimizers, optimizer_state_dicts):
        optimizer.load_state_dict(state_dict)
        
        # Need to manually move the optimizer params back to the gpu
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(get_device())
    
    epoch_idx = checkpoint["epoch"] + 1
    
    return epoch_idx