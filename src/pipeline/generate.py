import constants
import os
import torch
import util

import numpy                    as np

from data.data_util             import (GeoUtil, 
                                        DataVisualizer, 
                                        NormalizationMethod)
from datetime                   import datetime
from debug                      import Printer
from enum                       import Enum
from generation.perlin.perlin   import FractalPerlinGenerator
from PIL                        import Image
from pipeline.grid              import GenerationGrid
from tqdm                       import tqdm

import matplotlib.pyplot as plt

class MaskInterpolation(Enum):
    NONE                = "None"
    LINEAR              = "Linear"
    LEFT_COSINE         = "Left Cosine"
    RIGHT_COSINE        = "Right Cosine"
    LEFT_EXPONENTIAL    = "Left Exponential"
    RIGHT_EXPONENTIAL   = "Right Exponential"

@torch.no_grad()
def generate(model, 
             configuration  = None, 
             title          = "asd",
             save_only      = False):
    
    model.eval()
    model.apply_ema()
    model.to(util.get_device())
    
    data_visualizer     = DataVisualizer()
    printer             = Printer()
    generated_results   = []
    perlin_transform    = lambda x : x#(x / 6) - 0.8
    perlin_generator    = FractalPerlinGenerator(configuration["Perlin"],
                                                 perlin_transform)

    # No Configuration ========================================================
    if configuration is None:
        printer.print_log("Generating without configuration")

        generated_results.append(__generate_unguided(
            model,
            [[1,1], [2,5], [5, 12], [12, 28]],
            4,
            1))
    
    # Unguided ================================================================
    if configuration is not None and configuration["Unguided"]["active"]:
        printer.print_log("Generating unguided Samples")

        unguided_config = configuration["Unguided"]
        generated_results.append(__generate_unguided(
            model,
            unguided_config["labels"],
            unguided_config["samples"],
            unguided_config["iterations"]))

    # Image to Image ==========================================================
    if configuration is not None and configuration["Image2Image"]["active"]:
        printer.print_log("Generating sketch guided Samples")

        i2i_config = configuration["Image2Image"]
        generated_results.append(__generate_sketch_based(
            model,
            i2i_config["labels"],
            i2i_config["use_perlin"],
            perlin_generator,
            i2i_config["samples"],
            i2i_config["iterations"],
            i2i_config["weight"],
            i2i_config["sketch"],
            i2i_config["combine"],
            i2i_config["combination_alpha"]))

    # Grid ====================================================================
    if configuration is not None and configuration["Grid"]["active"]:
        printer.print_log("Generating Grid")

        grid_config = configuration["Grid"]
        generated_results.append(__generate_grid(
            model,
            grid_config["labels"],
            grid_config["use_perlin"],
            grid_config["grid_x"],
            grid_config["grid_y"],
            grid_config["iterations"],
            grid_config["weight"],
            perlin_generator,
            grid_config["regenerate_first_chunk"],
            MaskInterpolation.RIGHT_COSINE))

    # Inpainting ==============================================================
    if configuration is not None and configuration["Inpainting"]["active"]:
        printer.print_log("Generating inpainted Samples")

        inpainting_config = configuration["Inpainting"]
        generated_results.append(__generate_inpainting(
            model,
            inpainting_config["labels"],
            inpainting_config["samples"],
            inpainting_config["iterations"],
            inpainting_config["image"],
            inpainting_config["mask"],
            MaskInterpolation.RIGHT_COSINE))

    # Visualisation ===========================================================
    printer.print_log("Preparing Visualisations...")
    
    for result in generated_results:
        # 2D
        data_visualizer.create_image_tensor_tuple(result,
                                                  title=str(title),
                                                  three_dimensional=False) 
        # 3D
        if configuration["show_3d"]:
            data_visualizer.create_image_tensor_tuple(result,
                                                      title=str(title),
                                                      three_dimensional=True) 

    time = datetime.now().strftime("%m-%d_%H-%M-%S")
    save = (save_only 
            or (configuration is not None and configuration["save_plots"]))
    data_visualizer.show_ensemble(
        save        = save,
        filename    = f"infinite_{title}_{time}",
        model       = model,
        save_only   = save_only)
    
    # Saving ==================================================================
    # Save Full array
    if configuration["save_array"]:
        printer.print_log("Saving samples as arrays.")

        array_path_master = os.path.join(
            constants.LOG_PATH, 
            model.model_family,
            model.name,
            constants.LOG_HEIGHTMAPS)
        os.makedirs(array_path_master, exist_ok=True)

        time    = datetime.now().strftime("%m-%d_%H-%M-%S")
        index   = 0
        for result in generated_results:
            for sample in result:
                if isinstance(sample, torch.Tensor):
                    array = sample.to(util.get_device(idle=True)).numpy()
                else:
                    array = sample

                if len(array.shape) == 4: 
                    array = array[0][0]
                elif len(array.shape) == 3: 
                    array = array[0]

                index += 1
                array_path = os.path.join(
                    array_path_master,
                    f"infinite_{title}_{time}_{index}" 
                    + constants.LOG_HEIGHTMAP_FORMAT)

                np.save(array_path, array)  

    # Back to image space


# =============================================================================
# Unguided Generation
# =============================================================================
@torch.no_grad()
def __generate_unguided(model,
                        labels,
                        amount_samples,
                        iterations):
    samples = []
    
    for i in tqdm(range(iterations),
                  total     = iterations,
                  desc      = "Generating unguided Samples",
                  position  = 0,
                  leave     = False,
                  colour    = "magenta",
                  disable   = False):
            
        label = [labels[i % len(labels)]]

        sample = model.generate(
            label, 
            amount_samples,
            dynamic_device  = False, 
            fast_cfg        = True)
        
        samples.append(sample)
    
    return samples

# =============================================================================
# Image to Image
# =============================================================================
@torch.no_grad()
def __generate_sketch_based(model,
                            labels,
                            use_perlin,
                            perlin_generator,
                            amount_samples,
                            iterations,
                            weight,
                            sketch,
                            combine,
                            combination_alpha):
    
    sketch_tensor   = None 
    samples         = []

    # Use Perlin as Sketch, takes precedent over specified image ==============
    if use_perlin:
        coordinate      = (0, 0)
        perlin_image    = perlin_generator.generate_image(coordinate)
        sketch_tensor   = (torch.tensor(perlin_image, 
                                        dtype = torch.float32)
                           .unsqueeze(dim=0)
                           .unsqueeze(dim=0)
                           .to(util.get_device()))
    # Load Sketch if necessary ================================================
    if sketch is not None:
        input_image_path    = os.path.join(constants.RESOURCE_PATH_TEST_IMAGES,
                                           sketch)
        input_image         = Image.open(input_image_path)
        input_array         = np.array(input_image).transpose(2, 0, 1)[0]

        # Use only sketch
        if not combine and sketch_tensor is None:
            sketch_tensor        = GeoUtil.get_normalized_array(
                input_array,
                NormalizationMethod.CLIPPED_LINEAR,
                0,
                255).unsqueeze(dim=0).unsqueeze(dim=0).to(util.get_device())
        # Combine Perlin and sketch
        elif combine and sketch_tensor is not None:
            loaded_tensor = GeoUtil.get_normalized_array(
                input_array,
                NormalizationMethod.CLIPPED_LINEAR,
                0,
                255).unsqueeze(dim=0).unsqueeze(dim=0).to(util.get_device())
            
            sketch_tensor = (combination_alpha * loaded_tensor 
                             + (1 - combination_alpha) * sketch_tensor) 
    else:
        return []
    
    samples.append(sketch_tensor)

    for i in tqdm(range(iterations),
                    total     = iterations,
                    desc      = "Generating Samples",
                    position  = 0,
                    leave     = False,
                    colour    = "magenta",
                    disable   = False):
        
        label = [labels[i % len(labels)]]

        sample = model.generate(label, 
                                amount_samples, 
                                input_tensor       = sketch_tensor,
                                img2img_strength   = weight,
                                dynamic_device     = False,
                                fast_cfg           = True)
        
        samples.append(sample)

    return samples

# =============================================================================
# Grid Generation
# =============================================================================
@torch.no_grad()
def __generate_grid(model,
                    labels,
                    use_perlin,
                    grid_x, 
                    grid_y,
                    iterations              = 8,
                    weight                  = 0.8,
                    perlin_generator        = None,
                    regenerate_first_chunk  = True,
                    alpha_mode              = MaskInterpolation.RIGHT_COSINE):
     
    # Grid ====================================================================
    grid = GenerationGrid((1,) + model.get_output_shape(), 
                          perlin_generator.get_minimum_overlap())
    
    # Generate Mask Alphas ====================================================
    alpha = __get_alpha(perlin_generator.get_minimum_overlap(),
                        alpha_mode)
    
    amount_cells = grid_x * grid_y


    # Generate all Samples ====================================================
    for i in range(iterations):
        for cell_idx in tqdm(range(amount_cells),
                        total     = amount_cells,
                        desc      = f"Generating Grid ({i + 1})",
                        position  = 0,
                        leave     = False,
                        colour    = "magenta",
                        disable   = False):
            
            coordinate  = (cell_idx % grid_x, cell_idx // grid_x)
            label       = [labels[cell_idx % len(labels)]]

            __generate_chunk(model,
                             grid,
                             use_perlin,
                             perlin_generator,
                             coordinate,
                             weight,
                             alpha,
                             label)
        
    # First chunk behaviour ===================================================
    # saved_stitched_image, stitched_sketch = grid.stitch_image()
    if regenerate_first_chunk:
        Printer().print_log("Regenerating first Chunk")
        __generate_chunk(model,
                         grid,
                         use_perlin,
                         perlin_generator,
                         (0, 0),
                         weight,
                         alpha,
                         [labels[0]])
    
    # Stitch final image ======================================================
    stitched_image, stitched_sketch = grid.stitch_image()
        
    return [stitched_sketch, stitched_image]

def __generate_chunk(model,
                     grid,
                     use_perlin,
                     perlin_generator, 
                     coordinate,
                     weight,
                     alpha,
                     label,
                     amount_samples = 1):

    if use_perlin:
        # t = (coordinate[0]*10, coordinate[1]*12)
        perlin_image    = perlin_generator.generate_image(coordinate)
        perlin_tensor   = (torch.tensor(perlin_image, 
                                        dtype = torch.float32)
                           .unsqueeze(dim=0)
                           .unsqueeze(dim=0)
                           .to(util.get_device()))

    mask, masked_image = grid.get_mask_for_coordinate(coordinate, 
                                                      alpha,
                                                      sketch = perlin_tensor)
    samples = model.generate(label, 
                             amount_samples, 
                             input_tensor       = perlin_tensor,
                             img2img_strength   = weight,
                             mask               = mask,
                             masked_input       = masked_image,
                             dynamic_device     = False,
                             fast_cfg           = True)
                                
    final_image = grid.create_final_image(samples, 
                                          masked_image,
                                          mask)

    grid.insert(final_image[0], coordinate, perlin_tensor[0])



# =============================================================================
# Inpainting
# =============================================================================
@torch.no_grad()
def __generate_inpainting():
    pass

def __get_alpha(overlap_length, 
                interpolation_type = MaskInterpolation.NONE):

    match interpolation_type:
        case MaskInterpolation.NONE:
            alpha = 1
        case MaskInterpolation.LINEAR:
            alpha = torch.linspace(1, 
                                   0, 
                                   steps=overlap_length)
        case MaskInterpolation.LEFT_COSINE:
            alpha = torch.linspace(torch.pi / 2, 
                                   0, 
                                   steps=overlap_length)
            alpha = torch.sin(alpha)
        case MaskInterpolation.RIGHT_COSINE:
            alpha = torch.linspace(0, 
                                   torch.pi / 2,
                                   steps=overlap_length)
            
            alpha = -torch.sin(alpha) + 1

        case MaskInterpolation.RIGHT_EXPONENTIAL:
            alpha = torch.linspace(0, 
                                   7, 
                                   steps=overlap_length)
            alpha = torch.pow(2, -alpha)

    return alpha