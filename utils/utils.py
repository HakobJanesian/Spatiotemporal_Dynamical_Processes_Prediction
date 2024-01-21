from __future__ import annotations
import cv2
import torch
import imageio
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Optional
from joblib import Parallel, delayed
from torch.nn.parameter import Parameter
from IPython.display import display, Image


def transform_and_stack(input_obj, size, stack_times, to_parameter=False):
    """
    Function to transform an input tensor or a parameter to a larger tensor or parameter with the same elements.

    Args:
        input_obj (torch.Tensor or torch.nn.Parameter): Input tensor or parameter to transform.
        size (int): The height and width of the output matrices in the stacked tensor.
        stack_times (int): The number of times to stack each output matrix in the third dimension.
        to_parameter (bool, optional): If True, wraps the final result into a torch.nn.Parameter. 
            Default is False, which returns a plain tensor.

    Returns:
        torch.Tensor or torch.nn.Parameter: The transformed tensor or parameter.
    """

    # Initialize the result as an empty list.
    result = []

    # Check if the input is a Parameter. If it is, use its data (which is a tensor).
    # If the input is a tensor, use it as is.
    tensor = input_obj.data if isinstance(input_obj, Parameter) else input_obj

    # Loop over the first and second dimensions of the tensor.
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            # Create a matrix filled with the [i, j] element of the tensor.
            matrix = torch.full((size, size), tensor[i, j].item())
            # Stack the matrix `stack_times` times along the third dimension.
            stacked_matrix = torch.stack([matrix]*stack_times)
            # Append the stacked matrix to the result list.
            result.append(stacked_matrix)

    # Concatenate all the stacked matrices in the result list along the first dimension.
    result = torch.cat(result)

    # If to_parameter is True, wrap the result tensor in a Parameter.
    if to_parameter:
        result = Parameter(result)

    return result


def create_gifs(memory_rate: int, u_pred: np.ndarray, original: np.ndarray, save: bool=False, 
                path_for_gif: str="animation.gif", duration: int=20, title: str="Title") -> Optional[Image] | None:
    """
    Function to create and save or display GIF animations from given 2D numpy arrays.

    Args:
        memory_rate (int): Number of frames for the gif, reshaping the input arrays into this dimension.
        u_pred (np.ndarray): Predicted data used to create the gif.
        original (np.ndarray): Original data used to create the gif.
        save (bool, optional): Flag indicating whether to save the gif to a file. Default is False (will display in Jupyter Notebook).
        path_for_gif (str, optional): Path for saving the gif. Default is 'animation.gif'.
        duration (int, optional): Duration of each frame in the gif in seconds. Default is 20.
        title (str, optional): Title for the plots. Default is 'Title'.

    Returns:
        Image: If save is False, returns an Image object to display in Jupyter Notebook.
        None: If save is True, the function saves the gif to the given path and returns None.
    """

    images: List[np.ndarray] = []  # List of images to be compiled into a GIF
    
    # Create complex image data from u_pred and original
    u_im_real = (u_pred.reshape(memory_rate, original.shape[2], original.shape[3]).imag * 
                 u_pred.reshape(memory_rate, original.shape[2], original.shape[3]).real)
    
    o_im_real = (original.reshape(memory_rate, original.shape[2], original.shape[3]).imag * 
                 original.reshape(memory_rate, original.shape[2], original.shape[3]).real)
    
    # Loop over the frames
    for index in tqdm(range(memory_rate)):
        if index % 2 == 0:
            fig, axs = plt.subplots(1, 2, figsize=(12, 12))
            
            # Display the u_pred image
            im1 = axs[0].imshow(u_im_real[index])
            axs[0].set_title(title + " (u_pred) without normalization" + f" - Frame: {index}")
            axs[0].title.set_position([.5, 1.05])
            
            # Display the original image
            im2 = axs[1].imshow(o_im_real[index])
            axs[1].set_title(title + " (original) without normalization" + f" - Frame: {index}")
            axs[1].title.set_position([.5, 1.05])
            
            # Add colorbars
            fig.colorbar(im1, ax=axs[0])
            fig.colorbar(im2, ax=axs[1])
            
            # Draw the figure and store the image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            images.append(image)

            plt.close(fig)

    # Save or display the GIF
    if save:
        imageio.mimsave(path_for_gif, images, duration=duration)
        print("GIF successfully saved at", path_for_gif)
        return None
    else:
        # Display the animation in Jupyter Notebook
        gif_image = imageio.mimsave(imageio.RETURN_BYTES, images, format='GIF', duration=duration)
        return Image(data=gif_image)
    
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Optional
from torch.nn.parameter import Parameter
from IPython.display import display, Image

from .gl_solver import GLSolver
from .parameters_init import ParametersInit
from .random_input_field import RandomInputField


# def transform_and_stack(input_obj, size, stack_times, to_parameter=False):
#     """
#     Function to transform an input tensor or a parameter to a larger tensor or parameter with the same elements.

#     Args:
#         input_obj (torch.Tensor or torch.nn.Parameter): Input tensor or parameter to transform.
#         size (int): The height and width of the output matrices in the stacked tensor.
#         stack_times (int): The number of times to stack each output matrix in the third dimension.
#         to_parameter (bool, optional): If True, wraps the final result into a torch.nn.Parameter. 
#             Default is False, which returns a plain tensor.

#     Returns:
#         torch.Tensor or torch.nn.Parameter: The transformed tensor or parameter.
#     """

#     # Initialize the result as an empty list.
#     result = []

#     # Check if the input is a Parameter. If it is, use its data (which is a tensor).
#     # If the input is a tensor, use it as is.
#     tensor = input_obj.data if isinstance(input_obj, Parameter) else input_obj

#     # Loop over the first and second dimensions of the tensor.
#     for i in range(tensor.shape[0]):
#         for j in range(tensor.shape[1]):
#             # Create a matrix filled with the [i, j] element of the tensor.
#             matrix = torch.full((size, size), tensor[i, j].item())
#             # Stack the matrix `stack_times` times along the third dimension.
#             stacked_matrix = torch.stack([matrix]*stack_times)
#             # Append the stacked matrix to the result list.
#             result.append(stacked_matrix)

#     # Concatenate all the stacked matrices in the result list along the first dimension.
#     result = torch.cat(result)

#     # If to_parameter is True, wrap the result tensor in a Parameter.
#     if to_parameter:
#         result = Parameter(result)

#     return result

def transform_and_stack(input_obj, size, stack_times):
    # Extend input_obj to the desired size along the first two dimensions.
    input_extended = input_obj.repeat(size // input_obj.shape[0], size // input_obj.shape[1])
    
    # Extend input_extended to the desired size along the third dimension.
    result = input_extended.unsqueeze(-1).repeat(1, 1, stack_times)
    
    # Transpose the result to bring the stack_times dimension to the second position.
    result = result.view(size, size, stack_times)
    
    return result


def create_gifs(memory_rate: int, u_pred: np.ndarray, original: np.ndarray, save: bool=False, 
                path_for_gif: str="animation.gif", duration: int=20, title: str="Title") -> Optional[Image] | None:
    """
    Function to create and save or display GIF animations from given 2D numpy arrays.

    Args:
        memory_rate (int): Number of frames for the gif, reshaping the input arrays into this dimension.
        u_pred (np.ndarray): Predicted data used to create the gif.
        original (np.ndarray): Original data used to create the gif.
        save (bool, optional): Flag indicating whether to save the gif to a file. Default is False (will display in Jupyter Notebook).
        path_for_gif (str, optional): Path for saving the gif. Default is 'animation.gif'.
        duration (int, optional): Duration of each frame in the gif in seconds. Default is 20.
        title (str, optional): Title for the plots. Default is 'Title'.

    Returns:
        Image: If save is False, returns an Image object to display in Jupyter Notebook.
        None: If save is True, the function saves the gif to the given path and returns None.
    """

    images: List[np.ndarray] = []  # List of images to be compiled into a GIF
    
    # Create complex image data from u_pred and original
    u_im_real = (u_pred.reshape(memory_rate, original.shape[2], original.shape[3]).imag * 
                 u_pred.reshape(memory_rate, original.shape[2], original.shape[3]).real)
    
    o_im_real = (original.reshape(memory_rate, original.shape[2], original.shape[3]).imag * 
                 original.reshape(memory_rate, original.shape[2], original.shape[3]).real)
    
    # Loop over the frames
    for index in tqdm(range(memory_rate)):
        if index % 2 == 0:
            fig, axs = plt.subplots(1, 2, figsize=(12, 12))
            
            # Display the u_pred image
            im1 = axs[0].imshow(u_im_real[index])
            axs[0].set_title(title + " (u_pred) without normalization" + f" - Frame: {index}")
            axs[0].title.set_position([.5, 1.05])
            
            # Display the original image
            im2 = axs[1].imshow(o_im_real[index])
            axs[1].set_title(title + " (original) without normalization" + f" - Frame: {index}")
            axs[1].title.set_position([.5, 1.05])
            
            # Add colorbars
            fig.colorbar(im1, ax=axs[0])
            fig.colorbar(im2, ax=axs[1])
            
            # Draw the figure and store the image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            images.append(image)

            plt.close(fig)

    # Save or display the GIF
    if save:
        imageio.mimsave(path_for_gif, images, duration=duration)
        print("GIF successfully saved at", path_for_gif)
        return None
    else:
        # Display the animation in Jupyter Notebook
        gif_image = imageio.mimsave(imageio.RETURN_BYTES, images, format='GIF', duration=duration)
        return Image(data=gif_image)



def compute_A_norm(*, 
                   Nx=4, 
                   Ny=4, 
                   input_to_defect_ratio=2*2, 
                   mean=5.4, 
                   std_deviation=0.8, 
                   time_period=25, 
                   Lx=30, 
                   Ly=30, 
                   dt=0.005, 
                   T_End=1, 
                   parallel_runs=1, 
                   input_scale=0.75, 
                   mem_coef=1, 
                   time_period_parameter=100, 
                   _mean=5.4, 
                   std_deviation_run_computation=1,
                   input_myu=None):

    random_input_field: RandomInputField = RandomInputField(
        Nx=Nx,
        Ny=Ny,
        input_to_defect_ratio=input_to_defect_ratio,
        mean=mean,
        std_deviation=std_deviation,
        time_period=time_period
    )

    parameters: ParametersInit = ParametersInit(
        Lx=Lx,
        Ly=Ly,
        Nx=random_input_field.Nx,
        Ny=random_input_field.Ny,
        dt=dt,
        T_End=T_End,
        parallel_runs=parallel_runs,
        input_scale=input_scale,
        mem_coef=mem_coef
    )

    gl_solver: GLSolver = GLSolver(
        parameters=parameters,
        random_input_field=random_input_field,
        input_myu=input_myu
    )

    A_norm = gl_solver.run_computation(
        time_period_parameter=time_period_parameter, 
        _mean=_mean, 
        std_deviation=std_deviation_run_computation
    )
    
    print("Unique Myus count\t", np.count_nonzero(np.unique(parameters.myu_in[:, :, :])))
    unique_values, counts = np.unique(parameters.myu_in, return_counts=True)
    print("Max value of myu:\t", np.max(parameters.myu_in[:, :, :]))
    print("Min value of myu:\t", np.min(parameters.myu_in[:, :, :]))
    print("Unique values:", (unique_values.tolist()))
    print("Counts:\t\t", counts)
    print(f"A.shape={parameters.A_original[:, :, :].shape},\nMyu.shape={parameters.myu_in[:, :, :].shape},\n")
    print("Any NaN values in Myu\t\t", np.isnan(parameters.myu_in).any())
    print("Any NaN values in A_original\t", np.isnan(parameters.A_original).any())

    return A_norm, parameters.A_original, parameters.mem_rat, parameters.myu_in


def process_frame(index, u_im_real, o_im_real, title):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display the u_pred image
    im1 = axs[0].imshow(u_im_real[index])
    axs[0].set_title(title + " (u_pred) without normalization" + f" - Frame: {index}")
    axs[0].title.set_position([.5, 1.05])
    
    # Display the original image
    im2 = axs[1].imshow(o_im_real[index])
    axs[1].set_title(title + " (original) without normalization" + f" - Frame: {index}")
    axs[1].title.set_position([.5, 1.05])
    
    # Display the difference image
    im3 = axs[2].imshow(np.abs(u_im_real[index] - o_im_real[index]))
    axs[2].set_title(title + " (difference) without normalization" + f" - Frame: {index}")
    axs[2].title.set_position([.5, 1.05])
    
    # Add colorbars
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
    
    # Draw the figure and store the image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    
    return image

def create_video(memory_rate: int, u_pred: np.ndarray, original: np.ndarray, save: bool=False, 
                path_for_video: str="animation.mp4", fps: int=30, title: str="Title") -> Optional[Image] | None:
    # Create complex image data from u_pred and original
    u_im_real = (u_pred.reshape(memory_rate, original.shape[2], original.shape[3]).imag * 
                 u_pred.reshape(memory_rate, original.shape[2], original.shape[3]).real)
    
    o_im_real = (original.reshape(memory_rate, original.shape[2], original.shape[3]).imag * 
                 original.reshape(memory_rate, original.shape[2], original.shape[3]).real)
    
    # Use multiprocessing to process frames in parallel
    num_cores = multiprocessing.cpu_count()
    images = Parallel(n_jobs=num_cores)(delayed(process_frame)(index, u_im_real, o_im_real, title) for index in tqdm(range(memory_rate)))

    # Save or display the video
    if save:
        # Convert images to 8-bit color for video
        images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        height, width, _ = images[0].shape
        video = cv2.VideoWriter(path_for_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for img in images:
            video.write(img)
        video.release()
        print("Video successfully saved at", path_for_video)
        return None
    else:
        # Display the animation in Jupyter Notebook
        # Note: Displaying video in Jupyter Notebook is more complex than displaying GIFs
        print("Displaying video in Jupyter Notebook is not supported in this function.")
        return None

# How to use new method
# Uncomment this in the main code
# create_video(
#     memory_rate=mem_rate,
#     u_pred=net.predict(X_star),
#     original=A_original,
#     save=True,
#     path_for_video=path+"chaotic_test.mp4",
#     fps=30,
#     title=" "
# )


if __name__ == "__main__":
    myu = torch.rand(2, 2)
    print(myu)
    transformed_myu = transform_and_stack(myu, 4, 200).clone().detach().requires_grad_(True)
    print(transformed_myu.view(200, 4, 4))