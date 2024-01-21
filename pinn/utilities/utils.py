import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Optional
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
