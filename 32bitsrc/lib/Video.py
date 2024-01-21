import matplotlib.pyplot as plt
import numpy as np
import cv2

def process_subframe(fig, ax, m, title):
    
    im = ax.imshow(m)
    ax.title.set_position([.5, 1.05])
    im = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    
def process_frame(M, titles):
    figsize = (M.shape[0], M.shape[1])
    fig, axs = plt.subplots(*figsize, figsize=(18, 6))
    
    for i in range(figsize[0]):
        for j in range(figsize[1]):
            ax = axs[i][j]
            process_subframe(fig,axs[i][j], M[i][j],titles[i][j])
            
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    
    return image   

def create_video(M,titles, videotitle, save=True, fps=30):
    
    images = [process_frame(m,titles) for m in M]
    #Parallel(n_jobs=num_cores)(delayed(process_frame)(m, titles) for m in tqdm(M))
    if save:
        # Convert images to 8-bit color for video
        images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        height, width, _ = images[0].shape
        video = cv2.VideoWriter(videotitle, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for img in images:
            video.write(img)
        video.release()
        print("Video successfully saved at", videotitle)
    
    return None