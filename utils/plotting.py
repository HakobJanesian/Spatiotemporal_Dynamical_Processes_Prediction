import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from tqdm import tqdm


class Plotter:

    def __init__(self, data=None, figsize=(14, 12)) -> None:
        self.data = data
        self.fig = plt.figure(1, figsize=figsize)
        self.anim_fig = plt.figure(figsize=figsize)

    def get_data(self):
        return self.data
    
    def set_data(self, data):
        self.data = data
    
    def output_plot(self, mem_rat, save_fig=False, file_name="A_in_norm_80%04d.png"):
        for index in tqdm(range(mem_rat)):
            if index % 2 == 0:
                fig = self.fig
                plt.clf()
                plt.imshow(self.data[index, :, :], vmax=0.9*np.max(self.data))
                plt.colorbar()
                if save_fig:
                    filename = file_name % (index/2)
                    plt.savefig(filename)

    def output_animation(self, mem_rate, save_gif=False, file_name=r"animation.gif"):
        fig = plt.figure()
        im = plt.imshow(self.data[0, :, :])
        plt.colorbar()
        aa = 1

        def LC_update(i):
            plt.clf()
            im = plt.imshow(
                np.abs(self.data[aa*i, :, :]), vmax=0.8*np.max(self.data)
            )
            plt.colorbar()
            plt.title('frame='+str(i))
            return [im]

        plt.xlabel('Re(A)*Im(A)')
        anim = animation.FuncAnimation(
            fig=fig, 
            func=LC_update, 
            frames=mem_rate
        )
        if save_gif:
            anim.save(file_name, writer=PillowWriter(fps=30))

    def plot_curve(self, data):
        fig = self.fig
        plt.plot(data)
