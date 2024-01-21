import cv2
import torch
import imageio
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch import Tensor
from itertools import compress, cycle
from collections import OrderedDict
from scipy.interpolate import griddata
from IPython.display import Image

from utilities.utils import *

from src.plotting import Plotter
from src.gl_solver import GLSolver
from src.parameters_init import ParametersInit
from src.random_input_field import RandomInputField

import warnings
warnings.filterwarnings('ignore')


def main(path,mtlibpath_prefix,Nx,Ny,Lx,Ly,T_end,dt):
    N_ITERATIONS = int(T_end / dt)
    global mem_rate
    global Nx = Nx
    
    A_norm, A_original, mem_rate, myu_original = compute_A_norm(
        Nx=Nx, 
        Ny=Ny, 
        input_to_defect_ratio=2*2, 
        mean=5.4, 
        std_deviation=0.8, 
        time_period=80, 
        Lx=Lx, 
        Ly=Ly, 
        dt=dt, 
        T_End=T_end, 
        parallel_runs=1, 
        input_scale=0.75, 
        mem_coef=1, 
        time_period_parameter=8, 
        _mean=5.4, 
        std_deviation_run_computation=0.8,
        input_myu=None
    )




    x = np.linspace(0, Lx, Nx).flatten()[:, None]
    y = np.linspace(0, Ly, Ny).flatten()[:, None]
    t = np.linspace(0, T_end, N_ITERATIONS).flatten()[:, None]

    Exact = A_original.squeeze(0)

    X, T, Y = np.meshgrid(x, t, y)

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()
    u_star = np.hstack([u_star.real[:, None],u_star.imag[:, None]])

    device='cpu'
    torch.manual_seed(0)
    net = GALAXYPINN([3,8,32,64,32,8,2]).to(device)

    L1 = net.fastrmsebatchtrain(x = X_star, y = u_star, epochs = 100000)
    net.optimizer.param_groups[0]['lr'] = 0.001
    L2 = net.fastrmsebatchtrain(x = X_star, y = u_star, epochs = 100000)    
    

    plt.plot(L1)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('Custom Loss')
    plt.title('Training of the AllInputNet \n lr=0.01')
    plt.savefig(f'{mtlibpath_prefix}_allinputnet001.png')    

    plt.plot(L2)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('Custom Loss')
    plt.title('Training of the AllInputNet \n lr=0.001')
    plt.savefig(f'{mtlibpath_prefix}_allinputnet0001.png')
    
    
    net.loaddata_precalculate(X_star)
    net.myureset()    
    
    figure, axes = plt.subplots(nrows = 2,ncols = 2, figsize=(8, 4.5))

    for lr, ax in zip([10,3,1,0.3],np.array(axes).flatten()):
        L = net.myutrain(lr = lr, epochs = 10)
        ax.plot(L)    
        ax.set_yscale('log')
        ax.set_title(f'lr={lr}')

    figure.text(0.02, 0.5, 'FMSE', ha='center', va='center', rotation='vertical')
    figure.text(0.5, 0.002, 'epochs', ha='center', va='center')
    figure.suptitle('MYU Training', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{mtlibpath_prefix}_myutraining.png')   
    
    
    
    ploter = Plotter(net.myu.cpu().detach().numpy())
    ploter.output_animation(mem_rate, save_gif=True, file_name=rf"{path}_myupred.gif")

    ploter = Plotter(myu_original.squeeze(0))
    ploter.output_animation(mem_rate, save_gif=True, file_name=rf"{path}_myuorig.gif")


    create_gifs(
        memory_rate=mem_rate,
        u_pred=net.predict(X_star),
        original=A_original,
        save=True,
        path_for_gif=path+".gif",
        duration=500,
        title=" "
    )
    Image(filename=path+".gif")


class MYULOSS:
  def __init__(self, x, y, t, net, verbose = 0):
      self.msef = nn.MSELoss()
      self.FMSE = []
      self.x = x
      self.y = y
      self.t = t
      self.y_pred = net.forward(torch.stack((x,y,t)).T)
        
  def plot(self, title= 'MYU training'):
    plt.plot(self.FMSE)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('FMSE')
    self.clear()
    
  def clear(self):
    self.FMSE = []
    
  def fmse(self, myu):
    f_loss = torch.mean(torch.abs(self.net_f(myu)) ** 2)
    self.FMSE.append(f_loss.cpu().detach().numpy())
    return f_loss
  
  def net_f(self, myu, verbose = 0):
        return self.f_withoutmyu - myu*self.u 
    
  def calculate_f_withoutmyu(self):
        torch.autograd.set_detect_anomaly(True)
        x,y,t = self.x, self.y, self.t
        
        ru = self.y_pred[:,0]
        iu = self.y_pred[:,1]

        (ru_t, ru_x, ru_y) = torch.autograd.grad(ru, (t, x, y), grad_outputs=torch.ones_like(ru), create_graph=True, retain_graph=True)
        (iu_t, iu_x, iu_y) = torch.autograd.grad(iu, (t, x, y), grad_outputs=torch.ones_like(iu), create_graph=True, retain_graph=True)

        (ru_xx,) = torch.autograd.grad(ru_x, (x), grad_outputs=torch.ones_like(ru_x), create_graph=True)
        (iu_xx,) = torch.autograd.grad(iu_x, (x), grad_outputs=torch.ones_like(iu_x), create_graph=True)

        (ru_yy,) = torch.autograd.grad(ru_y, (y), grad_outputs=torch.ones_like(ru_y), create_graph=True)
        (iu_yy,) = torch.autograd.grad(iu_y, (y), grad_outputs=torch.ones_like(iu_y), create_graph=True)

        u =( ru + iu * 1j).view(mem_rate, Nx, Ny)
        u_t = (ru_t + iu_t * 1j).view(mem_rate, Nx, Ny)
        u_xx =( ru_xx + iu_xx *1j).view(mem_rate, Nx, Ny)
        u_yy = (ru_yy + iu_yy *1j).view(mem_rate, Nx, Ny)
        self.u = u.cpu().detach()
            
        f_withoutmyu = u_t - (u_xx + u_yy) + torch.pow(torch.abs(u), 2)*u #- myu*u
        free_memory(u_t, u_xx, u_yy)
        self.f_withoutmyu = f_withoutmyu.cpu().detach()
        torch.autograd.set_detect_anomaly(False)
      
def free_memory(*variables):
    del variables
    torch.cuda.empty_cache()
    
class GALAXYNET(nn.Module):
    def __init__(self, layers_list, activation_function_list = None, linm = None):
        super(GALAXYNET, self).__init__()
        self._depth = len(layers_list) - 1
        
        if activation_function_list is None:
            activation_function_list = [F.softplus for _ in range(self._depth - 1)]
            
        if linm is None:
            linm =  np.tril(np.ones(self._depth + 1, dtype = int))
        lin = linm@layers_list
        
        self._activation_function_list = activation_function_list
        
        self._Wtmx = nn.Sequential(*[torch.nn.Linear(lin[i], layers_list[i+1], dtype = torch.float64) for i in range(self._depth)])
        self._linm = linm
        
        self.optimizer = torch.optim.Adam( params = self._Wtmx.parameters(), lr=0.01 )    
        
    def forward(self, x):
        layers = [x,self._Wtmx[0](x)]
        for i in range(1, self._depth):
            layers[i] = self._activation_function_list[i-1](layers[i])
            ind = self._linm[i]
            inpind = np.where(ind)[0]
            inp = torch.concat([layers[i] for i in inpind], dim=1)
            layers.append(self._Wtmx[i](inp))
        return layers[-1]        

    def predict(self, x):
        self._Wtmx.eval()
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype = torch.float64).to(device)
        y =  self.forward(x).cpu().detach().numpy()
        return y[:,0] + y[:,1]*1j    

    def rmsef(self, y, y_pred):
        mseloss = torch.sum((y_pred - y)**2,dim=1)
        return torch.sum(torch.sqrt(mseloss))   

    def fastrmsebatchtrain(self, x, y, epochs=100, batch_size = 64):
        from torch.utils.data import DataLoader

        x = torch.tensor(x, dtype = torch.float64).to(device)
        y = torch.tensor(y, dtype = torch.float64).to(device)
        dataloader = DataLoader(dataset = torch.hstack((x,y)), batch_size=batch_size, shuffle=True)
        
        self.optimizer.zero_grad()
        L = []
        
        import math
        batchiter = (epochs * batch_size)  // x.size()[0]
        epochs =  batchiter * math.ceil(x.size()[0] / batch_size)
        
        pbar = tqdm(total=epochs)
        try:
            for _ in range(batchiter):
                for i, tmp in enumerate(dataloader):
                    (tmpx, tmpy, tmpt, tmpu_real, tmpu_img) = tmp.T
                    X = torch.stack((tmpx,tmpy,tmpt)).T
                    U = torch.stack((tmpu_real, tmpu_img)).T

                    y_pred = self.forward(X)
                    loss = self.rmsef(y_pred,U)
                    L.append(loss.cpu().detach().numpy())
                    loss.backward()
                    self.optimizer.step()
                    self._Wtmx.zero_grad()
                    # Update the progress bar
                    pbar.update(1)
        except:
            pass
        finally:
            # Close the progress bar
            pbar.close()          
        return L       


class GALAXYPINN(GALAXYNET):
    def __init__(self,*args,**kwargs):
        super(GALAXYPINN, self).__init__(*args,**kwargs)
#         myu = torch.randn(4, 2, dtype=torch.float64).to(device)
#         myu = nn.Parameter(myu)
#         self._Wtmx.register_parameter('myu', myu)
#         self.myuparam = myu
#         myu = transform_and_stack(myu, 4, 200).to(device).clone().requires_grad_(True)
#         self.myu = myu.view(200, 4, 4)
        self.myureset()
    
    def myureset(self):
        myu = torch.randn(mem_rate, Nx, Ny, dtype=torch.float64).to(device)
        myu = torch.abs(myu)
        myu = nn.Parameter(myu)
        self._Wtmx.register_parameter('myu', myu)
        self.myuparam = myu
        self.myu = myu
    
    def loaddata_precalculate(self,x):
        x = torch.tensor(x, dtype = torch.float64, requires_grad=True).to(device)
        myuloss = MYULOSS(*tuple(x.T),self,y)
        myuloss.calculate_f_withoutmyu()
        self.myuloss = myuloss
    
    def myutrain(self, epochs=100, lr = 0.01):
        myuoptimizer = torch.optim.Adam( params = [self.myuparam], lr=lr ) 
        myuoptimizer.zero_grad()

        for _ in tqdm(range(epochs)):
            self.myuloss.fmse(self.myu).backward(retain_graph=True)
            myuoptimizer.step()
            myuoptimizer.zero_grad()
        
        FMSE = self.myuloss.FMSE
        self.myuloss.clear()
        return FMSE

