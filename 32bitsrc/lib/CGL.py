      
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
def free_memory(*variables):
    del variables
    torch.cuda.empty_cache()


class MYULOSS:
  def net_f(net, state):
          myu = state.myu
          x,y,t = state.get_2d_tensor_xyt(requires_grad=True, device = net.device)

          pred = net.forward(torch.stack((x,y,t)).T)
          ru,iu = pred.T
          (ru_t, ru_x, ru_y) = torch.autograd.grad(ru, (t, x, y), grad_outputs=torch.ones_like(ru), create_graph=True, retain_graph=True)
          (iu_t, iu_x, iu_y) = torch.autograd.grad(iu, (t, x, y), grad_outputs=torch.ones_like(iu), create_graph=True, retain_graph=True)

          (ru_xx,) = torch.autograd.grad(ru_x, (x), grad_outputs=torch.ones_like(ru_x), create_graph=True)
          (iu_xx,) = torch.autograd.grad(iu_x, (x), grad_outputs=torch.ones_like(iu_x), create_graph=True)

          (ru_yy,) = torch.autograd.grad(ru_y, (y), grad_outputs=torch.ones_like(ru_y), create_graph=True)
          (iu_yy,) = torch.autograd.grad(iu_y, (y), grad_outputs=torch.ones_like(iu_y), create_graph=True)

          u = ( ru + iu * 1j)
          u_t = (ru_t + iu_t * 1j)
          u_xx =( ru_xx + iu_xx *1j)
          u_yy = (ru_yy + iu_yy *1j)

          f =  u_t - (u_xx + u_yy) + torch.pow(torch.abs(u), 2) * u - u * myu
          free_memory(u_t, u_xx, u_yy, x, y, t)
          
          return f, pred
  
  class MYUCACHE:
    def __init__(self, net, state, verbose = 0):
        self.msef = nn.MSELoss()
        self.FMSE = []
        self.state = state
        self.net = net

        
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

    MYU_BATCH_SIZE = 4096
    def calculate_f_withoutmyu(self, batch_size = MYU_BATCH_SIZE):
          u, u_t,u_xx,u_yy = MYULOSS.MYUCACHE.pref(self.net, self.state, batch_size = batch_size)
          
          self.u = u.detach()
              
          f_withoutmyu = u_t - (u_xx + u_yy) + torch.pow(torch.abs(u), 2)*u #- myu*u
          self.f_withoutmyu = f_withoutmyu.detach()
          free_memory(u_t, u_xx, u_yy, u, f_withoutmyu)

      
    def f_withoutmyu(xyt,ru,iu):
          
          x, y, t = xyt
          (ru_t, ru_x, ru_y) = torch.autograd.grad(ru, (t, x, y), grad_outputs=torch.ones_like(ru), create_graph=True, retain_graph=True)
          (iu_t, iu_x, iu_y) = torch.autograd.grad(iu, (t, x, y), grad_outputs=torch.ones_like(iu), create_graph=True, retain_graph=True)

          (ru_xx,) = torch.autograd.grad(ru_x, (x), grad_outputs=torch.ones_like(ru_x), create_graph=True)
          (iu_xx,) = torch.autograd.grad(iu_x, (x), grad_outputs=torch.ones_like(iu_x), create_graph=True)

          (ru_yy,) = torch.autograd.grad(ru_y, (y), grad_outputs=torch.ones_like(ru_y), create_graph=True)
          (iu_yy,) = torch.autograd.grad(iu_y, (y), grad_outputs=torch.ones_like(iu_y), create_graph=True)

          u = (ru + iu * 1j)
          u_t = (ru_t + iu_t * 1j)
          u_xx =( ru_xx + iu_xx *1j)
          u_yy = (ru_yy + iu_yy *1j)
          
          return u, u_t,u_xx,u_yy
    
    def pref(net, state, batch_size = MYU_BATCH_SIZE):
      cache = {
          'u':[],
          'u_t':[],
          'u_xx':[],
          'u_yy':[],
      }
      for s in state.iterate_over_data(batch_size = batch_size, shuffle = False, verbose = 1):
          x,y,t = s.get_2d_tensor_xyt(requires_grad = True, device = net.device)
          ru,iu = net.forward(torch.stack((x,y,t)).T).T
          u, u_t,u_xx,u_yy = MYULOSS.MYUCACHE.f_withoutmyu((x,y,t),ru,iu)
          cache['u'].append(u.detach())
          cache['u_t'].append(u_t.detach())
          cache['u_xx'].append(u_xx.detach())
          cache['u_yy'].append(u_yy.detach())
          
      return  torch.cat(cache['u']).view(*state.shape), \
              torch.cat(cache['u_t']).view(*state.shape),\
              torch.cat(cache['u_xx']).view(*state.shape),\
              torch.cat(cache['u_yy']).view(*state.shape)

