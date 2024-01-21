import torch
import numpy as np

class BatchProcessing:
    def batch_predict(self, x, batch_size=65536, verbose=0, device = "cpu"):
        self.eval()
        
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32).to(device)

        num_samples = x.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        predictions = []

        if verbose == 1:
            iterator = tqdm(range(num_batches), desc="Batch Prediction")
        else:
            iterator = range(num_batches)

        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_x = x[start_idx:end_idx]
            batch_y = self.forward(batch_x).cpu().detach().numpy()

            predictions.append(batch_y)

        return np.concatenate(predictions, axis=0)[:, 0] + np.concatenate(predictions, axis=0)[:, 1] * 1j

    def msef(self, y, y_pred):
        return torch.mean((y_pred - y)**2,dim = 0)       


    def y_proc(y):
        MS = y[:,1] * y[:,0]
        IR = y[:,1] ** 2 +  y[:,0] ** 2
        return torch.vstack((IR, MS)).T
        
    def mse_batch_train(self, state, epochs = 100, batch_size = 64, verbose = 2, is_simulation = False, device = "cpu"):

        L = []
        for s in state.generate_batches(nbatches = epochs, batch_size = batch_size, verbose = verbose):
            self.optimizer.zero_grad()
            x = s.get_2d_tensor_xyt(device = device).T
            y = s.get_2d_tensor_state(device = device).T
            y_pred = self.forward(x)
            loss = self.msef(
                y_pred if is_simulation else BatchProcessing.y_proc(y_pred)
                ,y) 
            L.append(loss.cpu().detach().numpy())
            torch.mean(loss).backward()
            self.optimizer.step()

        return np.array(L)    

    def mse_train(self, state, epochs = 100, verbose = 2, is_simulation = False, device = "cpu"):

        L = []
        state = state.flatten()
        x = state.get_2d_tensor_xyt(device = device).T
        y = state.get_2d_tensor_state(device = device).T
        for s in state.generate_batches(nbatches = epochs, batch_size = 1, verbose = 1 if verbose == 2 else verbose):
            self.optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.msef(
                y_pred if is_simulation else BatchProcessing.y_proc(y_pred)
                ,y) 
            L.append(loss.cpu().detach().numpy())
            torch.mean(loss).backward()
            self.optimizer.step()

        return np.array(L)   


class SaveLoad:
    def save_model(self, file_path):
        model_state = {
            'state_dict': self.state_dict(),
        }
        torch.save(model_state, file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path, device='cpu'):
        model_state = torch.load(file_path, map_location=device)
        self.load_state_dict(model_state['state_dict'])
        print(f'Model loaded from {file_path}')

