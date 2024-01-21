import warnings
import numpy as np

class StatisticsUtils:
    
    def __init__(self, A_in_norm, A_in_norm1, mem_rate, number_of_iterations) -> None:
        self._A_in_norm = A_in_norm
        self._A_in_norm1 = A_in_norm1
        self._mem_rat = mem_rate
        self._number_of_iterations = number_of_iterations

    def corelate_stepped(self, A_in, step, number_of_iterations, next_step = 2):
        if next_step + step > number_of_iterations:
            warnings.showwarning(
                f"The value of the number of iterations ({number_of_iterations}) is larger \n \
                than the sum of the step ({step}) and the next step ({next_step}). \
                The value of the next value changed to 2 automatically."
            )
            next_step = 2
        return np.corrcoef(A_in[step, :, :].reshape([-1]), A_in[next_step + step, :, :].reshape([-1]))[0][1]

    def compute_stepped_corelation(self, step = 10):
        cor = []
        for i in range(self._mem_rat - step):
            cor.append(
                self.corelate_stepped(
                    A_in = self._A_in_norm,
                    step = i,
                    number_of_iterations = self._number_of_iterations,
                    next_step = 2
            ))
        return cor

    def corelate_all(self, A_in, A_in1, step):
        return np.corrcoef(A_in[step, :, :].reshape([-1]), A_in1[step, :, :].reshape([-1]))[0][1]

    def compute_corelation_for_all(self):
        cor_all = []
        for i in range(self._mem_rat):
            cor_all.append(
                self.corelate_all(
                    A_in = self._A_in_norm,
                    A_in1 = self._A_in_norm1,  
                    step = i
            ))
        return cor_all

    def normal_difference(self):
        norm = []
        for i in range(self._mem_rat):
            norm.append(
                np.linalg.norm(
                    self._A_in_norm[i, :, :]-self._A_in_norm1[i, :, :]
            ))
        return norm