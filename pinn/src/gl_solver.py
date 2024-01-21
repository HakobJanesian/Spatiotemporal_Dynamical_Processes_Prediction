import numpy as np

from scipy import fft
from tqdm import tqdm

from src.parameters_init import ParametersInit
from src.random_input_field import RandomInputField


class GLSolver:
    
    def __init__(
        self,
        parameters: ParametersInit,
        random_input_field: RandomInputField,
        input_myu: np.ndarray = None
    ) -> None:
        
        self._parameters: ParametersInit = parameters
        self._random_input_field: RandomInputField = random_input_field
        
        self._A_hat: np.ndarray = np.zeros([
            self._parameters.Nx, 
            self._parameters.Ny, 
            self._parameters.parallel_runs
        ], dtype=complex)
        
        self._input_myu: np.ndarray = input_myu

    def get_paramters(self) -> ParametersInit:
        return self._parameters

    def set_parameters(self, parameters) -> None:
        self._parameters = parameters

    def non_linear_function(self, xx, yy):
        N_n = np.fft.fft2(yy * xx - xx * np.abs(xx)**2)
        return N_n

    def runge_kutta(self, xx, yy, qq):
        const = qq * self._parameters.dt
        coef1 = (np.exp(const)-1) / qq
        a_n = np.fft.fft2(xx) * np.exp(const) + self.non_linear_function(xx, yy) * coef1
        return a_n

    def gl_next_state(self, myu, A) -> np.ndarray:
        A = np.tile(A, [1, 1, int(self._parameters.parallel_runs/A.shape[2])])
        for k in range(self._parameters.parallel_runs):
            self._A_hat[:, :, k] = fft.fft2(A[:, :, k])
            self._parameters.q[:, :, k] = 10**(-6)-self._parameters.ksq**2
            a_n = self.runge_kutta(
                xx=A[:, :, k],
                yy=myu[:, :, k],
                qq=self._parameters.q[:, :, k]
            )
            coef1 = (
                np.exp(self._parameters.q[:, :, k]*self._parameters.dt)-1-self._parameters.dt *
                self._parameters.q[:, :, k])/(self._parameters.dt*self._parameters.q[:, :, k]**2
            )
            self._A_hat[:, :, k] = a_n + (
                self.non_linear_function(fft.ifft2(a_n), myu[:, :, k]) -
                self.non_linear_function(A[:, :, k], myu[:, :, k])
            ) * coef1
            A[:, :, k] = fft.ifft2(self._A_hat[:, :, k])
        return A

    def run_computation(
        self, 
        A=None, 
        time_period_parameter=80, 
        _mean=5.4, 
        std_deviation=0.8
    ) -> np.ndarray:
        np.random.seed(220)
        A_0 = A
        if A is None:
            A_0 = np.random.normal(
            loc=0,
            scale=1,
            size=(self._parameters.Nx, self._parameters.Ny, self._parameters.parallel_runs)
        )*0.01 + \
            np.random.normal(
            loc=0,
            scale=1,
            size=(self._parameters.Nx, self._parameters.Ny, self._parameters.parallel_runs)
        )*0.01j
            
        self.computing_helper(
            input=A_0,
            A_original=self._parameters.A_original,
            A_real_im=self._parameters.A_real_im,
            A_angle=self._parameters.A_angle,
            A_magnitude=self._parameters.A_magnitude,
            time_period_parameter=time_period_parameter,
            _mean=_mean,
            std_deviation=std_deviation
        )

        A_norm = self._parameters.A_real_im[0]/np.amax(self._parameters.A_real_im[0])
        A_norm = A_norm - np.amin(A_norm)
        return A_norm

    def computing_helper(
        self, 
        input,
        A_original,
        A_real_im, 
        A_angle, 
        A_magnitude,
        time_period_parameter=80,
        _mean=5.4,
        std_deviation=0.8
    ) -> None:
        time_period_set = set()
        for mean in range(len(self._parameters.MEAN)):
            A = input
            A_real_im[mean, :, :, :] = np.zeros([
                self._parameters.mem_rat,
                self._parameters.Nx,
                self._parameters.Ny
            ])
            for index in tqdm(range(self._parameters.N_ITERATIONS)):
                time_period = self._random_input_field.set_time_period(
                    index=index,
                    parameter=time_period_parameter
                )
                np.random.seed(time_period * 20) 
                if time_period not in time_period_set:
                    time_period_set.add(time_period)
                    if self._input_myu is None:
                        myu = self._random_input_field.set_myu(
                            mean=_mean,
                            std_deviation=std_deviation
                        )
                        myu = np.power(np.abs(myu), 2)
                        myu = myu.reshape(
                            int(np.sqrt(self._random_input_field.mm)), 
                            int(np.sqrt(self._random_input_field.mm)), 
                            self._parameters.parallel_runs
                        )
                        myu = np.kron(
                            a=myu, 
                            b=self._random_input_field.scale_matrix
                        )
                    else:
                        myu = self._input_myu
                
                A = self.gl_next_state(myu, A)
                self._parameters.myu_in[mean, index, :, :] = myu[:, :, 0]
                A_original[mean, index, :, :] = A[:, :, 0]
                A_real_im[mean, index, :, :] = A[:, :, 0].real*A[:, :, 0].imag
                ind1 = index // self._parameters.mem_coef
                if index % self._parameters.mem_coef == 0:
                    A_angle[mean, ind1, :, :] = np.angle(A[:, :, 0])
                    A_magnitude[mean, ind1, :, :] = (A[:, :, 0].real**2) + (A[:, :, 0].imag**2)
