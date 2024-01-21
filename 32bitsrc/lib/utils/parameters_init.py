from __future__ import annotations

import numpy as np
from scipy import fft


class ParametersInit:

    def __init__(
        self,
        Lx: int = 100,
        Ly: int = 100,
        Nx: int = 320,
        Ny: int = 320,
        dt: float = 0.01,
        T_End: int = 10,
        parallel_runs: int = 1,
        input_scale: float = 0.75,
        mem_coef: int = 20
    ) -> None:

        self.Lx = Lx
        self.Ly = Ly

        self.Nx = Nx
        self.Ny = Ny

        self.dt = dt
        self.T_End = T_End

        self.dx = self.Lx/Nx
        self.dy = self.Ly/Ny

        self.N_ITERATIONS = int(self.T_End/self.dt)
        self.parallel_runs = parallel_runs
        
        self.input_scale: float = input_scale
        self.mem_coef: int = mem_coef
        self.MEAN: np.ndarray = np.arange(7, 7.5, 0.5)

        self.mem_rat = int(self.N_ITERATIONS / self.mem_coef)

        # Meshgrid of real an freq spacies
        # self.x, self.y = np.meshgrid(np.arange(self.Nx) * self.Lx/self.Nx, np.arange(self.Ny) * self.Ly/self.Ny)
        self.KX, self.KY = np.meshgrid(np.fft.fftfreq(self.Nx, self.Lx/(self.Nx*2.0*np.pi)), np.fft.fftfreq(self.Ny, self.Ly/(self.Ny*2.0*np.pi)))

        # Wave vector
        self.ksq = self.KX**2 + self.KY**2
        self.fq = np.zeros([self.Nx, self.Ny, self.parallel_runs])
        self.q = np.zeros([self.Nx, self.Ny, self.parallel_runs])
        
        # Matricies
        self.A_original = np.zeros([len(self.MEAN), self.mem_rat, self.Nx, self.Ny], dtype=np.complex64)
        self.A_real_im = np.zeros([len(self.MEAN), self.mem_rat, self.Nx, self.Ny])
        self.myu_in = np.zeros([len(self.MEAN), self.N_ITERATIONS, self.Nx, self.Ny])
        self.A_magnitude = np.zeros([len(self.MEAN), self.mem_rat, self.Nx, self.Ny])
        self.A_angle = np.zeros([len(self.MEAN), self.mem_rat, self.Nx, self.Ny])
