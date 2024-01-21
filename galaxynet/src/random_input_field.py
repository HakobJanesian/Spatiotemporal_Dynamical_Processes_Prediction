import numpy as np


class RandomInputField:
    
    def __init__(
        self, 
        Nx: int=320, 
        Ny: int=320, 
        input_to_defect_ratio: int=16*16, 
        mean: float=5.4, 
        std_deviation: float=0.8, 
        time_period: float=80
    ) -> None:
        
        self.Nx: int = Nx
        self.Ny: int = Ny
        
        self.input_to_defect_ratio: int = input_to_defect_ratio
        
        self.mean: float = mean
        self.std_deviation: float = std_deviation
        self.time_period: float = time_period
        
        self.mm: int = int(Nx*Ny/input_to_defect_ratio)

        self.scale_matrix: np.ndarray = np.ones(
            shape=(
                int(np.sqrt(input_to_defect_ratio)),
                int(np.sqrt(input_to_defect_ratio)),
                1
        ))

        self.scale_matrix = np.ones(
            shape=(
                int(np.sqrt(self.input_to_defect_ratio)),
                int(np.sqrt(self.input_to_defect_ratio)),
                1
        ))
        
        self.myu = self.set_myu(
            mean=self.mean,
            std_deviation=self.std_deviation
        )
        
        self.time_period = self.set_time_period(
            index=1,
            parameter=80
        )

    def set_myu(self, mean, std_deviation):
        totimp = (
            np.random.normal(mean, std_deviation, size=(self.mm, 1)) +
            np.random.normal(mean, std_deviation, size=(self.mm, 1)) * 1j
        )
        return totimp

    def set_time_period(self, index, parameter):
        self.time_period = index // parameter
        return self.time_period
