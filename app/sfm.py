import numpy as np
from data import dataset
import yaml
from tqdm import tqdm
class SFM():
    def __init__(self):
        self.file_path=r"../params.yaml"
        self.get_params()
        self.data = dataset(self.data_path)
        self.num_frame=self.data.num_frame
        # init matrices
        self.init_matrices()

    def get_params(self,):
        try:
            with open(self.file_path, 'r') as yaml_file:
                self.params = yaml.safe_load(yaml_file)
                self._set_attributes()
            print("Parameters loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"Error loading parameters: {e}")

    def _set_attributes(self):
        if self.params:
            for key, value in self.params.items():
                setattr(self, key, value)

    def init_matrices(self):
        self.K = self.data.K
        self.R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.R_t_1 = np.empty((3, 4))
        self.P1 = np.matmul(self.K, self.R_t_0)
        self.P2 = np.empty((3, 4))



if __name__ == "__main__":
    sfm=SFM()
