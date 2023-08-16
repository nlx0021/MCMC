import numpy as np
import matplotlib.pyplot as plt


from .chains import BasicChain
from .targets import BasicTarget

class Visualizer():
    
    def __init__(self,
                 f_u: BasicTarget,
                 chain: BasicChain):
        
        self.f_u = f_u
        self.chain = chain
        
        
    def vis(self,
            burn=1000,
            s=.1):
        
        chain = self.chain
        f_u = self.f_u
        states = chain.get_chain(burn)
        
        assert chain.state_dim in [1, 2, 3], "Only support dim below 3."
        
        if chain.state_dim == 1:
            raise NotImplementedError
        
        elif chain.state_dim == 2:
            
            x, y = states[:, 0], states[:, 1]
            time = np.arange(x.shape[0])
            
            x_min, x_max = int(x.min())-1, int(x.max())+1
            y_min, y_max = int(y.min())-1, int(y.max())+1
            sup_x_min, sup_x_max = f_u.support[0]
            sup_y_min, sup_y_max = f_u.support[1]
            
            x_min, y_min = min(x_min, sup_x_min)-1, min(y_min, sup_y_min)-1
            x_max, y_max = max(x_max, sup_x_max)+1, max(y_max, sup_y_max)+1
            
            grid_step = min(
                (x_max-x_min) / 500,
                (y_max-y_min) / 500
            )
            
            grid_x, grid_y = np.mgrid[x_min:x_max:grid_step, y_min:y_max:grid_step]

            grid_z = np.array(
                [f_u(np.array([_x, _y])) for _x, _y in zip(grid_x.reshape(-1,), grid_y.reshape(-1,))]
            ).reshape(grid_x.shape)
            
            plt.contourf(
                grid_x, grid_y, grid_z,
                levels=15,origin='upper'
            )
            
            plt.scatter(
                x=x, y=y, s=s,
                # c=time
            )
            plt.show()
            
        else:
            raise NotImplementedError