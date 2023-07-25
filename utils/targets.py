import numpy as np 


class BasicTarget():
    
    def __init__(self, state_dim=1):
        
        self.state_dim = state_dim
        self.support = None
        
    
    def __call__(self,
                 state: np.ndarray,
                 temperature=1):
        
        assert self.state_dim == state.shape[0]
        pass
    
    
    
class MultiPeaksTarget(BasicTarget):
    
    def __init__(self,
                 seed=21,
                 peaks_num=4,
                 state_dim=1,
                 pos_range=[-5, 5],
                 var_range=[.01, 4]):
        
        super().__init__(
            state_dim=state_dim
        )
        
        np.random.seed(seed)
        
        # Randomly generate position of each peak.
        pos_min, pos_max = pos_range
        pos = np.random.uniform(
            low=pos_min, high=pos_max,
            size=(peaks_num, state_dim)         # [N, D].
        )
        
        # Randomly generate variance of each peak.
        var_min, var_max = var_range
        var = np.random.uniform(
            low=var_min, high=var_max,
            size=(peaks_num,)                   # [N,].
        )
        
        self.pos = pos
        self.var = var
        
        self.support = np.array(
            [pos_range for _ in range(state_dim)]
        )                                       # [D, 2].   
        
        
    def __call__(self, state, temperature=1):
        
        assert self.state_dim == state.shape[0]
        
        f_value = 0
        
        for var_v, pos_v in zip(self.var, self.pos):
            
            # import pdb; pdb.set_trace()
            _temp_value = np.exp( -np.linalg.norm(state - pos_v) ** 2 / var_v / 2 ) / var_v
            f_value += _temp_value
            
        return f_value ** temperature