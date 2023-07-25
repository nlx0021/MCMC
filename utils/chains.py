import numpy as np
from tqdm import tqdm

class BasicChain():
    
    def __init__(self,
                 init_state: np.ndarray,
                 kernal):
        
        '''
        Record the chain, state_dim and transition kernal.
        '''
        
        assert len(init_state.shape) == 1, "State should be a one-dimension array."
        
        self.chain = [init_state]
        self.state_dim = init_state.shape[0]
        self.cur_state = init_state
        self.kernal = kernal
        
    
    def _step(self):
        
        '''
        One transition.
        '''
        
        next_state = self.kernal(self.cur_state)
        self.chain.append(next_state)
        self.cur_state = next_state
        
        
    def __len__(self):
        
        return len(self.chain)
    
    
    def get_chain(self,
                  burn: int):
        
        return np.array(self.chain[burn:], dtype=np.float32)
    
    
    def get_cur_state(self):
        
        return self.cur_state
    
    
    def run(self,
            length: int):
        
        for _ in tqdm(range(length)):
            self._step()
    
    
    def reset(self,
              init_state):
        
        self.__init__(init_state, kernal=self.kernal)
        
        

class M_H_Chain(BasicChain):
    
    '''
    The same with BasicChain.
    '''
    
    def __init__(self,
                 init_state: np.ndarray,
                 kernal):
        
        super().__init__(
            init_state=init_state,
            kernal=kernal            
        )
        
        

class PT_M_H_Chain(BasicChain):
    
    '''
    Parallel Temperatures M-H Chain.
    '''
    
    def __init__(self,
                 init_state: np.ndarray,
                 kernal,
                 f_u,
                 temperatures=[.9, .8, .6, .3]):
        
        super().__init__(
            init_state=init_state,
            kernal=kernal
        )
        
        self.temperatures = [1] + temperatures
        self.B = len(temperatures)
        self.f_u = f_u
        
        # Complete state.
        comp_init_state = np.expand_dims(init_state, 0).repeat((self.B+1), axis=0)  # [B+1, D]
        self.comp_chain = [comp_init_state]
        self.comp_cur_state = comp_init_state
        self.temp_idx = 0
        
        
    def _step(self):
        
        '''
        Parallel transition.
        '''
        
        comp_cur_state = self.comp_cur_state
        comp_next_state = np.zeros_like(comp_cur_state)
        
        # parallel M-H transition.
        for idx in range(self.B+1):
            comp_next_state[idx] = self.kernal(
                comp_cur_state[idx], temperature=self.temperatures[idx]
            )
            
        # Switch temperature.
        temp_idx = self.temp_idx
        f_u = self.f_u
        
        next_temp_idx = np.clip(
            temp_idx + np.random.choice([-1, 1], p=[.5, .5]),
            a_min=0, a_max=self.B
        )
        
        alpha = min(
            1,
            f_u(comp_cur_state[temp_idx], temperature=next_temp_idx) * f_u(comp_cur_state[next_temp_idx], temperature=temp_idx) / \
            f_u(comp_cur_state[next_temp_idx], temperature=next_temp_idx) / f_u(comp_cur_state[temp_idx], temperature=temp_idx)
        )
        
        if np.random.random() < alpha:    # Switch.
            comp_next_state[[temp_idx, next_temp_idx], ...] = comp_next_state[[next_temp_idx, temp_idx], ...]
            
            
        self.comp_chain.append(comp_next_state)
        self.comp_cur_state = comp_next_state
        
        self.chain.append(comp_next_state[0])
        self.cur_state = comp_next_state[0]          