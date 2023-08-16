import numpy as np
from tqdm import tqdm

class BasicChain():
    
    def __init__(self,
                 f_u,
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
        self.f_u = f_u
        
        self.ct = 0
        
    
    def _step(self):
        
        '''
        One transition.
        '''
        self.ct += 1
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
                 f_u,
                 init_state: np.ndarray,
                 kernal):
        
        super().__init__(
            init_state=init_state,
            kernal=kernal,
            f_u=f_u         
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
            kernal=kernal,
            f_u=f_u
        )
        
        self.temperatures = [1] + temperatures
        self.B = len(temperatures)
        
        # Complete state.
        comp_init_state = np.expand_dims(init_state, 0).repeat((self.B+1), axis=0)  # [B+1, D]
        self.comp_chain = [comp_init_state]
        self.comp_cur_state = comp_init_state
        
        self.switch_n = 0
        self.ct = 0
        self.switch_ratio = 0
        
        if 'is_dual' in dir(kernal):

            assert kernal.is_dual is False, "Not implement of dual average for PT chain."
        
        
    def _step(self):
        
        '''
        Parallel transition.
        '''
        
        self.ct += 1 
        
        comp_cur_state = self.comp_cur_state
        comp_next_state = np.zeros_like(comp_cur_state)
        temperatures = self.temperatures
        
        # parallel M-H transition.
        for idx in range(self.B+1):
            comp_next_state[idx] = self.kernal(
                comp_cur_state[idx], temperature=temperatures[idx]
            )
            
        # Switch temperature.
        f_u = self.f_u
        
        next_temp_idx = np.random.choice(
            range(self.B+1)
        )
        
        alpha = min(
            1,
            f_u(comp_cur_state[next_temp_idx], temperature=temperatures[0]) /
            f_u(comp_cur_state[0], temperature=temperatures[0]) 
        )
        
        if np.random.random() < alpha:    
            comp_next_state[0, ...] = comp_next_state[next_temp_idx, ...]
            self.switch_n += 1
            self.switch_ratio = self.switch_n / self.ct
            
            
        self.comp_chain.append(comp_next_state)
        self.comp_cur_state = comp_next_state
        
        self.chain.append(comp_next_state[0])
        self.cur_state = comp_next_state[0]         