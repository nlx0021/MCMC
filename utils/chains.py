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