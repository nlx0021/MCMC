import numpy as np
from math import pi

class BasicKernal():
    
    def __init__(self,
                 state_dim=1,
                 f_u=None,
                 proposal_kernal=None,
                 delta=None):
        
        self.state_dim = state_dim
        self.f_u = f_u
        self.proposal_kernal = proposal_kernal
        self.delta = delta
        
        
    def __call__(self,
                 state):
        '''
            Transition.
            Returns:
                next_state: np.ndarray.
        '''
        
        pass
    
    
    def _pdf(self,
             state,
             next_state):
        '''
            pdf value.
            Returns:
                P(next_state | state): float.
        '''
    
        pass
    
    
    
class NormalKernal(BasicKernal):
    
    '''
        P(Y|X) ~ N(X, sigma**2 * I)
    '''
    
    def __init__(self,
                 state_dim=1,
                 sigma=1):
        
        super().__init__(
            delta=sigma**2,
            state_dim=state_dim
        )
        
    
    def __call__(self,
                 state):
        
        next_state = np.random.multivariate_normal(
            mean = state,
            cov = self.delta * np.eye(self.state_dim)
        )
        
        pdf_v = self.pdf(state, next_state)
        
        return next_state
        
    
    def pdf(self, state, next_state):
        
        return np.exp( -np.linalg.norm(state - next_state) ** 2 / self.delta / 2 )  \
            /  np.sqrt(2*pi*self.delta**self.state_dim)
            
            

class M_H_Kernal(BasicKernal):
    
    def __init__(self,
                 f_u,
                 state_dim=1,
                 proposal_kernal=None):
        
        if proposal_kernal is None:
            proposal_kernal = NormalKernal(state_dim=state_dim)
            
        super().__init__(
            state_dim=state_dim,
            f_u=f_u,
            proposal_kernal=proposal_kernal
        )
        
        self.reject_n = 0
        
    
    def __call__(self,
                 state):
        
        
        proposal_state = self.proposal_kernal(state)
        
        alpha = min([
            1,
            self.f_u(proposal_state) * self.proposal_kernal.pdf(state=proposal_state, next_state=state) /    \
            self.f_u(state)          * self.proposal_kernal.pdf(state=state, next_state=proposal_state)
        ])
        
        if np.random.random() < alpha:
            next_state = proposal_state 
        
        else:
            next_state = state
            self.reject_n += 1
        
        return next_state
    
    
    def pdf(self, state, next_state):
        
        pass