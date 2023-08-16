import numpy as np
import copy
from functools import reduce
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
                 state,
                 temperature=1):
        '''
            Transition.
            Returns:
                next_state: np.ndarray.
        '''
        
        pass
    
    
    def pdf(self,
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
                 sigma=1,
                 f_u=None,
                 proposal_kernal=None):
        
        super().__init__(
            delta=sigma**2,
            state_dim=state_dim
        )
        
        self.can_dual = True
        
    
    def __call__(self,
                 state,
                 temperature=1):
        
        next_state = np.random.multivariate_normal(
            mean = state,
            cov = self.delta * np.eye(self.state_dim)
        )
        
        pdf_v = self.pdf(state, next_state)
        
        return next_state
        
    
    def pdf(self, state, next_state, temperature=1):
        
        return np.exp( -np.linalg.norm(state - next_state) ** 2 / self.delta / 2 )  \
            /  np.sqrt(2*pi*self.delta**self.state_dim)
            
            
            
class LangevinKernal(BasicKernal):
    
    def __init__(self,
                 f_u,
                 state_dim=1,
                 delta=1,
                 proposal_kernal=None):
        
        super().__init__(
            state_dim=state_dim,
            f_u=f_u, delta=delta
        )
        
        self.can_dual = True
        
    
    def __call__(self,
                 state,
                 temperature=1):
        
        f_u = self.f_u
        
        # Compute gradient.
        grad = self.compute_grad(state, temperature)
    
        # Y ~ N( X+(delta/2)grad , delta)
        next_state = np.random.multivariate_normal(
            mean = state + grad * self.delta / 2,
            cov = self.delta * np.eye(self.state_dim)
        )
        
        return next_state
    
    
    def pdf(self, state, next_state, temperature=1):
        
        grad = self.compute_grad(state, temperature)
        
        return np.exp( -np.linalg.norm(state+(self.delta/2)*grad - next_state) ** 2 / self.delta / 2 )  \
            /  np.sqrt(2*pi*self.delta**self.state_dim)
    
    
    def compute_grad(self,
                     state,
                     temperature=1):
        
        f_u = self.f_u
        
        # Compute gradient.
        grad = np.zeros_like(state)
        for dim in range(self.state_dim):
            
            _diff = np.zeros_like(state)
            _diff[dim] = 1e-5    # epsilon to compute grad.

            grad[dim] = (
                    (np.log(f_u(state+_diff, temperature)) - np.log(f_u(state-_diff, temperature))) + 1e-13
                   ) / (2 * 1e-5 + 1e-13)        
    
        return grad