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
                 f_u=None):
        
        super().__init__(
            delta=sigma**2,
            state_dim=state_dim
        )
        
    
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
                 epsilon=1e-5):
        
        super().__init__(
            state_dim=state_dim,
            f_u=f_u, delta=delta
        )
        
        self.epsilon = epsilon
        
    
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
            _diff[dim] = self.epsilon

            grad[dim] = (np.log(f_u(state+_diff, temperature)) - np.log(f_u(state-_diff, temperature))) \
                        / (2 * self.epsilon)        
    
        return grad
    
    

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
                 state,
                 temperature=1):
        
        
        proposal_state = self.proposal_kernal(state, temperature)
        
        alpha = min([
            1,
            self.f_u(proposal_state, temperature) * self.proposal_kernal.pdf(state=proposal_state, next_state=state, temperature=temperature) /    \
            self.f_u(state, temperature)          * self.proposal_kernal.pdf(state=state, next_state=proposal_state, temperature=temperature)
        ])
        
        if np.random.random() < alpha:
            next_state = proposal_state 
        
        else:
            next_state = state
            self.reject_n += 1
        
        return next_state
    
    
    def pdf(self, state, next_state, temperature):
        
        pass
    
    
    
class MultiTry_M_H_Kernal(BasicKernal):
    
    def __init__(self,
                 f_u,
                 state_dim=1,
                 proposal_kernal=None,
                 try_n=5):
        
        '''
        Ensemble MCMC Approach.
        '''
        
        if proposal_kernal is None:
            proposal_kernal_list = [NormalKernal(state_dim=state_dim) for _ in range(try_n)]
        
        elif not isinstance(proposal_kernal, list):
            proposal_kernal_list = [copy.deepcopy(proposal_kernal) for _ in range(try_n)]
        
        else:
            proposal_kernal_list = proposal_kernal
            assert len(proposal_kernal) == try_n,  "Num of proposals should be equal to try_n!"
            
        super().__init__(
            state_dim=state_dim,
            f_u=f_u,
            proposal_kernal=None
        )
        
        self.try_n = try_n
        self.proposal_kernal_list = proposal_kernal_list
        
        
    def __call__(self,
                 state,
                 temperature=1):
        
        try_n = self.try_n
        f_u = self.f_u
        proposal_kernal_list = self.proposal_kernal_list
        
        proposal_state_list = [
            proposal_kernal_list[kernal_id](state, temperature) for kernal_id in range(try_n)
        ] + [state]
        
        filter_it = lambda x, leave_out: x in leave_out
        product = lambda ls: reduce(lambda x,y: x*y, ls)     # compute accumulate product of ls.
        prob_mass_list = [
            f_u(s) * product([
                proposal_kernal_list[i].pdf(state=s, next_state=proposal_state_list[j])
                for i, j in enumerate([_ for _ in range(try_n+1) if _ != idx])
            ]) for idx, s in enumerate(proposal_state_list)
        ]
        
        _sum = sum(prob_mass_list)
        prob_mass_list = [v / _sum for v in prob_mass_list]
        
        next_state_idx = np.random.choice(
            np.arange(try_n+1), p=prob_mass_list
        )
        
        next_state = proposal_state_list[next_state_idx]
        
        return next_state
    
    
    def pdf(self, state, next_state, temperature):
        
        pass    