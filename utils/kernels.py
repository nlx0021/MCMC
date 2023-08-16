import numpy as np
import copy
from functools import reduce
from math import pi

from utils.proposal_kernels import *

EPSILON = 1e-13

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

    

class M_H_Kernal(BasicKernal):
    
    def __init__(self,
                 f_u,
                 state_dim=1,
                 proposal_kernal=None,
                 is_dual=True,
                 dual_thres=10000):
        
        if proposal_kernal is None:
            proposal_kernal = NormalKernal(state_dim=state_dim)
            
        if isinstance(proposal_kernal, list):
            proposal_kernal = proposal_kernal[0]
            print("Multiple proposal kernals are provided. Only use the first one in M_H_Kernal.")
            
        super().__init__(
            state_dim=state_dim,
            f_u=f_u,
            proposal_kernal=proposal_kernal
        )
        
        self.is_dual = is_dual
        self.dual_thres = dual_thres
        
        if is_dual and proposal_kernal.can_dual:
            
            '''
            Use Dual Average to set delta.
            '''
            
            self.mu = proposal_kernal.delta
            self.gamma = .05
            self.t_0 = 10
            self.kappa = .75
            
            self.H_sum = 0
            self.t = 0            
        
        else:
            self.is_dual = False
        
        self.reject_n = 0
        self.ct = 0
        self.reject_ratio = 0
        
    
    def __call__(self,
                 state,
                 temperature=1):
        
        self.ct += 1
        
        proposal_state = self.proposal_kernal(state, temperature)
        
        alpha = min([
            1,
            self.f_u(proposal_state, temperature) * self.proposal_kernal.pdf(state=proposal_state, next_state=state, temperature=temperature) /    \
           (self.f_u(state, temperature)          * self.proposal_kernal.pdf(state=state, next_state=proposal_state, temperature=temperature) + EPSILON)
        ])
        
        if np.random.random() < alpha:
            next_state = proposal_state 
        
        else:
            next_state = state
            self.reject_n += 1
            self.reject_ratio = self.reject_n / self.ct
            
        if self.is_dual and self.ct < self.dual_thres:
            
            H = .65 - alpha
            self.H_sum += H
            self.t += 1
            
            # Adapt delta.
            # FIXME: not use x=log(delta). x=delta instead.
            _delta = self.mu - np.sqrt(self.t) / self.gamma / (self.t + self.t_0) * self.H_sum
            _ita = self.t ** -self.kappa
            self.proposal_kernal.delta = _ita * _delta + (1 - _ita) * self.proposal_kernal.delta
            
            
        
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
            f_u(s, temperature) * product([
                proposal_kernal_list[i].pdf(state=s, next_state=proposal_state_list[j])
                for i, j in enumerate([_ for _ in range(try_n+1) if _ != idx])
            ]) for idx, s in enumerate(proposal_state_list)
        ]
        
        _sum = sum(prob_mass_list)
        prob_mass_list = [v / (_sum) for v in prob_mass_list]
        
        next_state_idx = np.random.choice(
            np.arange(try_n+1), p=prob_mass_list
        )
        
        next_state = proposal_state_list[next_state_idx]
        
        return next_state
    
    
    def pdf(self, state, next_state, temperature):
        
        pass    
    
    

class HMC_Kernal(BasicKernal):
    
    def __init__(self,
                 f_u,
                 state_dim=1,
                 proposal_kernal=None,
                 epsilon=1e-3,
                 L=20,
                 is_dual=True,
                 dual_thres=10000):
        
        super().__init__(
            state_dim=state_dim,
            f_u=f_u
        )
        
        self.epsilon = epsilon
        self.L = L
        self.is_dual = is_dual
        self.dual_thres = dual_thres
        
        if is_dual:
            
            '''
            Use Dual Average to set epsilon.
            '''
            
            self.mu = epsilon
            self.gamma = .05
            self.t_0 = 10
            self.kappa = .75
            
            self.H_sum = 0
            self.t = 0
        
        self.reject_n = 0
        self.reject_ratio = 0
        self.ct = 0
        
        
    def __call__(self,
                 state,
                 temperature=1):
        
        self.ct += 1
        
        L, epsilon = self.L, self.epsilon
        f_u = self.f_u
        
        # Randomly choose a momentum.
        r = np.random.multivariate_normal(
            mean = np.zeros((self.state_dim, )),
            cov = np.eye(self.state_dim)
        )
        
        # Leapfrog.
        _state, _r = state.copy(), r.copy()
        for _ in range(L):
            _state, _r = self.leapfrog(
                state=_state, r=_r,
                epsilon=epsilon, temperature=temperature
            )
        proposal_state = _state
        
        # Reject-Accept.
        alpha = min(
            1,
            np.exp( -np.log(f_u(state,          temperature) + EPSILON) + np.dot(r,  r ) / 2 )  /              # Energy in (state, r).
           (np.exp( -np.log(f_u(proposal_state, temperature) + EPSILON) + np.dot(_r, _r) / 2 ) + EPSILON)        # Energy in (_state, _r).
        )
        
        if np.random.random() < alpha:
            next_state = proposal_state 
        
        else:
            next_state = state  
            self.reject_n += 1
            
        if self.is_dual and self.ct < self.dual_thres:
            
            H = .65 - alpha
            self.H_sum += H
            self.t += 1
            
            # Adapt epsilon.
            # FIXME: not use x=log(epsilon). x=epsilon instead.
            _epsilon = self.mu - np.sqrt(self.t) / self.gamma / (self.t + self.t_0) * self.H_sum
            _ita = self.t ** -self.kappa
            self.epsilon = _ita * _epsilon + (1 - _ita) * self.epsilon
            
        self.reject_ratio = self.reject_n / self.ct
            
        return next_state     
        
    
    def leapfrog(self,
                 state, r,
                 epsilon,
                 temperature=1):
        
        _r = r + epsilon / 2 * self.compute_grad(state, temperature)
        _state = state + epsilon * _r
        _r = _r + epsilon / 2 * self.compute_grad(_state, temperature)
        
        return _state, _r
    
    
    def compute_grad(self,
                     state,
                     temperature=1):
        
        f_u = self.f_u
        
        # Compute gradient.
        grad = np.zeros_like(state)
        for dim in range(self.state_dim):
            
            _diff = np.zeros_like(state)
            _diff[dim] = 1e-5

            grad[dim] = (
                    (np.log(f_u(state+_diff, temperature) + EPSILON) - np.log(f_u(state-_diff, temperature) + EPSILON))
                   ) / (2 * 1e-5 + EPSILON)        
    
        return grad
    
    
    def pdf(self, state, next_state, temperature):
        
        pass