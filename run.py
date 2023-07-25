import numpy as np

from utils.kernels import *
from utils.chains import *
from utils.targets import *
from utils.visualizer import *


def main():
    
    state_dim = 2
    length = 30000
    burn = 1000
    
    # Select target.
    f_u = MultiPeaksTarget(
        state_dim=state_dim
    )
    
    # Select algorithm.
    kernal = M_H_Kernal(
        f_u=f_u,
        state_dim=state_dim,
        proposal_kernal=NormalKernal(state_dim=state_dim, sigma=1)
    )
    
    chain = M_H_Chain(
        init_state=np.zeros((state_dim,)),
        kernal=kernal
    )
    
    # chain = PT_M_H_Chain(
    #     init_state=np.random.uniform(low=-1, high=1, size=(state_dim,)),
    #     kernal=kernal,
    #     f_u=f_u
    # )
    
    # Run the chain.
    chain.run(length=length)
    print("Reject num: %d" % chain.kernal.reject_n)      # For M-H Algorithm.

    # Visualize.
    visualizer = Visualizer(
        f_u=f_u,
        chain=chain
    )
    visualizer.vis(
        burn=burn,
        s=3
    )
    
    # import pdb; pdb.set_trace()
    
    
    
if __name__ == '__main__':
    
    main()