import numpy as np
import yaml

from utils.kernels import *
from utils.chains import *
from utils.targets import *
from utils.visualizer import *


OBJECT_DICT = {
    "MultiPeaksTarget": MultiPeaksTarget,
    "NormalKernal": NormalKernal,
    "M_H_Kernal": M_H_Kernal,
    "M_H_Chain": M_H_Chain,
    "PT_M_H_Chain": PT_M_H_Chain        
}


def get(cfg, type, kwargs={}):
    
    '''
    type in ["Target", "ProposalKernal", "Kernal", "Chain"]
    '''

    cls = cfg[type]["class"]
    if cfg[type][cls] is None:
        return OBJECT_DICT[cls](**kwargs)
    
    return OBJECT_DICT[cls](**cfg[type][cls], **kwargs)
    
    

def main(cfg):
    
    state_dim = cfg["Base"]["state_dim"]
    length = cfg["Base"]["length"]
    burn = cfg["Base"]["burn"]
    
    # Select target.    
    f_u = get(cfg, type="Target")
    
    # Select algorithm.
    proposal_kernal = get(cfg, type="ProposalKernal")
    
    kernal = get(cfg, type="Kernal",
                 kwargs={"f_u": f_u, "proposal_kernal": proposal_kernal})
    
    chain = get(cfg, type="Chain",
                kwargs={"f_u": f_u, "kernal": kernal, "init_state": np.random.uniform(low=-1, high=1, size=(state_dim,))})
    
    # Run the chain.
    chain.run(length=length)

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
    
    cfg_path = "./configs/M-H.yaml"
    with open(cfg_path, 'r', encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    main(cfg)