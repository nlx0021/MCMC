import numpy as np
import yaml

from utils.kernels import *
from utils.chains import *
from utils.targets import *
from utils.visualizer import *


OBJECT_DICT = {
    "MultiPeaksTarget": MultiPeaksTarget,
    "NormalKernal": NormalKernal,
    "LangevinKernal": LangevinKernal,
    "M_H_Kernal": M_H_Kernal,
    "MultiTry_M_H_Kernal": MultiTry_M_H_Kernal,
    "M_H_Chain": M_H_Chain,
    "PT_M_H_Chain": PT_M_H_Chain        
}


def get(cfg, type, kwargs={}):
    
    '''
    type in ["Target", "ProposalKernal", "MultiProposalKernal", "Kernal", "Chain"]
    '''

    if type == "MultiProposalKernal":
        _dict = cfg[type]
        kernal_list = []
        
        for _, kernal_dict in _dict.items():
            cls = kernal_dict["class"]
            
            if kernal_dict[cls] is None:
                kernal_list.append(
                    OBJECT_DICT[cls](**kwargs)
                )                
            else:
                kernal_list.append(
                    OBJECT_DICT[cls](**kernal_dict[cls], **kwargs)
                )
        
        return kernal_list

    else:
        cls = cfg[type]["class"]
        if cfg[type][cls] is None:
            return OBJECT_DICT[cls](**kwargs)
        
        return OBJECT_DICT[cls](**cfg[type][cls], **kwargs)
    
    

def main(cfg):
    
    state_dim = cfg["Base"]["state_dim"]
    length = cfg["Base"]["length"]
    burn = cfg["Base"]["burn"]
    is_multiproposal = cfg["Base"]["multi_proposal"]
    
    # Select target.    
    f_u = get(cfg, type="Target")
    
    # Select algorithm.
    if not is_multiproposal:
        proposal_kernal = get(cfg, type="ProposalKernal",
                            kwargs={"f_u": f_u})
    else:
        proposal_kernal = get(cfg, type="MultiProposalKernal",
                            kwargs={"f_u": f_u})
    
    
    kernal = get(cfg, type="Kernal",
                 kwargs={"f_u": f_u, "proposal_kernal": proposal_kernal})
    
    chain = get(cfg, type="Chain",
                kwargs={"f_u": f_u, "kernal": kernal, "init_state": np.random.uniform(low=-1, high=1, size=(state_dim,))})
    
    # Run the chain.
    chain.run(length=length)
    # print("Reject num: %d" % kernal.reject_n)     # Only valid for M-H Kernal.

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