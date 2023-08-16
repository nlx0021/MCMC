import numpy as np
import yaml

from utils.kernels import *
from utils.proposal_kernels import *
from utils.chains import *
from utils.targets import *
from utils.visualizer import *


OBJECT_DICT = {
    # Target.
    "MultiPeaksTarget": MultiPeaksTarget,
    
    # ProposalKernal.
    "NormalKernal": NormalKernal,
    "LangevinKernal": LangevinKernal,
    
    # Kernal.
    "M_H_Kernal": M_H_Kernal,
    "MultiTry_M_H_Kernal": MultiTry_M_H_Kernal,
    "HMC_Kernal": HMC_Kernal,
    
    # Chain.
    "M_H_Chain": M_H_Chain,
    "PT_M_H_Chain": PT_M_H_Chain        
}


def get(cfg, type, kwargs={}):
    
    '''
    type in ["Target", "Chain", "Kernal"]
    '''

    if type not in ["Kernal", "proposal_kernal"]:
        cls = cfg[type]["class"]
        if cfg[type][cls] is None:
            return OBJECT_DICT[cls](**kwargs)
        
        return OBJECT_DICT[cls](**cfg[type][cls], **kwargs)

    # Else, get the kernal.
    kernal_cfg = cfg[type]
    kernal_list = []
    
    for kernal_dict in kernal_cfg.values():
        kernal_cls = kernal_dict["class"]
        params = kernal_dict[kernal_cls]
        
        if "proposal_kernal" in params.keys():
            # Need to get proposal_kernal.
            proposal_kernal_list = get(cfg=params,
                                       type="proposal_kernal",
                                       kwargs=kwargs)
            params.pop("proposal_kernal")

        else:
            proposal_kernal_list = None
        
        kernal = OBJECT_DICT[kernal_cls](
            **params, **kwargs, proposal_kernal=proposal_kernal_list
        )
        
        kernal_list.append(kernal)
    
    if len(kernal_list) == 1:
        return kernal_list[0]
    
    return kernal_list


def main(cfg):
    
    state_dim = cfg["Base"]["state_dim"]
    length = cfg["Base"]["length"]
    burn = cfg["Base"]["burn"]
    
    # Select target.    
    f_u = get(cfg, type="Target")
    
    # Select algorithm (kernal & chain).
    kernal = get(cfg, type="Kernal",
                 kwargs={"f_u": f_u})

    chain = get(cfg, type="Chain",
                kwargs={"f_u": f_u, "kernal": kernal, "init_state": np.random.uniform(low=-1, high=1, size=(state_dim,))})
    
    # Run the chain.
    chain.run(length=length)

    # Stats.
    if not isinstance(chain, PT_M_H_Chain):            # Not implemented for PT Chain.
        kernal_stat_list = ['epsilon', 'reject_ratio', 'delta']
        for stat in kernal_stat_list:
            kernal = chain.kernal
            if stat in dir(kernal) and getattr(kernal, stat) is not None:
                print("Kernal's %s is: %f" % (stat, getattr(kernal, stat)))
            if (kernal.proposal_kernal is not None)    \
                and not isinstance(kernal.proposal_kernal, list)   \
                and stat in dir(kernal.proposal_kernal)    \
                and getattr(kernal.proposal_kernal, stat) is not None:
                print("Proposal kernal's %s is %f" % (stat, getattr(kernal.proposal_kernal, stat)))
                
    chain_stat_list = ['switch_ratio']
    for stat in chain_stat_list:
        if stat in dir(chain):
            print("%s is: %f" % (stat, getattr(chain, stat)))
    

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
    
    cfg_path = "./configs/General.yaml"
    with open(cfg_path, 'r', encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    main(cfg)