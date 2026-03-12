import torch
from torch import nn, tensor, float32
from .acts import activations

def generate_aufbau_features(max_atomic_number, target_device):
    '''
    Dynamic electronic feature generator based on the Aufbau principle.
    Perfectly suited for main-group organic elements (H, C, N, O, P, S, Halogens like I, Br).
    Calculates Pair (P) and Single (S) electrons per subshell.
    '''
    ORBITALS = [
        ('1s', 2), ('2s', 2), ('2p', 6), ('3s', 2), ('3p', 6), 
        ('4s', 2), ('3d', 10), ('4p', 6), ('5s', 2), ('4d', 10), 
        ('5p', 6), ('6s', 2), ('4f', 14), ('5d', 10), ('6p', 6)
    ]
    
    def calc_p_s(electrons, capacity):
        """根据洪特规则计算成对电子(P)和单电子(S)的数量"""
        if electrons <= capacity // 2:
            return 0, electrons  # 未过半满，全为单电子
        else:
            s = capacity - electrons  # 剩余空位就是单电子数量
            p = electrons - s         # 总电子减去单电子就是成对电子
            return p, s

    features = []
    
    for z in range(max_atomic_number + 1):
        if z == 0:
            features.append([0] * (len(ORBITALS) * 2))
            continue
            
        z_left = z
        z_feat = []
        
        # 逐个轨道填充电子
        for name, cap in ORBITALS:
            if z_left >= cap:
                p, s = calc_p_s(cap, cap) # 轨道全满
                z_left -= cap
            elif z_left > 0:
                p, s = calc_p_s(z_left, cap) # 轨道部分填充
                z_left -= cap # 置为负数，后续不再填充
            else:
                p, s = 0, 0 # 轨道为空
                
            z_feat.extend([p, s])
            
        features.append(z_feat)
        
    elec_tensor = tensor(features, dtype=float32)
    # 归一化特征
    return (elec_tensor / elec_tensor.max()).to(target_device), len(ORBITALS) * 2


class Embedding(nn.Module):
    def __init__(self, num_features, act, max_atomic_number=60):
        super(Embedding, self).__init__()
        
        # 获取特征矩阵以及其动态列数 (P和S的维度)
        elec_matrix, elec_dim = generate_aufbau_features(
            max_atomic_number=max_atomic_number, 
            target_device=torch.device('cpu')
        )
        
        # 注册 Buffer
        self.register_buffer('elec', elec_matrix)
        self.act = activations(type=act, num_features=num_features)

        self.elec_emb = nn.Linear(elec_dim, num_features, bias=False)
        self.nuclare_emb = nn.Embedding(max_atomic_number + 1, num_features)
        self.ls = nn.Linear(num_features, num_features)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ls.weight)
        self.ls.bias.data.fill_(0)
        self.nuclare_emb.reset_parameters()
        nn.init.xavier_uniform_(self.elec_emb.weight)

    def forward(self, z):
        '''
        The initial invariant feature of an atom consists of a mixture of nuclear one-hot and electronic features
        '''
        return self.act(self.ls(self.nuclare_emb(z) + self.elec_emb(self.elec[z, :])))