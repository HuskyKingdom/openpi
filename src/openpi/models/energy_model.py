from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    


class SeqPool(nn.Module):
    def __init__(self, mode="mean"):
        super().__init__()
        assert mode in ["cls", "mean"]
        self.mode = mode

    def forward(self, h):  # h: [B,S,Dh]
        if self.mode == "cls":
            return h[:, 0, :]                       
        else:
            return h.mean(dim=1)
        


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.act(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

    
def assert_finite(x, name):
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).nonzero(as_tuple=False)[:5]
        raise RuntimeError(f"[NaN] {name} has non-finite at {bad.shape[0]} positions, e.g. {bad[:3].tolist()}")



class EnergyModel(nn.Module):
    """
    E_phi(s, a):
    input: hN(s) [B, seq, D_h], a [B, chunk, D_a] 
    output: energy [B, 1]
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden: int = 512,
        head: int = 8,
        layers: int = 4,
    ):
        super().__init__()

     
        # self.energy_bc = FFWRelativeCrossAttentionModule(hidden,head,layers)
        self.cross = nn.MultiheadAttention(hidden, head, batch_first=True)

        # pos emb
        self.pe_layer = PositionalEncoding(hidden,0.2)

        self.state_linear = MLPResNet(
            num_blocks=1, input_dim=state_dim, hidden_dim=hidden, output_dim=hidden
        )
        self.action_linear = MLPResNet(
            num_blocks=1, input_dim=act_dim, hidden_dim=hidden, output_dim=hidden
        )
        self.prediction_head = MLPResNet(
            num_blocks=2, input_dim=hidden, hidden_dim=hidden, output_dim=1
        )
        self.pool = SeqPool(mode="mean")


        self.T = 30.0               # temperature for energy range
        
        self.act = nn.Sigmoid() 
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.energy_scale = 2.0
        self.energy_offset = 0.1
    

    def forward(self, hN: torch.Tensor, a: torch.Tensor, pad_mask = None, reduce="sum", gamma=None) -> torch.Tensor:
        """
        hN: [B, S, D_h], a: [B, H,  D_a]
        return: energy [B, 1]
        """

        # context_mapped = self.state_linear(hN)  # [B,S,Dh]
        # action_mapped  = self.pe_layer(self.action_linear(a))  # [B,H,Da]
        # cls_tokens = self.cls_token.expand(hN.shape[0], -1, -1)

        # energy_concat = torch.cat([cls_tokens, context_mapped, action_mapped], dim=1).to(hN.dtype)  # [B,S+H+1,Da]

        # energy_features = self.energy_bc(energy_concat.transpose(0,1), diff_ts=None,
        #         query_pos=None, context=None, context_pos=None,pad_mask=pad_mask)[-1].transpose(0,1)  # [B,S+H+1,Da]
        

        # # energy_cls = energy_features[:,0,:].squeeze(1)
        # E = self.prediction_head(energy_cls) # [B, 1]


        hN = hN.float()
        a  = a.float()
        
        assert_finite(hN, "hN")
        assert_finite(a,  "a")
        

        if pad_mask is not None:
            if pad_mask.all(dim=1).any():
                raise RuntimeError("[NaN-risk] some rows key_padding_mask are all True")


    
        # return E
        context_mapped = self.state_linear(hN)  # [B,S,Dh]
        action_mapped  = self.pe_layer(self.action_linear(a))  # [B,H,Da]
        # assert_finite(context_mapped, "context_mapped")
        # assert_finite(action_mapped,  "action_mapped")

        # energy_feat = self.energy_bc(query=action_mapped.transpose(0, 1),
        #     value=context_mapped.transpose(0, 1),
        #     query_pos=None,
        #     value_pos=None,
        #     diff_ts=None)[-1].transpose(0,1) # [B,H,Da]
        
        # energy_feat = energy_feat + self.gate_a * action_mapped 

        # energy = self.pool(energy_feat) # [B,Da]
        # E = self.prediction_head(energy) # [B, 1]


        Z, _ = self.cross(query=action_mapped, key=context_mapped, value=context_mapped, need_weights=False, key_padding_mask=pad_mask)
        # assert_finite(Z, "attn_out")

        energy_feature_step = self.prediction_head(Z)
        # assert_finite(energy_feature_step, "energy_feature_step")


        # raw = self.T * torch.tanh(raw / self.T)
        energy_feature_step = energy_feature_step * 0.5
        E = self.act(energy_feature_step) * self.energy_scale + self.energy_offset
        # assert_finite(E, "E")

        energy_avg = self.pool(E)
        assert_finite(energy_avg, "energy_avg")

        return energy_avg