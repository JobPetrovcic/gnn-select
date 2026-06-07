"""Minimal GNN module for serving — extracted verbatim from the training code
(gnn_main.py GNN/L2Norm), with the debug/logging branch removed so it depends
only on torch + torch_geometric (no wandb / leandojo / dataset)."""
from typing import Dict, Any
import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, GCNConv, GraphNorm, JumpingKnowledge


class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class GNN(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(GNN, self).__init__()
        self.config = config

        input_size = config['input_size']
        hidden_size = config['hidden_size']

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        layer_type = config['layer_type']
        norm_type = config.get('normalization', 'none')

        for i in range(config['n_layers']):
            in_channels = input_size if i == 0 else hidden_size

            if layer_type == 'RGCN':
                conv = RGCNConv(in_channels, hidden_size, num_relations=config['n_relations'])
            elif layer_type == 'RGAT':
                assert hidden_size % config['heads'] == 0
                conv = RGATConv(in_channels, hidden_size // config['heads'], num_relations=config['n_relations'], heads=config['heads'])
            elif layer_type == 'GAT':
                assert hidden_size % config['heads'] == 0
                conv = GATConv(in_channels, hidden_size // config['heads'], heads=config['heads'])
            elif layer_type == 'GCN':
                conv = GCNConv(in_channels, hidden_size)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.convs.append(conv)

            if norm_type in ['batchnorm', 'batch']:
                norm = torch.nn.BatchNorm1d(hidden_size)
            elif norm_type in ['layernorm', 'layer']:
                norm = torch.nn.LayerNorm(hidden_size)
            elif norm_type in ['l2']:
                norm = L2Norm()
            elif norm_type in ['graphnorm', 'graph']:
                norm = GraphNorm(hidden_size)
            else:
                norm = torch.nn.Identity()
            self.norms.append(norm)

        self.jk_mode = config.get('jumping_knowledge', 'none')
        if self.jk_mode != 'none':
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=hidden_size, num_layers=config['n_layers'])

        if self.config['residual'] and input_size != hidden_size:
            self.residual_projection = torch.nn.Linear(input_size, hidden_size)
        else:
            self.residual_projection = torch.nn.Identity()

        if config['n_layers'] == 0 and input_size != hidden_size:
            self.input_projection = torch.nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = torch.nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        if self.config['n_layers'] == 0:
            return self.input_projection(x)

        xs = []
        res_position = self.config.get('res_position', 'postnorm')

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x

            if res_position == 'prenorm':
                x = norm(x)

            if isinstance(conv, (RGCNConv, RGATConv)):
                x = conv(x, edge_index, edge_type)
            else:
                x = conv(x, edge_index)

            act = self.config['activation']
            if act == 'relu':
                x = torch.relu(x)
            elif act == 'gelu':
                x = torch.nn.functional.gelu(x)
            elif act == 'swish':
                x = torch.nn.functional.silu(x)
            elif act == 'elu':
                x = torch.nn.functional.elu(x)

            if self.config['dropout'] > 0.0:
                x = torch.nn.functional.dropout(x, p=self.config['dropout'], training=self.training)

            if self.config['residual']:
                if i == 0:
                    x_res = self.residual_projection(x_res)
                x = x + x_res

            if res_position == 'postnorm':
                x = norm(x)

            if self.jk_mode != 'none':
                xs.append(x)

        if self.jk_mode != 'none':
            x = self.jk(xs)

        return x
