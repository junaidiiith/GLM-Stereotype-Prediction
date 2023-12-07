import torch.nn as nn
import torch_geometric
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GNNModel(torch.nn.Module):
  """GraphSage Network"""
  def __init__(
          self, model_name, 
          input_dim, 
          hidden_dim, 
          out_dim, 
          num_layers, 
          num_heads=None, 
          residual=False, 
          l_norm=False, 
          dropout=0.1
    ):
    super(GNNModel, self).__init__()
    gnn_model = getattr(torch_geometric.nn, model_name)
    self.conv_layers = nn.ModuleList()
    if model_name == 'GINConv':
        input_layer = gnn_model(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()), train_eps=True)
    elif num_heads is None:
        input_layer = gnn_model(input_dim, hidden_dim, aggr='SumAggregation')
    else:
        input_layer = gnn_model(input_dim, hidden_dim, heads=num_heads, aggr='SumAggregation')
    self.conv_layers.append(input_layer)

    for _ in range(num_layers - 2):
        if model_name == 'GINConv':
            self.conv_layers.append(gnn_model(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), train_eps=True))
        elif num_heads is None:
            self.conv_layers.append(gnn_model(hidden_dim, hidden_dim, aggr='SumAggregation'))
        else:
            self.conv_layers.append(gnn_model(num_heads*hidden_dim, hidden_dim, heads=num_heads, aggr='SumAggregation'))

    if model_name == 'GINConv':
        self.conv_layers.append(gnn_model(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU()), train_eps=True))
    else:
        self.conv_layers.append(gnn_model(hidden_dim if num_heads is None else num_heads*hidden_dim, out_dim, aggr='SumAggregation'))
        
    self.activation = nn.ReLU()
    self.layer_norm = nn.LayerNorm(hidden_dim if num_heads is None else num_heads*hidden_dim) if l_norm else None
    self.residual = residual
    self.dropout = nn.Dropout(dropout)


  def forward(self, in_feat, edge_index):
    h = in_feat
    h = self.conv_layers[0](h, edge_index)
    h = self.activation(h)
    if self.layer_norm is not None:
        h = self.layer_norm(h)
    h = self.dropout(h)

    for conv in self.conv_layers[1:-1]:
        h = conv(h, edge_index) if not self.residual else conv(h, edge_index) + h
        h = self.activation(h)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        h = self.dropout(h)
    
    h = self.conv_layers[-1](h, edge_index)
    return h
