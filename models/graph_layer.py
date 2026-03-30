import math
import time

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, Parameter, ReLU, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GraphLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        inter_dim=-1,
        **kwargs
    ):
        super(GraphLayer, self).__init__(aggr="add", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))

        out = self.propagate(
            edge_index,
            x=x,
            embedding=embedding,
            edges=edge_index,
            return_attention_weights=return_attention_weights,
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(
        self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights
    ):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = (
                embedding[edge_index_i],
                embedding[edges[0]],
            )
            # Ensure embeddings have correct dimensions for broadcasting
            # In PyG 2.7.0, embeddings need to match the message tensor shape exactly
            if embedding_i.dim() == 2:
                embedding_i = embedding_i.unsqueeze(1).expand(-1, self.heads, -1)
            if embedding_j.dim() == 2:
                embedding_j = embedding_j.unsqueeze(1).expand(-1, self.heads, -1)

            # Ensure embedding dimensions match x_i and x_j exactly
            # This prevents the scatter operation dimension mismatch
            if embedding_i.shape[-1] != x_i.shape[-1]:
                # Truncate or pad embedding to match x_i dimensions
                target_dim = x_i.shape[-1]
                if embedding_i.shape[-1] > target_dim:
                    embedding_i = embedding_i[..., :target_dim]
                    embedding_j = embedding_j[..., :target_dim]
                else:
                    # Pad with zeros if needed
                    pad_size = target_dim - embedding_i.shape[-1]
                    embedding_i = F.pad(embedding_i, (0, pad_size))
                    embedding_j = F.pad(embedding_j, (0, pad_size))

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)

        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Return message with proper shape for PyG 2.7.0 aggregation
        # The shape should be [num_edges, heads, out_channels]
        message = x_j * alpha.view(-1, self.heads, 1)
        return message.contiguous()

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
