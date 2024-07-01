from utils.functions import *

def make_mlp(dim_list, bias_list, act_list, drop_list):

    num_layers = len(dim_list) - 1

    layers = []
    for i in range(num_layers):

        # layer info
        dim_in, dim_out = dim_list[i], dim_list[i+1]
        bias = bias_list[i]
        activation = act_list[i]
        drop_prob = drop_list[i]

        # linear layer
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))

        # add activation
        if (activation == 'relu'): layers.append(nn.ReLU())
        elif (activation == 'sigmoid'): layers.append(nn.Sigmoid())
        elif (activation == 'tanh'): layers.append(nn.Tanh())

        # add dropout
        if (drop_prob > 0): layers.append(nn.Dropout(p=drop_prob))

    return nn.Sequential(*layers)

def init_hidden(num_layers, batch, h_dim):
    c = torch.zeros(num_layers, batch, h_dim).cuda()
    h = torch.zeros(num_layers, batch, h_dim).cuda()
    return (h, c)
