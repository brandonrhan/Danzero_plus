import numpy as np
#import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# 构建张量的函数
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# 对 layer 的权重矩阵进行 正交初始化
# gain 是一个缩放因子
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    # 设置偏置项为0
    nn.init.constant_(layer.bias, 0)

# 可作为 Actor 输出动作概率分布，或 Critic 输出状态价值估计。
def mlp(sizes, activation, output_activation=nn.Identity, use_init=False):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if use_init:
            net = nn.Linear(sizes[j], sizes[j+1])
            orthogonal_init(net)
            layers += [net, act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# 输入观测值（observation）后，输出一个高层特征表示（feature embedding）
def shared_mlp(obs_dim, sizes, activation, use_init=False):  # 分两个叉，一个是过softmax的logits，另一个不过，就是单纯的q(s,a)，这里是前面的共享层
    layers = []
    shapes = [obs_dim] + list(sizes)
    for j in range(len(shapes) - 1):
        act = activation
        if use_init:
            net = nn.Linear(shapes[j], shapes[j+1])
            orthogonal_init(net)
            layers += [net, act()]
        else:
            layers += [nn.Linear(shapes[j], shapes[j + 1]), act()]
    return nn.Sequential(*layers)

# 计算模型参数量
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):
    # 两个抛异常
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    # 计算log概率分布
    def forward(self, obs, act=None, legalaction=torch.tensor(list(range(10))).to(torch.float32)):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, legalaction)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


# Actor的实例
class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    # 非法动作会变成-999999 softmax后几乎为0
    def _distribution(self, obs, legal_action):
        logits = torch.squeeze(self.logits_net(obs)) - (1 - legal_action) * 1e6
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


# 状态价值函数的神经网络 评估总体期望
class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

# 评估每个动作的价值
class MLPQ(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.q_net(obs), -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(512, 512, 512, 512, 256), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space
        # 共享的特征提取网络
        self.shared = shared_mlp(obs_dim[1], hidden_sizes, activation, use_init=True)
        # 输出动作分布的 Actor 网络（pi）
        self.pi = mlp([hidden_sizes[-1], 128, action_space], activation, use_init=True)  # 输出logits
        # 估计状态价值的 Critic 网络（v）
        self.v = mlp([hidden_sizes[-1], 128, 1], activation, use_init=True)    # 输出q(s,a)
        #输入 obs → shared_mlp → 特征表示
        #                  ↘
        #                   → pi → logits（action_space）
        #                  ↘
        #                   → v  → value（scalar）



    # 在给定观测值和合法动作掩码的情况下，采样一个动作并估计其价值（value）和 log 概率
    def step(self, obs, legal_action):
        obs = torch.tensor(obs).to(torch.float32)
        legal_action = torch.tensor(legal_action).to(torch.float32)
        with torch.no_grad():
            shared_feature = self.shared(obs)
            # print(shared_feature.shape, legal_action.shape)
            logits = torch.squeeze(self.pi(shared_feature)) - (1 - legal_action) * 1e8
            #print('share_feature', self.pi(shared_feature).shape, 'logits', logits.shape, 'legal_action', legal_action)
            pi = Categorical(logits=logits)
            a = pi.sample()
            logp_a = pi.log_prob(a)   # 该动作的log(pi)

            value = torch.squeeze(self.v(shared_feature), -1)

        return a.numpy(), value.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def get_weights(self):
        return self.state_dict()
    

class MLPQNetwork(nn.Module):
    def __init__(self, observation_space,
                 hidden_sizes=(512, 512, 512, 512, 512), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space

        # build Q function
        self.q = MLPQ(obs_dim, hidden_sizes, activation)

    def load_tf_weights(self, weights):
        name = ['q_net.0.weight', 'q_net.0.bias', 'q_net.2.weight', 'q_net.2.bias', 'q_net.4.weight', 'q_net.4.bias', 'q_net.6.weight', 'q_net.6.bias', 'q_net.8.weight', 'q_net.8.bias', 'q_net.10.weight', 'q_net.10.bias']
        tensor_weights = []
        for weight in weights:
            temp = torch.tensor(weight).T
            tensor_weights.append(temp)
        new_weights = dict(zip(name, tensor_weights))
        self.q.load_state_dict(new_weights)
        print('load tf weights success')

    def get_max_n_index(self, data, n):
        q_list = self.q(torch.tensor(data).to(torch.float32))
        q_list = q_list.detach().numpy()
        return q_list.argsort()[-n:][::-1].tolist()


if __name__ == '__main__':
    model  = MLPActorCritic((10, 567), 1)
    # state = np.random.random((513, ))
    # action1 = np.random.random((54, ))
    # action2 = np.random.random((54, ))
    # action3 = np.random.random((54, ))
    # b = np.load("/home/zhaoyp/guandan_tog/actor_torch/debug145.npy", allow_pickle=True).item()
    # print(b.keys())
    # print(b['obs_cut'].shape)
    # print(b['obs'].shape)
    import objgraph
    
    # print('time1')
    # objgraph.show_most_common_types(limit=30)
    # objgraph.show_growth()

    state = np.zeros((10,567))
    legal_index = np.ones(10)

    # print('time2')
    # objgraph.show_most_common_types(limit=30)
    # objgraph.show_growth()

    # a, v, p = model.step(state, legal_index)

    # print('time3')
    # objgraph.show_most_common_types(limit=30)
    # objgraph.show_growth()

    # print(a,v,p)
    # print(type(a),type(v),type(p))
