"""
DanZero Q-Network Actor for positions 1 and 3 (opponents)
监听端口 6001 和 6003
"""
import os
import pickle
import time
from argparse import ArgumentParser
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import zmq
from pyarrow import deserialize, serialize
from tensorflow.keras.backend import set_session

from model import GDModel

parser = ArgumentParser()
parser.add_argument('--observation_space', type=int, default=(567, ),
                    help='The YAML configuration file')
parser.add_argument('--action_space', type=int, default=(5, 216),
                    help='The YAML configuration file')
parser.add_argument('--model', type=str, default='../model/',
                    help='The YAML configuration file')

class Player():
    def __init__(self, args) -> None:
        # Set 'allow_growth'
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # 数据初始化
        self.args = args
        self.init_time = time.time()

        # 模型初始化
        self.model  = GDModel(args.observation_space, (5, 216))
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(script_dir, 'q_network.ckpt')
        with open(ckpt_path, 'rb') as f:
            new_weights = pickle.load(f)
        self.model.set_weights(new_weights)
    
    def sample(self, state) -> int:
        output = self.model.forward(state['x_batch'])
        action_idx = np.argmax(output)
        return action_idx


def run_one_player(index, args):
    player = Player(args)

    # 初始化zmq - 监听端口 6000 + index
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = 6000 + index
    socket.bind(f'tcp://*:{port}')
    print(f'Actor for position {index} listening on port {port}')

    action_index = 0
    while True:
        state = deserialize(socket.recv())
        action_index = player.sample(state)
        socket.send(serialize(action_index).to_buffer())


def main():
    # 参数传递
    args, _ = parser.parse_known_args()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_player(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 1:
                for _i, _p in enumerate(players):
                    if _i != index:
                        _p.terminate()

    players = []
    # 位置 1 和 3 (对手队)
    for i in [1, 3]:
        print(f'Starting actor for position {i}')
        p = Process(target=exit_wrapper, args=(i, args))
        p.start()
        time.sleep(0.5)
        players.append(p)

    for player in players:
        player.join()

if __name__ == '__main__':
    main()
