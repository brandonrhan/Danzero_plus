import multiprocessing
import os
import io
import pickle
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Process

import torch
import numpy as np
import zmq
from ppo import PPOAgent
from common import create_experiment_dir, load_yaml_config, save_yaml_config
from mem_pool import MemPoolManager, MultiprocessingMemPool
from model import MLPActorCritic, MLPQNetwork
from pyarrow import deserialize
# from torch.utils.tensorboard import SummaryWriter
from utils import logger
from utils.cmdline import parse_cmdline_kwargs

warnings.filterwarnings("ignore")

# Parser arguments
parser = ArgumentParser()
parser.add_argument('--env', type=str, default='GuanDan', help='The game environment')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to receive training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server to publish model parameters')
# parser.add_argument('--pool_size', type=int, default=256, help='The max length of data pool')
# parser.add_argument('--batch_size', type=int, default=256, help='The batch size for training')
parser.add_argument('--pool_size', type=int, default=2048, help='The max length of data pool')
parser.add_argument('--batch_size', type=int, default=2048, help='The batch size for training')
parser.add_argument('--training_freq', type=int, default=13,
                    help='How many receptions of new data are between each training, '
                         'which can be fractional to represent more than one training per reception')
parser.add_argument('--keep_training', type=bool, default=False,
                    help="No matter whether new data is received recently, keep training as long as the data is enough "
                         "and ignore `--training_freq`")
parser.add_argument('--config', type=str, default=None, help='Directory to config file')
parser.add_argument('--exp_path', type=str, default=None, help='Directory to save logging data and config file')
parser.add_argument('--record_throughput_interval', type=int, default=20,
                    help='The time interval between each throughput record')
parser.add_argument('--ckpt_save_freq', type=int, default=500, help='The number of updates between each weights saving')
parser.add_argument('--ckpt_save_type', type=str, default='weight', help='Type of checkpoint file will be recorded : weight(smaller) or checkpoint(bigger')


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def main():
    ActionNumber = 2
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'learner')


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = MLPActorCritic((2, 516 + ActionNumber * 54), ActionNumber).to(device)
    #with open('./ppo79000.pth', 'rb') as f:
        #new_weights = CPU_Unpickler(f).load()
    #model.set_weights(new_weights)
    agent = PPOAgent(model)
    model_update_times = 0
    with open(f'/home/zhaoyp/guandan_tog/learner_torch/ckpt_bak/{model_update_times}.pth', 'wb') as f:
        pickle.dump(agent.get_weights(), f)

    # Configure experiment directory
    create_experiment_dir(args, 'LEARNER-')    
    save_yaml_config(args.exp_path / 'config.yaml', args, 'learner', agent)
    args.log_path = args.exp_path / 'log'
    args.ckpt_path = args.exp_path / 'ckpt'
    args.ckpt_path.mkdir()
    args.log_path.mkdir()

    logger.configure(str(args.log_path))

    receiving_condition = multiprocessing.Condition()
    num_receptions = multiprocessing.Value('i', 0)

    # Start memory pool in another process
    manager = MemPoolManager()
    manager.start()
    mem_pool = manager.MemPool(capacity=args.pool_size)
    Process(target=recv_data,
            args=(args.data_port, mem_pool, receiving_condition, num_receptions, args.keep_training)).start()

    # Print throughput statistics
    Process(target=MultiprocessingMemPool.record_throughput, args=(mem_pool, args.record_throughput_interval)).start()

    model_save_freq = 0
    # model_init_flag = 0
    log_times = 0
    while True:
        # if model_init_flag == 0:
        #     weights_socket.send(pickle.dumps(agent.get_weights()))
        #     model_init_flag = 1

        if len(mem_pool) >= args.batch_size:
            # Sync weights to actor
            weights = agent.get_weights() # Retrieve current weights from the ppo agent
            # weights_socket.send(pickle.dumps(weights))

            if model_save_freq%args.ckpt_save_freq == 0:
                if args.ckpt_save_type == 'checkpoint':
                    agent.save(args.ckpt_path / 'ckpt') # Save full checkpoint
                elif args.ckpt_save_type == 'weight':
                    with open(args.ckpt_path / f'ppo{model_save_freq}.pth', 'wb') as f:
                        pickle.dump(weights, f) # Save model weights only

            if args.keep_training: # Off policy
                agent.update(mem_pool.sample(size=args.batch_size))
            else: # On policy
                with receiving_condition:
                    while num_receptions.value < args.training_freq:
                        receiving_condition.wait() # Wait for new data signal
                    data = mem_pool.sample(size=args.batch_size) # Sample batch
                    num_receptions.value -= args.training_freq
                stat = agent.update(data)
                model_update_times += 1
                weights = agent.get_weights() # Get new weights
                with open(f'/home/zhaoyp/guandan_tog/learner_torch/ckpt_bak/{model_update_times}.pth', 'wb') as f:
                    pickle.dump(weights, f)
                
                if model_update_times > 4:
                    os.remove(f'/home/zhaoyp/guandan_tog/learner_torch/ckpt_bak/{model_update_times - 5}.pth') # Rotate old files
                # print(stat)
                # if log_times%1000 == 0:
                
                # Logging
                if log_times%100 == 0:
                    stats = defaultdict(list)
                    for k, v in stat.items():
                        stats[k].append(v)
                    stat = {k: np.array(v).mean() for k, v in stats.items()}
                    if stat is not None:
                        for k, v in stat.items():
                            logger.record_tabular(k, v)  # Log training stat
                        logger.record_tabular('model_id', model_save_freq) # Save model version
                    logger.dump_tabular()  # Print log to terminal or file
                else:
                    log_times += 1

            model_save_freq += 1


def recv_data(data_port, mem_pool, receiving_condition, num_receptions, keep_training):
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.bind(f'tcp://*:{data_port}')

    while True:
        data: dict = deserialize(data_socket.recv()) # BLOCKING
        data_socket.send(b'200') 

        if keep_training:
            mem_pool.push(data)
        else:
            with receiving_condition:
                mem_pool.push(data)
                num_receptions.value += 1
                receiving_condition.notify() # Signals the trainer that new data is ready


if __name__ == '__main__':
    main()
