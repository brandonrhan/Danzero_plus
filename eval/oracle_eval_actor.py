#!/usr/bin/env python3
"""
Oracle Eval Actor - 评估模式的推理服务

只做推理，不发送训练数据
加载最新的模型权重进行评估
"""

import os
import sys
import time
import pickle
from argparse import ArgumentParser
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import zmq
from pyarrow import deserialize, serialize
from tensorflow.keras.backend import set_session

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from actor_n.model import GDOracleModel, ORACLE_INPUT_DIM

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model checkpoint directory or file')
parser.add_argument('--positions', type=str, default='0,2',
                    help='Comma-separated positions to serve (e.g., "0,2")')
parser.add_argument('--base_port', type=int, default=6000,
                    help='Base port for actor services')


class OracleEvalActor:
    """评估模式的Oracle Actor"""
    
    def __init__(self, model_path: str, position: int, port: int):
        self.position = position
        self.port = port
        
        # TensorFlow 配置
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        # 加载模型
        self.model = GDOracleModel((ORACLE_INPUT_DIM,), (5, 216))
        self._load_model(model_path)
        
        # ZMQ 服务
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://*:{port}')
        
        print(f"[Actor Pos {position}] Loaded model, listening on port {port}")
    
    def _load_model(self, model_path: str):
        """加载模型权重"""
        # 查找最新的检查点
        if os.path.isdir(model_path):
            # 支持 .ckpt, .npy, .pkl 文件
            ckpt_files = [f for f in os.listdir(model_path) 
                         if f.endswith('.ckpt') or f.endswith('.npy') or f.endswith('.pkl')]
            if ckpt_files:
                # 按数字排序选择最新的（文件名格式: 75065.ckpt）
                ckpt_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0, reverse=True)
                model_path = os.path.join(model_path, ckpt_files[0])
        
        if os.path.exists(model_path):
            print(f"[Actor Pos {self.position}] Loading weights from {model_path}")
            if model_path.endswith('.npy'):
                weights = np.load(model_path, allow_pickle=True)
            elif model_path.endswith('.pkl') or model_path.endswith('.ckpt'):
                # .ckpt 文件也是 pickle 格式
                with open(model_path, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = np.load(model_path, allow_pickle=True)
            
            self.model.set_weights(weights)
            print(f"[Actor Pos {self.position}] Loaded weights successfully")
        else:
            print(f"[Actor Pos {self.position}] WARNING: No weights found at {model_path}, using random init")
    
    def run(self):
        """运行推理服务"""
        print(f"[Actor Pos {self.position}] Ready to serve")
        
        while True:
            try:
                # 接收状态
                state = deserialize(self.socket.recv())
                
                # 推理
                x_batch = state['x_batch']
                output = self.model.forward(x_batch)
                # output shape: (num_actions, 1) - 每个动作的Q值
                # 选择Q值最大的动作索引
                action_idx = int(np.argmax(output.flatten()))
                
                # 返回动作索引
                self.socket.send(serialize(action_idx).to_buffer())
                
            except KeyboardInterrupt:
                print(f"[Actor Pos {self.position}] Shutting down")
                break
            except Exception as e:
                print(f"[Actor Pos {self.position}] Error: {e}")
                # 返回默认动作
                self.socket.send(serialize(0).to_buffer())


def run_actor(model_path: str, position: int, port: int):
    """运行单个 actor 进程"""
    import sys
    sys.stdout.flush()
    print(f"[Actor Pos {position}] Starting...", flush=True)
    try:
        actor = OracleEvalActor(model_path, position, port)
        actor.run()
    except Exception as e:
        import traceback
        print(f"[Actor Pos {position}] FATAL ERROR: {e}", flush=True)
        traceback.print_exc()


def main():
    args, _ = parser.parse_known_args()
    
    print(f"Model path: {args.model_path}", flush=True)
    print(f"Starting actors...", flush=True)
    
    positions = [int(p) for p in args.positions.split(',')]
    
    processes = []
    for pos in positions:
        port = args.base_port + pos
        p = Process(target=run_actor, args=(args.model_path, pos, port))
        p.start()
        processes.append(p)
        time.sleep(1)  # 增加延迟确保初始化完成
    
    print(f"Started actors for positions: {positions}", flush=True)
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping all actors...")
        for p in processes:
            p.terminate()


if __name__ == '__main__':
    main()
