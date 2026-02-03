#!/usr/bin/env python3
"""
Standalone test for Oracle Proxy - no DanZero dependencies.
This tests the full flow: server -> proxy -> test clients
"""

import asyncio
import json
import websockets
import subprocess
import time
import os
import signal
import sys

ACTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/actor_n"
PROXY_DIR = os.path.dirname(os.path.abspath(__file__))


class TestBot:
    """Simple test bot that plays random actions"""
    def __init__(self, client_id: int, proxy_url: str):
        self.client_id = client_id
        self.proxy_url = proxy_url
        self.oracle_received_count = 0
        self.actions_taken = 0
        self.my_pos = -1
        
    async def run(self, max_actions=20):
        try:
            uri = f"{self.proxy_url}/game/client{self.client_id}"
            async with websockets.connect(uri) as ws:
                print(f"[Bot {self.client_id}] Connected")
                
                async for message in ws:
                    try:
                        msg = json.loads(message)
                        msg_type = msg.get('type', '')
                        stage = msg.get('stage', '')
                        
                        if msg_type == 'notify' and stage == 'beginning':
                            self.my_pos = msg.get('myPos', -1)
                            cards = len(msg.get('handCards', []))
                            print(f"[Bot {self.client_id}] Beginning: pos={self.my_pos}, cards={cards}")
                        
                        elif msg_type == 'act' and stage == 'play':
                            # Check for oracle info
                            all_hands = msg.get('allHandCards', None)
                            oracle_enabled = msg.get('oracleEnabled', False)
                            
                            if oracle_enabled and all_hands:
                                self.oracle_received_count += 1
                                hand_sizes = [len(h) for h in all_hands]
                                if self.oracle_received_count <= 3:
                                    print(f"[Bot {self.client_id}] Oracle: {hand_sizes}")
                            
                            # Send action (index 0 = usually PASS or first action)
                            action_list = msg.get('actionList', [[]])
                            await ws.send(json.dumps({"actIndex": 0}))
                            self.actions_taken += 1
                            
                            if self.actions_taken >= max_actions:
                                print(f"[Bot {self.client_id}] Max actions reached, stopping")
                                break
                        
                        elif msg_type == 'notify' and stage == 'episodeOver':
                            print(f"[Bot {self.client_id}] Episode over")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"[Bot {self.client_id}] JSON error: {e}")
                        
        except Exception as e:
            print(f"[Bot {self.client_id}] Error: {e}")
        
        return self.oracle_received_count


async def run_test(episodes=2):
    processes = []
    
    try:
        # Start guandan server
        print("Starting guandan server...")
        server_proc = subprocess.Popen(
            [f"{ACTOR_DIR}/guandan", str(episodes)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=ACTOR_DIR
        )
        processes.append(server_proc)
        await asyncio.sleep(2)
        
        if server_proc.poll() is not None:
            print("ERROR: Server failed to start")
            return False
        
        # Start proxy
        print("Starting oracle proxy...")
        proxy_proc = subprocess.Popen(
            ["python3", f"{PROXY_DIR}/proxy_server.py", 
             "--proxy-port", "23457", "--upstream-port", "23456"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=PROXY_DIR
        )
        processes.append(proxy_proc)
        await asyncio.sleep(2)
        
        if proxy_proc.poll() is not None:
            print("ERROR: Proxy failed to start")
            return False
        
        # Run test bots
        print("Starting test bots...")
        bots = [TestBot(i, "ws://localhost:23457") for i in range(4)]
        tasks = [asyncio.create_task(bot.run(max_actions=30)) for bot in bots]
        
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=60)
        
        print("\n" + "="*50)
        print("RESULTS:")
        total_oracle = sum(results)
        for i, count in enumerate(results):
            print(f"  Bot {i}: received oracle info {count} times")
        print(f"  Total: {total_oracle}")
        
        if total_oracle > 0:
            print("\n✓ Oracle Proxy is working correctly!")
            return True
        else:
            print("\n✗ No oracle info received")
            return False
            
    except asyncio.TimeoutError:
        print("Test timed out")
        return False
    finally:
        # Cleanup
        print("\nCleaning up...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                proc.kill()


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    success = asyncio.run(run_test(episodes))
    sys.exit(0 if success else 1)
