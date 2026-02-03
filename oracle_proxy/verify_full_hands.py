#!/usr/bin/env python3
"""
Robust verification test for Oracle Proxy.
Shows FULL hand cards (not just counts) for all players.
"""

import asyncio
import json
import subprocess
import os
import sys
import signal

# Use ws4py like the original game.py does (synchronous, more stable)
from ws4py.client.threadedclient import WebSocketClient
import threading
import time

ACTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/actor_n"
PROXY_DIR = os.path.dirname(os.path.abspath(__file__))


class VerificationClient(WebSocketClient):
    """Synchronous WebSocket client for verification"""
    
    def __init__(self, url, client_id, results):
        super().__init__(url)
        self.client_id = client_id
        self.results = results
        self.my_pos = -1
        self.act_count = 0
        self.max_acts = 3
        self.done = threading.Event()
        
    def opened(self):
        print(f"[Client {self.client_id}] Connected")
    
    def closed(self, code, reason=None):
        print(f"[Client {self.client_id}] Closed: {code} {reason}")
        self.done.set()
    
    def received_message(self, message):
        try:
            msg = json.loads(str(message))
            msg_type = msg.get('type', '')
            stage = msg.get('stage', '')
            
            if msg_type == 'notify' and stage == 'beginning':
                self.my_pos = msg.get('myPos', -1)
                my_cards = msg.get('handCards', [])
                print(f"[Client {self.client_id}] Position={self.my_pos}, My cards ({len(my_cards)}): {my_cards[:5]}...")
            
            elif msg_type == 'act' and stage == 'play':
                all_hands = msg.get('allHandCards', None)
                oracle_enabled = msg.get('oracleEnabled', False)
                my_cards = msg.get('handCards', [])
                
                self.act_count += 1
                
                if all_hands and self.act_count <= 2:
                    print(f"\n{'='*60}")
                    print(f"[Client {self.client_id}] ACT #{self.act_count} - Position {self.my_pos}'s turn")
                    print(f"{'='*60}")
                    print(f"Oracle Enabled: {oracle_enabled}")
                    print(f"My hand ({len(my_cards)} cards): {my_cards}")
                    print(f"\n*** ALL PLAYERS' HANDS (Oracle Info) ***")
                    for pos, hand in enumerate(all_hands):
                        marker = " <-- ME" if pos == self.my_pos else ""
                        print(f"  Position {pos} ({len(hand):2d} cards): {hand}{marker}")
                    print(f"{'='*60}\n")
                    
                    # Record success
                    self.results[self.client_id] = {
                        'oracle_works': True,
                        'all_hands': all_hands,
                        'my_pos': self.my_pos
                    }
                
                # Send action (pick last one to progress game)
                action_list = msg.get('actionList', [[]])
                idx = len(action_list) - 1
                self.send(json.dumps({"actIndex": idx}))
                
                if self.act_count >= self.max_acts:
                    self.done.set()
            
            elif msg_type == 'notify' and stage == 'episodeOver':
                print(f"[Client {self.client_id}] Episode over")
                self.done.set()
                
        except Exception as e:
            print(f"[Client {self.client_id}] Error processing message: {e}")


def run_verification():
    processes = []
    clients = []
    
    try:
        print("=" * 60)
        print("ORACLE PROXY VERIFICATION TEST")
        print("Shows FULL hand cards for all players")
        print("=" * 60)
        
        # Kill any existing processes
        os.system("pkill -f 'guandan' 2>/dev/null")
        os.system("pkill -f 'proxy_server' 2>/dev/null")
        time.sleep(1)
        
        # Start guandan server
        print("\n[1] Starting guandan server on port 23456...")
        server_proc = subprocess.Popen(
            [f"{ACTOR_DIR}/guandan", "10"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=ACTOR_DIR
        )
        processes.append(server_proc)
        time.sleep(2)
        
        if server_proc.poll() is not None:
            print("ERROR: Server failed to start")
            return False
        print("    ✓ Server started")
        
        # Start proxy
        print("[2] Starting oracle proxy on port 23457...")
        proxy_proc = subprocess.Popen(
            ["python3", f"{PROXY_DIR}/proxy_server.py", 
             "--proxy-port", "23457", "--upstream-port", "23456"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=PROXY_DIR
        )
        processes.append(proxy_proc)
        time.sleep(2)
        
        if proxy_proc.poll() is not None:
            print("ERROR: Proxy failed to start")
            return False
        print("    ✓ Proxy started")
        
        # Connect clients
        print("[3] Connecting 4 clients...\n")
        
        results = [{} for _ in range(4)]
        
        for i in range(4):
            url = f"ws://localhost:23457/game/client{i}"
            client = VerificationClient(url, i, results)
            client.connect()
            clients.append(client)
            time.sleep(0.2)
        
        # Wait for clients to finish (max 30 seconds)
        print("[4] Waiting for game actions...\n")
        timeout = 30
        start = time.time()
        
        while time.time() - start < timeout:
            all_done = all(c.done.is_set() for c in clients)
            any_success = any(r.get('oracle_works', False) for r in results)
            
            if all_done or (any_success and time.time() - start > 5):
                break
            time.sleep(0.5)
        
        # Results
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        
        success_count = sum(1 for r in results if r.get('oracle_works', False))
        
        for i, r in enumerate(results):
            if r.get('oracle_works'):
                pos = r.get('my_pos', -1)
                all_hands = r.get('all_hands', [])
                total_cards = sum(len(h) for h in all_hands)
                print(f"  Client {i} (Pos {pos}): ✓ Oracle works - saw {total_cards} total cards")
            else:
                print(f"  Client {i}: ✗ No oracle data received")
        
        print(f"\nOracle success: {success_count}/4 clients")
        
        if success_count > 0:
            print("\n✅ VERIFICATION PASSED!")
            print("   Players can see ALL cards (not just counts) of all players!")
            return True
        else:
            print("\n❌ VERIFICATION FAILED!")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\n[5] Cleaning up...")
        for client in clients:
            try:
                client.close()
            except:
                pass
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                try:
                    proc.kill()
                except:
                    pass
        print("Done.")


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
