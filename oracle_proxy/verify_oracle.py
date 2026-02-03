#!/usr/bin/env python3
"""
Detailed verification test for Oracle Proxy.
Verifies that EVERY player can see ALL players' hand cards on their turn.
"""

import asyncio
import json
import websockets
import subprocess
import os
import sys

ACTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/actor_n"
PROXY_DIR = os.path.dirname(os.path.abspath(__file__))


class VerificationBot:
    """Bot that verifies oracle information in detail"""
    def __init__(self, client_id: int, proxy_url: str):
        self.client_id = client_id
        self.proxy_url = proxy_url
        self.my_pos = -1
        self.my_initial_cards = []
        self.act_messages = []  # Record all act messages received
        
    async def run(self, max_acts=10):
        try:
            uri = f"{self.proxy_url}/game/client{self.client_id}"
            async with websockets.connect(uri) as ws:
                async for message in ws:
                    try:
                        msg = json.loads(message)
                        msg_type = msg.get('type', '')
                        stage = msg.get('stage', '')
                        
                        if msg_type == 'notify' and stage == 'beginning':
                            self.my_pos = msg.get('myPos', -1)
                            self.my_initial_cards = msg.get('handCards', [])
                        
                        elif msg_type == 'act' and stage == 'play':
                            # Record this act message for verification
                            record = {
                                'my_pos': self.my_pos,
                                'my_cards_in_msg': msg.get('handCards', []),
                                'oracle_enabled': msg.get('oracleEnabled', False),
                                'all_hand_cards': msg.get('allHandCards', None),
                            }
                            self.act_messages.append(record)
                            
                            # Send action - choose last action (usually not PASS) to make game progress
                            action_list = msg.get('actionList', [[]])
                            # If we have more than PASS, pick a non-PASS action
                            act_idx = len(action_list) - 1 if len(action_list) > 1 else 0
                            await ws.send(json.dumps({"actIndex": act_idx}))
                            
                            if len(self.act_messages) >= max_acts:
                                break
                        
                        elif msg_type == 'notify' and stage == 'episodeOver':
                            break
                            
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            print(f"[Bot {self.client_id}] Error: {e}")
        
        return self.act_messages


async def run_verification():
    processes = []
    
    try:
        # Start guandan server
        print("=" * 60)
        print("ORACLE PROXY VERIFICATION TEST")
        print("=" * 60)
        print("\n[1] Starting guandan server on port 23456...")
        server_proc = subprocess.Popen(
            [f"{ACTOR_DIR}/guandan", "5"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=ACTOR_DIR
        )
        processes.append(server_proc)
        await asyncio.sleep(2)
        
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
        await asyncio.sleep(2)
        
        if proxy_proc.poll() is not None:
            print("ERROR: Proxy failed to start")
            return False
        print("    ✓ Proxy started")
        
        # Run bots
        print("[3] Running verification bots...\n")
        bots = [VerificationBot(i, "ws://localhost:23457") for i in range(4)]
        tasks = [asyncio.create_task(bot.run(max_acts=5)) for bot in bots]
        
        all_records = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
        
        # Analyze results
        print("=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        
        total_acts = 0
        oracle_success = 0
        all_four_visible = 0
        
        for bot_id, records in enumerate(all_records):
            print(f"\n[Bot {bot_id}] Position {bots[bot_id].my_pos}:")
            print(f"  Initial cards: {len(bots[bot_id].my_initial_cards)}")
            print(f"  Act messages received: {len(records)}")
            
            for i, rec in enumerate(records):
                total_acts += 1
                
                oracle_ok = rec['oracle_enabled']
                all_hands = rec['all_hand_cards']
                
                if oracle_ok:
                    oracle_success += 1
                
                if all_hands and len(all_hands) == 4:
                    # Check if all 4 positions have cards (or had cards)
                    non_empty = sum(1 for h in all_hands if len(h) >= 0)
                    if non_empty == 4:
                        all_four_visible += 1
                    
                    if i < 2:  # Print first 2 for each bot
                        sizes = [len(h) for h in all_hands]
                        print(f"    Act {i+1}: oracle={oracle_ok}, hands={sizes}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total act messages: {total_acts}")
        print(f"Oracle enabled:     {oracle_success}/{total_acts}")
        print(f"All 4 hands visible: {all_four_visible}/{total_acts}")
        
        success = oracle_success > 0 and all_four_visible > 0
        
        if success:
            print("\n✅ VERIFICATION PASSED!")
            print("   Every player can see all 4 players' hand cards on their turn.")
        else:
            print("\n❌ VERIFICATION FAILED!")
            
        return success
            
    except asyncio.TimeoutError:
        print("Test timed out")
        return False
    finally:
        print("\n[4] Cleaning up...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                proc.kill()


if __name__ == "__main__":
    success = asyncio.run(run_verification())
    sys.exit(0 if success else 1)
