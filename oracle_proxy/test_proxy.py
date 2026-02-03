#!/usr/bin/env python3
"""
Test script to verify the Oracle Proxy Server works correctly.
This connects 4 simple test clients and prints received messages.
"""

import asyncio
import json
import websockets
import sys

class TestClient:
    def __init__(self, client_id: int, proxy_url: str):
        self.client_id = client_id
        self.proxy_url = proxy_url
        self.received_oracle = False
        
    async def run(self):
        try:
            async with websockets.connect(f"{self.proxy_url}/game/client{self.client_id}") as ws:
                print(f"[Client {self.client_id}] Connected to {self.proxy_url}")
                
                async for message in ws:
                    try:
                        msg = json.loads(message)
                        msg_type = msg.get('type', '')
                        stage = msg.get('stage', '')
                        
                        if msg_type == 'notify' and stage == 'beginning':
                            hand_cards = msg.get('handCards', [])
                            my_pos = msg.get('myPos', -1)
                            print(f"[Client {self.client_id}] Beginning: myPos={my_pos}, cards={len(hand_cards)}")
                        
                        elif msg_type == 'act':
                            hand_cards = msg.get('handCards', [])
                            all_hand_cards = msg.get('allHandCards', None)
                            oracle_enabled = msg.get('oracleEnabled', False)
                            action_list = msg.get('actionList', [])
                            
                            print(f"\n[Client {self.client_id}] ACT message received:")
                            print(f"  - My cards: {len(hand_cards)}")
                            print(f"  - Oracle enabled: {oracle_enabled}")
                            
                            if all_hand_cards:
                                self.received_oracle = True
                                print(f"  - ALL HAND CARDS (oracle):")
                                for pos, cards in enumerate(all_hand_cards):
                                    print(f"    Position {pos}: {len(cards)} cards")
                            else:
                                print(f"  - No oracle info (allHandCards missing)")
                            
                            # Send PASS action
                            if action_list:
                                response = {"actIndex": 0}  # PASS is usually index 0
                                await ws.send(json.dumps(response))
                                print(f"[Client {self.client_id}] Sent PASS")
                        
                        elif msg_type == 'notify' and stage == 'episodeOver':
                            print(f"[Client {self.client_id}] Episode over")
                            break
                            
                    except json.JSONDecodeError:
                        print(f"[Client {self.client_id}] Invalid JSON: {message[:100]}")
                        
        except Exception as e:
            print(f"[Client {self.client_id}] Error: {e}")
        
        return self.received_oracle


async def main():
    proxy_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:23457"
    print(f"Testing Oracle Proxy at {proxy_url}")
    print("=" * 50)
    
    # Start 4 clients concurrently
    clients = [TestClient(i, proxy_url) for i in range(4)]
    tasks = [asyncio.create_task(client.run()) for client in clients]
    
    # Wait for all with timeout
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
        
        print("\n" + "=" * 50)
        print("RESULTS:")
        success = all(results)
        for i, received in enumerate(results):
            status = "✓ Oracle received" if received else "✗ No oracle"
            print(f"  Client {i}: {status}")
        
        if success:
            print("\n✓ Oracle Proxy is working correctly!")
        else:
            print("\n✗ Oracle info not received by all clients")
            
    except asyncio.TimeoutError:
        print("\nTimeout - test took too long")
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
