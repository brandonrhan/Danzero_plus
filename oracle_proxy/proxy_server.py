#!/usr/bin/env python3
"""
Oracle Proxy Server for Guandan

This proxy sits between the original guandan server and bot clients.
It intercepts messages and injects oracle information (all players' hand cards).

Architecture:
    Bots (0-3) ←→ Proxy (port 23457) ←→ Original Server (port 23456)

The proxy:
1. Collects hand cards from all players at game start (notify/beginning)
2. Tracks card plays and updates hands accordingly
3. Injects 'allHandCards' field into 'act' messages
"""

import asyncio
import json
import websockets
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PlayerState:
    """Track individual player's state"""
    hand_cards: List[str] = field(default_factory=list)
    position: int = -1
    websocket: Optional[object] = None


class OracleProxyServer:
    """Proxy server that injects oracle information into game messages"""
    
    def __init__(self, proxy_port: int = 23457, upstream_host: str = "localhost", upstream_port: int = 23456):
        self.proxy_port = proxy_port
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        
        # Player states indexed by position (0-3)
        self.players: Dict[int, PlayerState] = {}
        
        # Map client websocket to upstream websocket
        self.client_to_upstream: Dict[object, object] = {}
        self.upstream_to_client: Dict[object, object] = {}
        
        # Map client websocket to player position
        self.client_to_position: Dict[object, int] = {}
        
        # Track which positions have connected
        self.connected_positions: set = set()
        
        # Lock for thread-safe operations
        self.lock = asyncio.Lock()
    
    def get_all_hand_cards(self) -> List[List[str]]:
        """Get all players' hand cards as a list of lists"""
        result = [[], [], [], []]
        for pos, player in self.players.items():
            if 0 <= pos < 4:
                result[pos] = player.hand_cards.copy()
        return result
    
    def update_hand_after_play(self, position: int, played_cards: List[str]):
        """Remove played cards from player's hand"""
        if position not in self.players:
            return
        
        player = self.players[position]
        for card in played_cards:
            if card in player.hand_cards:
                player.hand_cards.remove(card)
    
    def inject_oracle_info(self, msg: dict, for_position: int) -> dict:
        """Inject oracle information into act message"""
        if msg.get('type') == 'act':
            # Add all players' hand cards
            msg['allHandCards'] = self.get_all_hand_cards()
            msg['oracleEnabled'] = True
            logger.debug(f"Injected oracle info for position {for_position}")
        return msg
    
    async def handle_upstream_message(self, msg_str: str, client_ws, position: int) -> str:
        """Process message from upstream server before forwarding to client"""
        try:
            msg = json.loads(msg_str)
        except json.JSONDecodeError:
            return msg_str
        
        async with self.lock:
            msg_type = msg.get('type', '')
            stage = msg.get('stage', '')
            
            # Handle beginning notification - collect initial hand cards
            if msg_type == 'notify' and stage == 'beginning':
                hand_cards = msg.get('handCards', [])
                my_pos = msg.get('myPos', position)
                
                if my_pos not in self.players:
                    self.players[my_pos] = PlayerState()
                
                self.players[my_pos].hand_cards = hand_cards.copy()
                self.players[my_pos].position = my_pos
                self.players[my_pos].websocket = client_ws
                self.client_to_position[client_ws] = my_pos
                
                # 显示具体手牌（只在第一局显示前5张）
                sample_cards = hand_cards[:5] if len(hand_cards) > 5 else hand_cards
                logger.info(f"Player {my_pos} received {len(hand_cards)} cards: {sample_cards}...")
            
            # Handle play notification - update hand cards
            elif msg_type == 'notify' and stage == 'play':
                cur_pos = msg.get('curPos', -1)
                cur_action = msg.get('curAction', [])
                
                if cur_action and len(cur_action) >= 3:
                    action_type = cur_action[0]  # 牌型
                    action_rank = cur_action[1]  # 等级
                    played_cards = cur_action[2]  # [type, rank, cards]
                    if isinstance(played_cards, list):
                        self.update_hand_after_play(cur_pos, played_cards)
                        # 详细记录出牌信息
                        cards_str = ','.join(played_cards) if played_cards else 'PASS'
                        remaining = len(self.players[cur_pos].hand_cards) if cur_pos in self.players else '?'
                        logger.info(f"P{cur_pos} played: [{action_type}] {cards_str} (remaining: {remaining})")
            
            # Handle episode over - reset for next round
            elif msg_type == 'notify' and stage == 'episodeOver':
                logger.info("Episode over, hands will be reset on next beginning")
            
            # Handle act message - inject oracle info
            elif msg_type == 'act':
                msg = self.inject_oracle_info(msg, position)
        
        return json.dumps(msg)
    
    async def handle_client_message(self, msg_str: str, client_ws, position: int) -> str:
        """Process message from client before forwarding to upstream"""
        # Currently just pass through, but could add tracking here
        return msg_str
    
    async def forward_messages(self, source, dest, handler, position: int, direction: str):
        """Forward messages between source and dest, applying handler"""
        try:
            async for message in source:
                processed = await handler(message, source if direction == "upstream" else dest, position)
                await dest.send(processed)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for position {position} ({direction})")
        except Exception as e:
            logger.error(f"Error in {direction} forwarding: {e}")
    
    async def handle_client(self, client_ws, path=None):
        """Handle a new client connection"""
        # Extract path from websocket request (compatible with different websockets versions)
        if path is None:
            # Try different ways to get the path
            if hasattr(client_ws, 'path'):
                path = client_ws.path
            elif hasattr(client_ws, 'request') and hasattr(client_ws.request, 'path'):
                path = client_ws.request.path
            else:
                path = ""
        
        position = len(self.connected_positions)
        self.connected_positions.add(position)
        
        logger.info(f"New client connected, assigned position {position}, path={path}")
        
        # Connect to upstream server with the same path
        upstream_uri = f"ws://{self.upstream_host}:{self.upstream_port}{path}"
        try:
            async with websockets.connect(upstream_uri) as upstream_ws:
                self.client_to_upstream[client_ws] = upstream_ws
                self.upstream_to_client[upstream_ws] = client_ws
                
                # Create bidirectional forwarding tasks
                client_to_upstream = asyncio.create_task(
                    self.forward_messages(client_ws, upstream_ws, 
                                         self.handle_client_message, position, "client")
                )
                upstream_to_client = asyncio.create_task(
                    self.forward_messages(upstream_ws, client_ws,
                                         self.handle_upstream_message, position, "upstream")
                )
                
                # Wait for either direction to complete
                done, pending = await asyncio.wait(
                    [client_to_upstream, upstream_to_client],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel the remaining task
                for task in pending:
                    task.cancel()
                    
        except Exception as e:
            logger.error(f"Error connecting to upstream: {e}")
        finally:
            # Cleanup
            if client_ws in self.client_to_position:
                pos = self.client_to_position[client_ws]
                self.connected_positions.discard(pos)
                del self.client_to_position[client_ws]
            if client_ws in self.client_to_upstream:
                del self.client_to_upstream[client_ws]
            logger.info(f"Client at position {position} disconnected")
    
    async def start(self):
        """Start the proxy server"""
        logger.info(f"Starting Oracle Proxy Server on port {self.proxy_port}")
        logger.info(f"Forwarding to upstream server at {self.upstream_host}:{self.upstream_port}")
        
        async with websockets.serve(self.handle_client, "0.0.0.0", self.proxy_port):
            logger.info("Proxy server is running. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Oracle Proxy Server for Guandan")
    parser.add_argument("--proxy-port", type=int, default=23457, help="Port for proxy server")
    parser.add_argument("--upstream-host", type=str, default="localhost", help="Upstream server host")
    parser.add_argument("--upstream-port", type=int, default=23456, help="Upstream server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    proxy = OracleProxyServer(
        proxy_port=args.proxy_port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port
    )
    await proxy.start()


if __name__ == "__main__":
    asyncio.run(main())
