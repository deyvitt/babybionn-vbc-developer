# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

import os
import json
import libp2p
import asyncio
import logging
from pathlib import Path
from .peer_registry import PeerRegistry
from .protocols import register_handlers
from .discovery import start_mdns, start_dht
from libp2p.crypto.ed25519 import Ed25519PrivateKey
from libp2p.peer.peerinfo import info_from_p2p_addr

logger = logging.getLogger(__name__)

class P2PNode:
    def __init__(self, listen_addr='/ip4/0.0.0.0/tcp/9000', key_path=None):
        self.listen_addr = listen_addr
        self.key_path = key_path or str(Path.home() / '.babybionn' / 'peer_key')
        self.host = None
        self.peer_id = None
        self.peer_registry = PeerRegistry()

    async def start(self):
        # Ensure directory for key exists
        os.makedirs(os.path.dirname(self.key_path), exist_ok=True)

        # Load or create Ed25519 key
        key_pair = self._load_or_create_key()
        self.host = await libp2p.new_node(
            transport_opt=['/ip4/0.0.0.0/tcp/9000'],
            muxer_opt=['/mplex/6.7.0'],
            sec_opt=['/secio/1.0.0'],
            peer_key=key_pair
        )
        await self.host.get_network().listen(self.listen_addr)
        self.peer_id = self.host.get_id().pretty()
        logger.info(f"P2P Node started with ID: {self.peer_id}")

        # Register protocol handlers
        register_handlers(self.host)

        # Start discovery
        asyncio.create_task(start_mdns(self.host))
        asyncio.create_task(start_dht(self.host, bootstrap_nodes=[]))

        # Background task to persist peers
        asyncio.create_task(self._periodic_peer_save())

    def _load_or_create_key(self):
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                key_bytes = f.read()
            return Ed25519PrivateKey.deserialize(key_bytes)
        else:
            key = Ed25519PrivateKey.generate()
            with open(self.key_path, 'wb') as f:
                f.write(key.serialize())
            return key

    async def _periodic_peer_save(self):
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            # In a full implementation, you'd sync the peer registry to disk
            # self.peer_registry.save_to_disk()
            pass

    async def send_message(self, peer_id, protocol, data):
        """Send a JSON message to a peer and return response."""
        stream = await self.host.new_stream(peer_id, [protocol])
        await stream.write(json.dumps(data).encode())
        response = await stream.read()
        return json.loads(response.decode()) 
