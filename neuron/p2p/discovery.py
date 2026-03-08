# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# neuron/p2p/discovery.py
import asyncio
import logging

from libp2p.discovery import mdns
from libp2p.routing.kademlia import KademliaServer

logger = logging.getLogger(__name__)

async def start_mdns(host):
    """Start mDNS discovery on the local network."""
    discovery = mdns.MDNS(host)
    await discovery.start()
    logger.info("mDNS discovery started")

async def start_dht(host, bootstrap_nodes):
    """Start DHT discovery with given bootstrap nodes."""
    dht = KademliaServer(host)
    if bootstrap_nodes:
        await dht.bootstrap(bootstrap_nodes)
    host.set_routing(dht)
    logger.info("DHT discovery started")
    # Optionally, start a periodic bootstrap loop
    while True:
        await asyncio.sleep(3600)  # rebootstrap every hour
        if bootstrap_nodes:
            await dht.bootstrap(bootstrap_nodes) 
