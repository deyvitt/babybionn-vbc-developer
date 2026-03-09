# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

import asyncio
import logging

from libp2p.discovery import mdns
from libp2p.kad_dht import KadDHT  # Changed from libp2p.routing.kademlia

logger = logging.getLogger(__name__)

async def start_mdns(host):
    """Start mDNS discovery on the local network."""
    try:
        # Check if mdns has the expected interface
        if hasattr(mdns, 'MDNS'):
            discovery = mdns.MDNS(host)
            await discovery.start()
            logger.info("mDNS discovery started")
        else:
            # Alternative initialization for newer versions
            logger.warning("mDNS MDNS class not found, trying alternative")
            # You might need to check the actual mdns API
            await mdns.start(host)
    except Exception as e:
        logger.error(f"Failed to start mDNS: {e}")

async def start_dht(host, bootstrap_nodes):
    """Start DHT discovery with given bootstrap nodes."""
    try:
        # Create Kademlia DHT instance
        dht = KadDHT(host)
        
        # Start the DHT
        await dht.start()
        
        # Bootstrap with provided nodes
        if bootstrap_nodes:
            await dht.bootstrap(bootstrap_nodes)
        
        # Set the routing system on the host
        host.set_routing(dht)
        
        logger.info("DHT discovery started")
        
        # Periodic bootstrap loop
        while True:
            await asyncio.sleep(3600)  # rebootstrap every hour
            if bootstrap_nodes:
                try:
                    await dht.bootstrap(bootstrap_nodes)
                except Exception as e:
                    logger.error(f"Failed to rebootstrap DHT: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to start DHT: {e}")
