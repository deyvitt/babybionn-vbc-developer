# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# neuron/p2p/protocols.py
import os
import json
import logging
from typing import List, Optional
from .peer_registry import PeerRegistry
from dataclasses import dataclass, asdict
from .messages import IdentifyMessage, QueryMessage, QueryResponse, to_json, from_json

# These will be set from main.py
_aggregator = None
_mesh_core = None

logger = logging.getLogger(__name__)

def set_aggregator(agg):
    """Set the global aggregator reference for protocol handlers."""
    global _aggregator
    _aggregator = agg

def set_mesh_core(core):
    """Set the global mesh core reference for protocol handlers."""
    global _mesh_core
    _mesh_core = core

# ==================== PROTOCOL HANDLERS ====================

async def handle_identify(stream):
    """Handle /babybionn/identify/1.0.0 – exchange manifests."""
    data = await stream.read()
    try:
        msg = json.loads(data.decode())
        # Store the remote peer's info in registry
        peer_id = stream.muxed_conn.peer_id
        addrs = msg.get('listen_addrs', [])
        manifest = msg.get('vnis', [])
        PeerRegistry().add_or_update(peer_id, addrs, {'vnis': manifest})

        # Respond with our own manifest
        if _mesh_core is None:
            raise RuntimeError("Mesh core not set")
        our_manifest = _mesh_core.get_capability_manifest()
        response = {
            'node_id': _mesh_core.node_id,
            'listen_addrs': [f"/ip4/0.0.0.0/tcp/{os.getenv('P2P_PORT', '9000')}"],
            'vnis': our_manifest,
            'version': '1.0.0'
        }
        await stream.write(json.dumps(response).encode())
    except Exception as e:
        logger.error(f"Identify handler error: {e}")
    finally:
        await stream.close()

async def handle_query(stream):
    """Handle /babybionn/query/1.0.0 – process a query from a remote VBC."""
    data = await stream.read()
    try:
        query = json.loads(data.decode())
        if _aggregator is None:
            raise RuntimeError("Aggregator not set")
        # Forward to aggregator's remote query handler
        result = await _aggregator.handle_remote_query(query)
        await stream.write(json.dumps(result).encode())
    except Exception as e:
        logger.error(f"Query handler error: {e}")
        # Send back an error response
        error_resp = {'error': str(e), 'query_id': query.get('query_id', '')}
        await stream.write(json.dumps(error_resp).encode())
    finally:
        await stream.close()

def register_handlers(host):
    """Register all protocol handlers with the libp2p host."""
    host.set_stream_handler('/babybionn/identify/1.0.0', handle_identify)
    host.set_stream_handler('/babybionn/query/1.0.0', handle_query)
    logger.info("Protocol handlers registered")

# ==================== MESSAGE DEFINITIONS (unchanged) ====================

@dataclass
class IdentifyMessage:
    """Node identification and capability manifest."""
    node_id: str
    listen_addrs: List[str]
    vnis: List[dict]  # each dict: { "type": str, "domain": str, "subdomains": List[str] }
    version: str = "1.0.0"

@dataclass
class QueryMessage:
    """A query to be processed by a remote VNI."""
    query_id: str
    query_text: str
    session_id: str
    target_domain: Optional[str] = None
    ttl: int = 3

@dataclass
class QueryResponse:
    """Response from a remote VNI."""
    query_id: str
    responder_id: str
    response_text: str
    confidence: float
    vni_domain: str
    processing_time: float  # seconds

# Serialization helpers
def to_json(obj):
    return json.dumps(asdict(obj))

def from_json(cls, data):
    return cls(**json.loads(data))
