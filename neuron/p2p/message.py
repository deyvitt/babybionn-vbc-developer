# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# neuron/p2p/message.py
import json
from typing import List, Optional
from dataclasses import dataclass, asdict

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
