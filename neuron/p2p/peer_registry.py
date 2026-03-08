# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# neuron/p2p/peer_registry.py
import json
import sqlite3
from datetime import datetime

class PeerRegistry:
    DB_PATH = 'peer_registry.db'

    def __init__(self):
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS peers (
                    peer_id TEXT PRIMARY KEY,
                    addrs TEXT,
                    manifest TEXT,
                    last_seen TIMESTAMP,
                    reputation REAL DEFAULT 1.0
                )
            ''')

    def add_or_update(self, peer_id, addrs, manifest):
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO peers (peer_id, addrs, manifest, last_seen)
                VALUES (?, ?, ?, ?)
            ''', (peer_id, json.dumps(addrs), json.dumps(manifest), datetime.utcnow()))

    def get_peers_by_capability(self, domain):
        with sqlite3.connect(self.DB_PATH) as conn:
            # Simple query – for production you'd want JSON query support.
            cur = conn.execute('SELECT peer_id, addrs, manifest FROM peers')
            results = []
            for row in cur:
                manifest = json.loads(row[2])
                for vni in manifest.get('vnis', []):
                    if vni.get('domain') == domain:
                        results.append({
                            'peer_id': row[0],
                            'addrs': json.loads(row[1]),
                            'manifest': manifest
                        })
                        break
            return results

    def get_all_peers(self):
        with sqlite3.connect(self.DB_PATH) as conn:
            cur = conn.execute('SELECT peer_id, addrs, manifest, last_seen, reputation FROM peers')
            return [
                {
                    'peer_id': row[0],
                    'addrs': json.loads(row[1]),
                    'manifest': json.loads(row[2]),
                    'last_seen': row[3],
                    'reputation': row[4]
                }
                for row in cur
            ] 
