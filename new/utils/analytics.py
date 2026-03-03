"""
REAL Learning Analytics Module - Complete Implementation
"""
import logging
import json
import os
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import numpy as np
import networkx as nx  # ADDED for SynapticVisualizer
import asyncio  # ADDED for SynapticVisualizer
import plotly.graph_objects as go  # ADDED for SynapticVisualizer
from plotly.subplots import make_subplots  # ADDED for SynapticVisualizer

logger = logging.getLogger(__name__)

# ============================================================================
# ADDED: SynapticVisualizer Class (as requested by import error)
# ============================================================================

class SynapticVisualizer:
    """Visualize synaptic connections between VNIs"""
    
    def __init__(self, autonomy_engine=None):
        self.autonomy_engine = autonomy_engine
        self.graph = nx.DiGraph()
        self.update_interval = 30  # seconds
        self.history_size = 100
        self.connection_history = deque(maxlen=self.history_size)
        
    async def start_visualization(self):
        """Start background visualization updates"""
        logger.info("🔄 Starting synaptic visualization")
        
        while getattr(self.autonomy_engine, 'running', False):
            try:
                await asyncio.sleep(self.update_interval)
                await self.update_visualization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Visualization update error: {e}")
    
    async def update_visualization(self):
        """Update the synaptic connection graph"""
        if not self.autonomy_engine or not hasattr(self.autonomy_engine, 'orchestrator'):
            return
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes for each VNI
        for vni_id, vni_instance in self.autonomy_engine.orchestrator.vni_instances.items():
            vni_type = getattr(vni_instance, 'vni_type', 'unknown')
            self.graph.add_node(vni_id, type=vni_type, label=vni_id)
        
        # Add edges for connections
        for connection_id, pathway in self.autonomy_engine.orchestrator.synaptic_connections.items():
            if '→' in connection_id:
                source, target = connection_id.split('→')
                if source in self.graph.nodes and target in self.graph.nodes:
                    self.graph.add_edge(
                        source, 
                        target, 
                        weight=getattr(pathway, 'strength', 0.5),
                        label=f"{getattr(pathway, 'strength', 0.5):.2f}",
                        activation_count=getattr(pathway, 'activation_count', 0),
                        success_rate=getattr(pathway, 'success_count', 0) / max(getattr(pathway, 'activation_count', 1), 1)
                    )
        
        # Store history
        self.connection_history.append({
            'timestamp': datetime.now().isoformat(),
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'average_strength': self._calculate_average_strength()
        })
    
    def _calculate_average_strength(self) -> float:
        """Calculate average synaptic strength"""
        if self.graph.number_of_edges() == 0:
            return 0.0
        
        total_strength = sum(data.get('weight', 0) for _, _, data in self.graph.edges(data=True))
        return total_strength / self.graph.number_of_edges()
    
    def create_network_plot(self, figsize=(12, 8)):
        """Create a matplotlib network plot"""
        if self.graph.number_of_nodes() == 0:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate node positions
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Get node colors by type
        node_colors = []
        node_types = {
            'medical': 'lightcoral',
            'legal': 'lightblue',
            'general': 'lightgreen',
            'unknown': 'lightgray'
        }
        
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_colors.append(node_types.get(node_type, 'lightgray'))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=800,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges with weights
        edges = self.graph.edges(data=True)
        edge_weights = [data.get('weight', 0.5) * 3 for _, _, data in edges]
        
        nx.draw_networkx_edges(
            self.graph, pos,
            width=edge_weights,
            alpha=0.5,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, ax=ax)
        
        # Draw edge labels (weights)
        edge_labels = {(u, v): f"{d.get('weight', 0):.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        ax.set_title("Synaptic Connections Between VNIs", fontsize=16)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=label, alpha=0.8)
            for label, color in node_types.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plot(self):
        """Create an interactive Plotly network graph"""
        if self.graph.number_of_nodes() == 0:
            return None
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_weights = []
        edge_text = []
        
        pos = nx.spring_layout(self.graph, seed=42)
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = edge[2].get('weight', 0.5)
            edge_weights.append(weight)
            activation_count = edge[2].get('activation_count', 0)
            success_rate = edge[2].get('success_rate', 0)
            edge_text.append(
                f"Strength: {weight:.2f}<br>"
                f"Activations: {activation_count}<br>"
                f"Success: {success_rate:.1%}"
            )
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=[w * 3 for w in edge_weights], color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        node_types = {
            'medical': 'red',
            'legal': 'blue',
            'general': 'green',
            'unknown': 'gray'
        }
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_text.append(f"VNI: {node}<br>Type: {node_type}")
            node_colors.append(node_types.get(node_type, 'gray'))
            
            # Size based on connection strength
            in_strength = sum(data.get('weight', 0) for _, _, data in self.graph.in_edges(node, data=True))
            out_strength = sum(data.get('weight', 0) for _, _, data in self.graph.out_edges(node, data=True))
            node_sizes.append(20 + (in_strength + out_strength) * 30)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.split('_')[0] for node in self.graph.nodes()],
            textposition="top center",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='darkgray')
            ),
            textfont=dict(size=10)
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Synaptic Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get metrics about synaptic connections"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        metrics = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'average_strength': self._calculate_average_strength(),
            'strongest_connections': [],
            'weakest_connections': [],
            'node_centrality': {}
        }
        
        # Find strongest connections
        edges_with_weights = [(u, v, d.get('weight', 0)) for u, v, d in self.graph.edges(data=True)]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        
        metrics['strongest_connections'] = [
            {'source': u, 'target': v, 'strength': w}
            for u, v, w in edges_with_weights[:5]
        ]
        
        metrics['weakest_connections'] = [
            {'source': u, 'target': v, 'strength': w}
            for u, v, w in edges_with_weights[-5:]
        ]
        
        # Calculate centrality measures
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            for node in self.graph.nodes():
                metrics['node_centrality'][node] = {
                    'degree': degree_centrality.get(node, 0),
                    'betweenness': betweenness_centrality.get(node, 0),
                    'type': self.graph.nodes[node].get('type', 'unknown')
                }
        except:
            pass
        
        return metrics
    
    def create_metrics_dashboard(self) -> Dict[str, Any]:
        """Create a comprehensive metrics dashboard"""
        metrics = self.get_connection_metrics()
        
        # Add history trends
        if self.connection_history:
            history_data = list(self.connection_history)
            metrics['history_trends'] = {
                'timestamps': [h['timestamp'] for h in history_data],
                'node_counts': [h['node_count'] for h in history_data],
                'edge_counts': [h['edge_count'] for h in history_data],
                'average_strengths': [h['average_strength'] for h in history_data]
            }
        
        # Add VNI type distribution
        type_distribution = defaultdict(int)
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            type_distribution[node_type] += 1
        
        metrics['type_distribution'] = dict(type_distribution)
        
        return metrics

# ============================================================================
# ORIGINAL LearningAnalytics Class (keep as is)
# ============================================================================

class LearningAnalytics:
    """Complete learning analytics with real data storage, visualization, and insights"""
    
    def __init__(self, db_path: str = "./analytics/analytics.db"):
        self.db_path = db_path
        self.data_dir = os.path.dirname(db_path)
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Cache for performance
        self.cache = {
            'daily_stats': None,
            'top_vnis': None,
            'last_update': None
        }
        
        logger.info(f"✅ LearningAnalytics initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Interactions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            vni_id TEXT NOT NULL,
            vni_type TEXT NOT NULL,
            query TEXT,
            response TEXT,
            response_length INTEGER,
            confidence REAL,
            success BOOLEAN,
            generation_used BOOLEAN,
            session_id TEXT,
            processing_time_ms INTEGER,
            concepts_used TEXT,
            patterns_matched TEXT
        )
        ''')
        
        # VNI metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vni_metrics (
            vni_id TEXT PRIMARY KEY,
            vni_type TEXT NOT NULL,
            total_interactions INTEGER DEFAULT 0,
            successful_interactions INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            avg_response_time_ms REAL DEFAULT 0,
            last_used DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Learning patterns table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash TEXT UNIQUE,
            pattern_text TEXT,
            vni_type TEXT,
            success_rate REAL,
            usage_count INTEGER DEFAULT 0,
            first_seen DATETIME,
            last_used DATETIME
        )
        ''')
        
        # System metrics table (hourly aggregates)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            hour_start DATETIME PRIMARY KEY,
            total_queries INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            avg_response_time_ms REAL DEFAULT 0,
            error_rate REAL DEFAULT 0,
            unique_users INTEGER DEFAULT 0,
            vni_distribution TEXT
        )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_vni_id ON interactions(vni_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_hash ON learning_patterns(pattern_hash)')
        
        conn.commit()
        conn.close()
    
    def record_interaction(self, interaction_data: Dict[str, Any]):
        """Record a complete interaction with detailed metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert interaction
        cursor.execute('''
        INSERT INTO interactions 
        (timestamp, vni_id, vni_type, query, response, response_length, 
         confidence, success, generation_used, session_id, processing_time_ms,
         concepts_used, patterns_matched)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interaction_data.get('timestamp', datetime.now().isoformat()),
            interaction_data.get('vni_id', 'unknown'),
            interaction_data.get('vni_type', 'unknown'),
            interaction_data.get('query', '')[:500],
            interaction_data.get('response', '')[:2000],
            interaction_data.get('response_length', 0),
            interaction_data.get('confidence', 0.0),
            interaction_data.get('success', True),
            interaction_data.get('generation_used', False),
            interaction_data.get('session_id', 'default'),
            interaction_data.get('processing_time_ms', 0),
            json.dumps(interaction_data.get('concepts_used', [])),
            json.dumps(interaction_data.get('patterns_matched', []))
        ))
        
        # Update VNI metrics
        vni_id = interaction_data.get('vni_id', 'unknown')
        success = interaction_data.get('success', True)
        confidence = interaction_data.get('confidence', 0.0)
        processing_time = interaction_data.get('processing_time_ms', 0)
        
        # Get current metrics
        cursor.execute('SELECT * FROM vni_metrics WHERE vni_id = ?', (vni_id,))
        current = cursor.fetchone()
        
        if current:
            # Update existing
            total = current[2] + 1
            successful = current[3] + (1 if success else 0)
            avg_conf = ((current[4] * current[2]) + confidence) / total
            avg_time = ((current[5] * current[2]) + processing_time) / total
            
            cursor.execute('''
            UPDATE vni_metrics 
            SET total_interactions = ?, successful_interactions = ?, 
                avg_confidence = ?, avg_response_time_ms = ?, last_used = ?
            WHERE vni_id = ?
            ''', (total, successful, avg_conf, avg_time, datetime.now().isoformat(), vni_id))
        else:
            # Insert new
            cursor.execute('''
            INSERT INTO vni_metrics 
            (vni_id, vni_type, total_interactions, successful_interactions, 
             avg_confidence, avg_response_time_ms, last_used)
            VALUES (?, ?, 1, ?, ?, ?, ?)
            ''', (
                vni_id,
                interaction_data.get('vni_type', 'unknown'),
                1 if success else 0,
                confidence,
                processing_time,
                datetime.now().isoformat()
            ))
        
        # Record learning pattern if available
        if 'pattern_hash' in interaction_data:
            pattern_hash = interaction_data['pattern_hash']
            pattern_text = interaction_data.get('pattern_text', '')
            
            cursor.execute('SELECT * FROM learning_patterns WHERE pattern_hash = ?', (pattern_hash,))
            existing = cursor.fetchone()
            
            if existing:
                # Update pattern usage
                usage_count = existing[5] + 1
                success_rate = ((existing[4] * existing[5]) + (1 if success else 0)) / usage_count
                
                cursor.execute('''
                UPDATE learning_patterns 
                SET usage_count = ?, success_rate = ?, last_used = ?
                WHERE pattern_hash = ?
                ''', (usage_count, success_rate, datetime.now().isoformat(), pattern_hash))
            else:
                # Insert new pattern
                cursor.execute('''
                INSERT INTO learning_patterns 
                (pattern_hash, pattern_text, vni_type, success_rate, usage_count, first_seen, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_hash,
                    pattern_text,
                    interaction_data.get('vni_type', 'unknown'),
                    1.0 if success else 0.0,
                    1,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
        
        # Update hourly system metrics (aggregate)
        hour_start = datetime.now().replace(minute=0, second=0, microsecond=0).isoformat()
        
        cursor.execute('SELECT * FROM system_metrics WHERE hour_start = ?', (hour_start,))
        hour_metrics = cursor.fetchone()
        
        if hour_metrics:
            # Update existing hour
            new_total = hour_metrics[1] + 1
            new_avg_conf = ((hour_metrics[2] * hour_metrics[1]) + confidence) / new_total
            new_avg_time = ((hour_metrics[3] * hour_metrics[1]) + processing_time) / new_total
            new_error_rate = ((hour_metrics[4] * hour_metrics[1]) + (0 if success else 1)) / new_total
            
            # Update VNI distribution
            vni_dist = json.loads(hour_metrics[6]) if hour_metrics[6] else {}
            vni_id = interaction_data.get('vni_id', 'unknown')
            vni_dist[vni_id] = vni_dist.get(vni_id, 0) + 1
            
            cursor.execute('''
            UPDATE system_metrics 
            SET total_queries = ?, avg_confidence = ?, avg_response_time_ms = ?,
                error_rate = ?, vni_distribution = ?
            WHERE hour_start = ?
            ''', (new_total, new_avg_conf, new_avg_time, new_error_rate, 
                  json.dumps(vni_dist), hour_start))
        else:
            # Insert new hour
            vni_dist = {interaction_data.get('vni_id', 'unknown'): 1}
            cursor.execute('''
            INSERT INTO system_metrics 
            (hour_start, total_queries, avg_confidence, avg_response_time_ms,
             error_rate, unique_users, vni_distribution)
            VALUES (?, 1, ?, ?, ?, 1, ?)
            ''', (hour_start, confidence, processing_time, 
                  0 if success else 1.0, json.dumps(vni_dist)))
        
        conn.commit()
        conn.close()
        
        # Invalidate cache
        self.cache['last_update'] = datetime.now()
        self.cache['daily_stats'] = None
        self.cache['top_vnis'] = None
        
        logger.debug(f"📊 Recorded interaction: {interaction_data.get('vni_id')}, "
                    f"confidence: {confidence:.2f}, success: {success}")
    
    def get_detailed_metrics(self, time_period: str = "24h") -> Dict[str, Any]:
        """Get detailed metrics for specified time period"""
        conn = sqlite3.connect(self.db_path)
        
        # Calculate time filter
        if time_period == "24h":
            time_filter = datetime.now() - timedelta(hours=24)
        elif time_period == "7d":
            time_filter = datetime.now() - timedelta(days=7)
        elif time_period == "30d":
            time_filter = datetime.now() - timedelta(days=30)
        else:
            time_filter = datetime.now() - timedelta(hours=24)  # Default
        
        # Get interaction statistics
        cursor = conn.cursor()
        cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
            AVG(confidence) as avg_confidence,
            AVG(processing_time_ms) as avg_response_time,
            COUNT(DISTINCT vni_id) as unique_vnis,
            COUNT(DISTINCT session_id) as unique_sessions
        FROM interactions 
        WHERE timestamp >= ?
        ''', (time_filter.isoformat(),))
        
        stats = cursor.fetchone()
        
        # Get top VNIs by usage
        cursor.execute('''
        SELECT vni_id, vni_type, COUNT(*) as usage_count,
               AVG(confidence) as avg_confidence,
               AVG(processing_time_ms) as avg_response_time
        FROM interactions 
        WHERE timestamp >= ?
        GROUP BY vni_id, vni_type
        ORDER BY usage_count DESC
        LIMIT 10
        ''', (time_filter.isoformat(),))
        
        top_vnis = []
        for row in cursor.fetchall():
            top_vnis.append({
                'vni_id': row[0],
                'vni_type': row[1],
                'usage_count': row[2],
                'avg_confidence': round(row[3], 3),
                'avg_response_time_ms': round(row[4], 1)
            })
        
        # Get success rate over time (hourly)
        cursor.execute('''
        SELECT 
            strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
            COUNT(*) as total,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
            AVG(confidence) as avg_confidence
        FROM interactions 
        WHERE timestamp >= ?
        GROUP BY hour
        ORDER BY hour DESC
        LIMIT 24
        ''', (time_filter.isoformat(),))
        
        hourly_data = []
        for row in cursor.fetchall():
            hourly_data.append({
                'hour': row[0],
                'total': row[1],
                'success_rate': round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0,
                'avg_confidence': round(row[3], 3)
            })
        
        # Get concept usage
        cursor.execute('''
        SELECT concepts_used
        FROM interactions 
        WHERE timestamp >= ? AND concepts_used != '[]'
        ''', (time_filter.isoformat(),))
        
        concept_counts = defaultdict(int)
        for row in cursor.fetchall():
            try:
                concepts = json.loads(row[0])
                for concept in concepts:
                    concept_counts[concept] += 1
            except:
                continue
        
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Get generation usage stats
        cursor.execute('''
        SELECT 
            generation_used,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            AVG(processing_time_ms) as avg_response_time
        FROM interactions 
        WHERE timestamp >= ?
        GROUP BY generation_used
        ''', (time_filter.isoformat(),))
        
        generation_stats = {}
        for row in cursor.fetchall():
            key = 'with_generation' if row[0] == 1 else 'without_generation'
            generation_stats[key] = {
                'count': row[1],
                'avg_confidence': round(row[2], 3),
                'avg_response_time_ms': round(row[3], 1)
            }
        
        conn.close()
        
        return {
            'time_period': time_period,
            'total_interactions': stats[0],
            'successful_interactions': stats[1],
            'success_rate': round(stats[1] / stats[0] * 100, 1) if stats[0] > 0 else 0,
            'avg_confidence': round(stats[2], 3),
            'avg_response_time_ms': round(stats[3], 1),
            'unique_vnis': stats[4],
            'unique_sessions': stats[5],
            'top_vnis': top_vnis,
            'hourly_trends': hourly_data,
            'top_concepts': dict(top_concepts),
            'generation_stats': generation_stats,
            'time_filter': time_filter.isoformat(),
            'generated_at': datetime.now().isoformat()
        }
    
    def get_vni_performance(self, vni_id: str = None) -> Dict[str, Any]:
        """Get performance metrics for specific VNI or all VNIs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if vni_id:
            # Specific VNI
            cursor.execute('''
            SELECT * FROM vni_metrics WHERE vni_id = ?
            ''', (vni_id,))
            
            row = cursor.fetchone()
            if not row:
                return {'error': f'VNI {vni_id} not found'}
            
            # Get recent interactions for this VNI
            cursor.execute('''
            SELECT timestamp, query, confidence, success, generation_used
            FROM interactions 
            WHERE vni_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
            ''', (vni_id,))
            
            recent = []
            for r in cursor.fetchall():
                recent.append({
                    'timestamp': r[0],
                    'query': r[1][:100] + '...' if len(r[1]) > 100 else r[1],
                    'confidence': round(r[2], 3),
                    'success': bool(r[3]),
                    'generation_used': bool(r[4])
                })
            
            conn.close()
            
            return {
                'vni_id': row[0],
                'vni_type': row[1],
                'total_interactions': row[2],
                'successful_interactions': row[3],
                'success_rate': round(row[3] / row[2] * 100, 1) if row[2] > 0 else 0,
                'avg_confidence': round(row[4], 3),
                'avg_response_time_ms': round(row[5], 1),
                'last_used': row[6],
                'created_at': row[7],
                'recent_interactions': recent
            }
        else:
            # All VNIs
            cursor.execute('''
            SELECT * FROM vni_metrics ORDER BY total_interactions DESC
            ''')
            
            vnis = []
            for row in cursor.fetchall():
                vnis.append({
                    'vni_id': row[0],
                    'vni_type': row[1],
                    'total_interactions': row[2],
                    'successful_interactions': row[3],
                    'success_rate': round(row[3] / row[2] * 100, 1) if row[2] > 0 else 0,
                    'avg_confidence': round(row[4], 3),
                    'avg_response_time_ms': round(row[5], 1),
                    'last_used': row[6]
                })
            
            # Overall statistics
            cursor.execute('''
            SELECT 
                COUNT(DISTINCT vni_id),
                SUM(total_interactions),
                AVG(avg_confidence),
                AVG(avg_response_time_ms)
            FROM vni_metrics
            ''')
            
            overall = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_vnis': overall[0],
                'total_interactions': overall[1],
                'overall_avg_confidence': round(overall[2], 3),
                'overall_avg_response_time': round(overall[3], 1),
                'vnis': vnis
            }
    
    def export_to_csv(self, output_path: str = "./analytics/export"):
        """Export analytics data to CSV files"""
        os.makedirs(output_path, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        # Export interactions
        interactions_df = pd.read_sql_query("SELECT * FROM interactions", conn)
        interactions_df.to_csv(os.path.join(output_path, "interactions.csv"), index=False)
        
        # Export VNI metrics
        vni_metrics_df = pd.read_sql_query("SELECT * FROM vni_metrics", conn)
        vni_metrics_df.to_csv(os.path.join(output_path, "vni_metrics.csv"), index=False)
        
        # Export system metrics
        system_metrics_df = pd.read_sql_query("SELECT * FROM system_metrics", conn)
        system_metrics_df.to_csv(os.path.join(output_path, "system_metrics.csv"), index=False)
        
        conn.close()
        
        logger.info(f"📤 Exported analytics data to {output_path}")
        return {
            'interactions_count': len(interactions_df),
            'vni_count': len(vni_metrics_df),
            'output_path': output_path
        }
    
    def create_visualization(self, output_path: str = "./analytics/visualizations"):
        """Create visualization charts from analytics data"""
        os.makedirs(output_path, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        # 1. Success Rate Over Time
        df = pd.read_sql_query('''
        SELECT date(timestamp) as date,
               COUNT(*) as total,
               SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
        FROM interactions
        GROUP BY date(timestamp)
        ORDER BY date
        LIMIT 30
        ''', conn)
        
        if not df.empty:
            df['success_rate'] = df['successful'] / df['total'] * 100
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['success_rate'], marker='o', linewidth=2)
            plt.title('Success Rate Over Time')
            plt.xlabel('Date')
            plt.ylabel('Success Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'success_rate_trend.png'), dpi=150)
            plt.close()
        
        # 2. VNI Usage Distribution
        vni_df = pd.read_sql_query('''
        SELECT vni_type, COUNT(*) as usage_count
        FROM interactions
        GROUP BY vni_type
        ORDER BY usage_count DESC
        ''', conn)
        
        if not vni_df.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(vni_df['vni_type'], vni_df['usage_count'], color=sns.color_palette("husl", len(vni_df)))
            plt.title('VNI Usage Distribution')
            plt.xlabel('VNI Type')
            plt.ylabel('Number of Interactions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'vni_usage.png'), dpi=150)
            plt.close()
        
        # 3. Confidence Distribution
        conf_df = pd.read_sql_query('''
        SELECT confidence FROM interactions WHERE confidence > 0
        ''', conn)
        
        if not conf_df.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(conf_df['confidence'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Confidence Score Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'confidence_distribution.png'), dpi=150)
            plt.close()
        
        # 4. Response Time Analysis
        time_df = pd.read_sql_query('''
        SELECT vni_type, AVG(processing_time_ms) as avg_time
        FROM interactions
        WHERE processing_time_ms > 0
        GROUP BY vni_type
        ''', conn)
        
        if not time_df.empty:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(time_df['vni_type'], time_df['avg_time'], color='lightcoral')
            plt.title('Average Response Time by VNI Type')
            plt.xlabel('VNI Type')
            plt.ylabel('Average Response Time (ms)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.0f} ms', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'response_times.png'), dpi=150)
            plt.close()
        
        conn.close()
        
        logger.info(f"📊 Created visualizations in {output_path}")
        return {
            'visualizations_created': [
                'success_rate_trend.png',
                'vni_usage.png', 
                'confidence_distribution.png',
                'response_times.png'
            ],
            'output_path': output_path
        }
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to maintain database size"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        # Delete old interactions
        cursor.execute('DELETE FROM interactions WHERE timestamp < ?', (cutoff_date,))
        interactions_deleted = cursor.rowcount
        
        # Delete old system metrics (keep only 30 days of hourly data)
        metrics_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute('DELETE FROM system_metrics WHERE hour_start < ?', (metrics_cutoff,))
        metrics_deleted = cursor.rowcount
        
        # Vacuum database to reclaim space
        cursor.execute('VACUUM')
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Cleaned up {interactions_deleted} old interactions and "
                   f"{metrics_deleted} old metrics (keeping {days_to_keep} days)")
        
        return {
            'interactions_deleted': interactions_deleted,
            'metrics_deleted': metrics_deleted,
            'cutoff_date': cutoff_date
        }

# ============================================================================
# Global instances
# ============================================================================

# Keep original LearningAnalytics instance
analytics = LearningAnalytics()

# ADDED: Create SynapticVisualizer factory function for imports
def create_synaptic_visualizer(autonomy_engine=None):
    """Factory function to create SynapticVisualizer"""
    return SynapticVisualizer(autonomy_engine)

# ADDED: For backward compatibility
SynapticVisualizer = SynapticVisualizer
