# /neuron/shared/synaptic_visualization.py
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from typing import Dict, Any, List 

from neuron.shared.constants import COLOR_MAP, FIGURE_SIZE, DPI, ANIMATION_FPS
from neuron.shared.types import SynapticConnection

class NeuralPathway:
    """Minimal NeuralPathway class for visualization"""
    def __init__(self, source: str, target: str, strength: float = 0.5):
        self.source = source
        self.target = target
        self.strength = strength
        self.activation_count = 0

class SynapticVisualizer:
    """Visualize synaptic connections between VNI instances"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.pos = None
        self.connection_history = []
    
    def update_connections(self, pathways: Dict[str, NeuralPathway]):
        """Update graph with current pathways"""
        self.graph.clear()
        
        # Add nodes and edges
        for pathway_id, pathway in pathways.items():
            self.graph.add_node(pathway.source, type=pathway.source.split('_')[0])
            self.graph.add_node(pathway.target, type=pathway.target.split('_')[0])
            self.graph.add_edge(
                pathway.source, 
                pathway.target,
                weight=pathway.strength,
                activation_count=pathway.activation_count
            )
        
        # Store connection snapshot for history
        self.connection_history.append({
            'timestamp': datetime.now(),
            'connections': len(pathways),
            'average_strength': sum(p.strength for p in pathways.values()) / len(pathways) if pathways else 0
        })
    
    def create_static_visualization(self, filename="synaptic_network.png"):
        """Create a static visualization of the synaptic network"""
        plt.figure(figsize=(12, 8))
        
        if not self.graph.nodes:
            plt.text(0.5, 0.5, "No connections yet!\nStart chatting to build synaptic pathways.", 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        # Choose layout
        self.pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Node colors by type
        node_colors = []
        for node in self.graph.nodes():
            vni_type = self.graph.nodes[node]['type']
            if vni_type == 'medical':
                node_colors.append('#FF6B6B')  # Red
            elif vni_type == 'legal':
                node_colors.append('#4ECDC4')  # Teal
            elif vni_type == 'technical':
                node_colors.append('#45B7D1')  # Blue
            else:
                node_colors.append('#96CEB4')  # Green
        
        # Edge widths by strength
        edge_weights = [self.graph[u][v]['weight'] * 3 for u, v in self.graph.edges()]
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors, 
                              node_size=800, alpha=0.9)
        nx.draw_networkx_edges(self.graph, self.pos, width=edge_weights, 
                              alpha=0.6, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=8, font_weight='bold')
        
        # Edge labels for weights
        edge_labels = {(u, v): f"{self.graph[u][v]['weight']:.2f}" 
                      for u, v in self.graph.edges()}
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("BabyBIONN Synaptic Network\nLine thickness = Connection Strength", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_animated_evolution(self, filename="synaptic_evolution.gif"):
        """Create animated GIF showing network evolution"""
        if len(self.connection_history) < 3:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def animate(frame):
            ax.clear()
            # Show connection growth over time
            times = [h['timestamp'] for h in self.connection_history[:frame+1]]
            connections = [h['connections'] for h in self.connection_history[:frame+1]]
            strengths = [h['average_strength'] for h in self.connection_history[:frame+1]]
            
            ax.plot(times, connections, 'b-', label='Number of Connections', linewidth=2)
            ax.set_ylabel('Connections', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax.twinx()
            ax2.plot(times, strengths, 'r-', label='Average Strength', linewidth=2)
            ax2.set_ylabel('Strength', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_title('Synaptic Network Evolution Over Time')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.connection_history), 
                                     interval=500, repeat=False)
        anim.save(filename, writer='pillow', fps=2)
        plt.close()

    # Standalone demonstration function
    def demonstrate_visualization():
        """Demonstrate the visualization system with sample data"""
        visualizer = SynapticVisualizer()
    
        # Create sample neural pathways
        sample_pathways = {
            'path1': NeuralPathway('VNI_medical_001', 'VNI_health_001', 0.8),
            'path2': NeuralPathway('VNI_legal_001', 'VNI_compliance_001', 0.6),
            'path3': NeuralPathway('VNI_technical_001', 'VNI_software_001', 0.9),
            'path4': NeuralPathway('VNI_medical_001', 'VNI_bio_001', 0.7),
            'path5': NeuralPathway('VNI_legal_001', 'VNI_ethics_001', 0.5),
        }
    
        # Update and create visualization
        visualizer.update_connections(sample_pathways)
        visualizer.create_static_visualization("sample_synaptic_network.png")
    
        print("Sample visualization created: 'sample_synaptic_network.png'")
    
        # Create evolution animation with multiple updates
        for i in range(5):
            # Simulate strengthening connections
            for pathway in sample_pathways.values():
                pathway.strength = min(pathway.strength + 0.1, 1.0)
                pathway.activation_count += 1
        
            visualizer.update_connections(sample_pathways)
    
        visualizer.create_animated_evolution("sample_synaptic_evolution.gif")
        print("Sample animation created: 'sample_synaptic_evolution.gif'")

    if __name__ == "__main__":
        demonstrate_visualization()
