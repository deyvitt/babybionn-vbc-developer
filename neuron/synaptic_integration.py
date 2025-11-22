# synaptic_integration.py
"""
Integration module for synaptic visualization with FastAPI
"""

import logging
from fastapi import HTTPException
from typing import Dict
from synaptic_visualization import SynapticVisualizer

# Set up logging
logger = logging.getLogger("synaptic_visualization")

# Global instances (these would be initialized in your main application)
visualizer = None
orchestrator = None

def initialize_visualization_system():
    """Initialize the visualization system"""
    global visualizer, orchestrator
    
    from synaptic_visualization import SynapticVisualizer
    visualizer = SynapticVisualizer()
    
    # Mock orchestrator for demonstration - replace with your actual orchestrator
    class MockOrchestrator:
        def __init__(self):
            from synaptic_visualization import NeuralPathway
            self.synaptic_connections = {
                'init_path': NeuralPathway('VNI_base_001', 'VNI_general_001', 0.5)
            }
    
    orchestrator = MockOrchestrator()
    
    logger.info("Synaptic visualization system initialized")

# FastAPI endpoint (this would be in your main.py)
def create_synaptic_visualization_endpoint(app):
    """Create the synaptic visualization endpoint"""
    
    @app.get("/api/synaptic-visualization")
    async def get_synaptic_visualization():
        """Generate and return synaptic visualization"""
        try:
            if visualizer is None or orchestrator is None:
                initialize_visualization_system()
            
            visualizer.update_connections(orchestrator.synaptic_connections)
            visualizer.create_static_visualization("static/synaptic_network.png")
            
            return {
                "status": "success", 
                "image_url": "/static/synaptic_network.png",
                "connection_count": len(orchestrator.synaptic_connections),
                "average_strength": sum(p.strength for p in orchestrator.synaptic_connections.values()) / len(orchestrator.synaptic_connections) if orchestrator.synaptic_connections else 0
            }
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Example integration with your existing system
def integrate_with_babybionn(babybionn_orchestrator):
    """Integrate visualization with your BabyBIONN orchestrator"""
    global visualizer, orchestrator
    
    visualizer = SynapticVisualizer()
    orchestrator = babybionn_orchestrator
    
    # Create initial visualization
    visualizer.update_connections(orchestrator.synaptic_connections)
    visualizer.create_static_visualization("static/initial_synaptic_network.png")
    
    return visualizer
 