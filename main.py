# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors
#!/usr/bin/env python3
"""BabyBIONN Autonomous Main Entry Point - Enhanced Hybrid Architecture
Combines autonomous neural mesh features with clean software engineering patterns.
"""
import os
import sys
import json
import uuid
import secrets
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional, Any
from Babybionn_integration import BabyBIONNSystem
from fastapi import Request, HTTPException, Cookie, Response
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
# from llm_Gateway import LLMGateway, LLMConfig, LLMProvider

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_medical")
logger = logging.getLogger("BabyBIONN.Main")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "babybionn_admin_2024")  # Change default in production!
active_sessions = {}  # In‑memory session store (for demo; use Redis in production)
sys.path.insert(0, str(PROJECT_ROOT))

# Try multiple import strategies like main.py
try:
    from new.api.app import create_app
    from new.utils.logging import setup_logging
    from new.utils.orchestrator_config import Config
    from enhanced_neural_mesh import EnhancedNeuralMeshCore
    from enhanced_vni_classes.managers.vni_manager import VNIManager
    from neuron.aggregator import UnifiedAggregator, ResponseAggregator

    HAS_NEW_STRUCTURE = True
except ImportError as e:
    # Fallback for alternative structure like main.py
    try:
        from api.app import create_app
        from utils.logging import setup_logging
        from utils.orchestrator_config import Config
        HAS_NEW_STRUCTURE = False
    except ImportError:
        print(f"❌ Failed to import required modules: {e}", file=sys.stderr)
        print("📁 Ensure project structure includes enhanced_neural_mesh.py and enhanced_vni_classes.py", file=sys.stderr)
        sys.exit(1)

# ========== ADD NEW LEARNING IMPORTS ==========
try:
    from neuron.synaptic_visualization import SynapticVisualizer
    from neuron.synaptic_learning_engine import integrate_with_babybionn

    LEARNING_AVAILABLE = True
    logger.info("✅ Learning engine and visualization available")
except ImportError as e:
    LEARNING_AVAILABLE = False
    logger.warning(f"⚠️  Learning system not available: {e}")
    logger.warning("   Install: pip install networkx matplotlib")

# Import uvicorn after path setup
try:
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form, APIRouter

except ImportError:
    print("❌ uvicorn not installed. Run: pip install uvicorn", file=sys.stderr)
    sys.exit(1)

# ========== ADD DEBUG CODE HERE ==========
# DEBUG: Check VNIType enum values
try:
    from enhanced_vni_classes.core.capabilities import VNIType
    print(f"🔍 DEBUG: VNIType members: {dir(VNIType)}")
    print(f"🔍 DEBUG: VNIType enum values: {list(VNIType.__members__.keys())}")
except Exception as e:
    print(f"🔍 DEBUG: Could not import VNIType: {e}")

# ========== GLOBAL APP MANAGEMENT ==========
# Add this after imports, before any other code
_global_app = None

def get_global_app():
    """Get the global FastAPI app instance"""
    global _global_app
    return _global_app

def set_global_app(app_instance):
    """Set the global FastAPI app instance"""
    global _global_app
    _global_app = app_instance

# Models for clean API design (from main.py pattern)
# ========== ADD NEW MODELS ==========
class LearningStatus(BaseModel):
    enabled: bool
    spontaneous_activations: int
    correlation_clusters: int
    connection_history: int
    last_learning: Optional[str]

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    context: Optional[Dict[str, Any]] = None

class AdminLogin(BaseModel):
    password: str

class EnhancedChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    learning_insights: Optional[Dict[str, Any]] = None
    spontaneous_learning: Optional[List[Dict]] = None
    timestamp: str

class SystemStatus(BaseModel):
    system_id: str
    status: str
    vni_manager: Dict[str, Any]
    neural_mesh: Dict[str, Any]
    autonomous_features: Dict[str, bool]

# ========== HELPER FUNCTIONS ==========
def validate_environment() -> bool:
    """Enhanced environment validation combining both approaches"""
    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (current: {sys.version_info.major}.{sys.version_info.minor})")

    # ========== PART 1: Check core files (UPDATED FOR PACKAGE SUPPORT) ==========
    required_items = [
        {"name": "enhanced_neural_mesh", "type": "file"},  # Single file
        {"name": "enhanced_vni_classes", "type": "package_or_file"}  # Package OR file
    ]
    
    missing_items = []

    for item in required_items:
        item_name = item["name"]
        item_type = item["type"]
        found = False
        
        if item_type == "file":
            # Check for single file
            file_path = PROJECT_ROOT / f"{item_name}.py"
            possible_paths = [
                PROJECT_ROOT / f"{item_name}.py",               # Root
                PROJECT_ROOT / 'new' / f"{item_name}.py",       # /new/
                PROJECT_ROOT / 'babybionn' / f"{item_name}.py", # /babybionn/
                PROJECT_ROOT / 'core' / f"{item_name}.py",      # /core/
                PROJECT_ROOT / 'babybionn' / 'new' / f"{item_name}.py"
            ]
            
        elif item_type == "package_or_file":
            # Check for package directory with __init__.py OR single file
            possible_paths = [
                # Package paths
                PROJECT_ROOT / item_name / "__init__.py",                # Package in root
                PROJECT_ROOT / 'new' / item_name / "__init__.py",        # Package in /new/
                PROJECT_ROOT / 'babybionn' / item_name / "__init__.py",  # Package in /babybionn/
                # Single file paths (backward compatibility)
                PROJECT_ROOT / f"{item_name}.py",               # Single file in root
                PROJECT_ROOT / 'new' / f"{item_name}.py",       # Single file in /new/
                PROJECT_ROOT / 'babybionn' / f"{item_name}.py", # Single file in /babybionn/
                PROJECT_ROOT / 'core' / f"{item_name}.py"       # Single file in /core/
            ]
        
        # Check all possible paths
        for path in possible_paths:
            if path.exists():
                found = True
                logger.debug(f"✅ Found {item_name} ({item_type}) at: {path}")
                break
        
        if not found:
            missing_items.append(f"{item_name} ({item_type})")

    if missing_items:
        issues.append(f"Missing required items: {', '.join(missing_items)}")
        issues.append("  Search paths tried for enhanced_vni_classes:")
        # Show actual paths checked (FIXED BUG HERE)
        check_paths = [
            PROJECT_ROOT / "enhanced_vni_classes.py",
            PROJECT_ROOT / "enhanced_vni_classes" / "__init__.py",
            PROJECT_ROOT / "new" / "enhanced_vni_classes.py",
            PROJECT_ROOT / "new" / "enhanced_vni_classes" / "__init__.py",
            PROJECT_ROOT / "babybionn" / "enhanced_vni_classes.py",
            PROJECT_ROOT / "babybionn" / "enhanced_vni_classes" / "__init__.py",
            PROJECT_ROOT / "core" / "enhanced_vni_classes.py"
        ]
        for path in check_paths:
            issues.append(f"    • {path}")

    # ========== PART 2: Check optional learning files (KEEP AS-IS) ==========
    optional_files = ['synaptic_learning_engine.py', 'synaptic_visualization.py']
    learning_files_found = []
    learning_files_missing = []
    
    for file in optional_files:
        file_path = PROJECT_ROOT / "neuron" / file
        
        if file_path.exists():
            learning_files_found.append(file)
            logger.info(f"✅ Optional learning file found: {file} at {file_path}")
        else:
            learning_files_missing.append(file)
            logger.warning(f"⚠️  Optional file not found: {file}")

    if learning_files_missing:
        logger.warning(f"⚠️  Learning capabilities will be limited")
        logger.warning(f"   Missing files: {', '.join(learning_files_missing)}")
    # ========== PART 3: Check critical directories (KEEP AS-IS) ==========

    critical_dirs = ['api', 'utils']
    found_any = False

    for dir_name in critical_dirs:
        possible_paths = [
            PROJECT_ROOT / dir_name,
            PROJECT_ROOT / 'new' / dir_name,
            PROJECT_ROOT / 'babybionn' / dir_name,
            PROJECT_ROOT / 'babybionn' / 'new' / dir_name
        ]

        if any(path.exists() for path in possible_paths):
            found_any = True
            break

    if not found_any:
        issues.append("Missing required API/utility directories")

    # ========== PART 4: Report results ==========
    if issues:
        print("❌ Environment validation failed:", file=sys.stderr)
        for issue in issues:
            print(f"  • {issue}", file=sys.stderr)
        return False

    print("✅ Environment validation passed")
    
    # Log learning availability
    if learning_files_found:
        print(f"✅ Learning system available: {len(learning_files_found)}/{len(optional_files)} files found")
    else:
        print(f"⚠️  Learning system not available (optional)")
    
    return True

def check_html_file():
    """Check if HTML file exists and log its location"""
    html_file = PROJECT_ROOT / "bionn_demo_chatbot.html"

    print(f"🔍 Looking for HTML file at: {html_file}")
    print(f"   Exists: {html_file.exists()}")

    if not html_file.exists():
        # Try alternative locations
        possible_locations = [
            PROJECT_ROOT / "static" / "bionn_demo_chatbot.html",
            PROJECT_ROOT / "templates" / "bionn_demo_chatbot.html",
            PROJECT_ROOT / "new" / "static" / "bionn_demo_chatbot.html",
            PROJECT_ROOT / "babybionn" / "static" / "bionn_demo_chatbot.html"
        ]

        for location in possible_locations:
            print(f"🔍 Checking: {location}")
            print(f"   Exists: {location.exists()}")
            if location.exists():
                return location

    return html_file if html_file.exists() else None

# ========== ADD THE NEW HELPER FUNCTION HERE ==========
async def serve_html_file_anywhere(filename: str):
    """Universal HTML file server with multiple fallback locations"""
    possible_paths = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "static" / filename,
        PROJECT_ROOT / "templates" / filename,
        PROJECT_ROOT / "new" / "static" / filename,
        PROJECT_ROOT / "babybionn" / "static" / filename,
        Path("/app") / filename,  # Docker path
        # Common alternative names
        PROJECT_ROOT / "static" / "index.html",
        PROJECT_ROOT / "index.html",
        PROJECT_ROOT / "chat.html",
        PROJECT_ROOT / "static" / "chat.html"
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.debug(f"Found HTML file at: {path}")
            return FileResponse(
                path,
                media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
    
    logger.warning(f"HTML file not found: {filename}")
    return None

# ========== THEN THE create_main_app FUNCTION ==========
def create_main_app(mesh_core: EnhancedNeuralMeshCore, aggregator: UnifiedAggregator = None) -> FastAPI:
    """Create FastAPI application using EnhancedNeuralMeshCore directly"""
    # Get base app with dependency injection
    base_app = create_app(mesh_core) if HAS_NEW_STRUCTURE else FastAPI()
    # Initialize BabyBIONN state to None (will be set by main())
    base_app.state.babybionn = None     

    # Add CORS middleware (from main.py)
    base_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @base_app.middleware("http")
    async def allow_inline_scripts(request: Request, call_next):
        response = await call_next(request)
        # Allow inline scripts and styles
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' http://localhost:8002; "
            "img-src 'self' data: blob:;"
        )
        return response
    # Store aggregator in app state for access in endpoints
    base_app.state.aggregator = aggregator
    base_app.state.mesh_core = mesh_core

    # Helper function to get VNI counts by domain
    def get_vni_counts_by_domain():
        counts = {}
        for vni_id, vni in mesh_core.vni_manager.vni_instances.items():
            domain = getattr(vni, 'vni_type', 'unknown')
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    # Define routes with proper typing
    @base_app.get("/")
    async def serve_chat_interface():
        """Serve chat interface with multiple fallback strategies"""
        logger.info("Serving chat interface...")
        
        # Try main HTML file
        response = await serve_html_file_anywhere("bionn_demo_chatbot.html")
        if response:
            return response
        
        # Try alternative names
        response = await serve_html_file_anywhere("chatbot.html")
        if response:
            return response
            
        response = await serve_html_file_anywhere("index.html")
        if response:
            return response
        
        # Last resort: Create a basic interface
        logger.warning("Chat interface not found, serving basic version")
        return await get_basic_chat_interface()

    async def get_basic_chat_interface():
        """Return a basic chat interface as fallback"""
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BabyBIONN Chat</title>
            <style>
                body { font-family: Arial; padding: 20px; background: #f0f2f5; }
                .container { max-width: 800px; margin: 0 auto; }
                h1 { color: #333; text-align: center; }
                .chat-box { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
                .message { margin: 10px 0; padding: 10px 15px; border-radius: 15px; max-width: 70%; }
                .user-message { background: #007bff; color: white; margin-left: auto; }
                .bot-message { background: #e9ecef; color: #333; }
                .input-area { display: flex; gap: 10px; }
                input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 5px; }
                button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                .status { text-align: center; color: #666; font-size: 0.9em; margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 BabyBIONN Chat</h1>
                <div class="status">System is online • Enhanced Neural Mesh Active</div>
                <div class="chat-box">
                    <div class="messages" id="messages"></div>
                    <div class="input-area">
                        <input type="text" id="userInput" placeholder="Type your message..." autofocus>
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
            
            <script>
                const messagesDiv = document.getElementById('messages');
                const userInput = document.getElementById('userInput');
                
                function addMessage(text, isUser = false) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                    messageDiv.textContent = text;
                    messagesDiv.appendChild(messageDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }
                
                async function sendMessage() {
                    const text = userInput.value.trim();
                    if (!text) return;
                    
                    addMessage(text, true);
                    userInput.value = '';
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                query: text,
                                session_id: 'web_session_' + Date.now()
                            })
                        });
                        
                        const data = await response.json();
                        addMessage(data.response || "No response received");
                    } catch (error) {
                        addMessage("Error connecting to BabyBIONN API. Please try again.");
                        console.error('Chat error:', error);
                    }
                }
                
                userInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') sendMessage();
                });
                
                // Add welcome message
                setTimeout(() => {
                    addMessage("Hello! I'm BabyBIONN. How can I help you today?");
                }, 500);
            </script>
        </body>
        </html>
        """)

    # Update the /api/chat endpoint to use aggregator properly
    @base_app.post("/api/chat")
    async def api_chat(request: ChatRequest):
        """Main chat endpoint using UnifiedAggregator for intelligent response coordination"""    
        # Get aggregator and mesh_core from app state
        aggregator = request.app.state.aggregator
        mesh_core = request.app.state.mesh_core
        
        try:
            context = {
                "session_id": request.session_id or "default",
                "timestamp": datetime.now().isoformat(),
                "query_type": "chat"
            }
    
            # USE AGGREGATOR (this is what we've been missing!)
            if aggregator:
                # Check which method aggregator has
                if hasattr(aggregator, 'process_query_advanced'):
                    aggregated_response = await aggregator.process_query_advanced(
                        query=request.query,
                        session_id=request.session_id or "default",
                        context={**context, **(request.context or {})}
                    )
                elif hasattr(aggregator, 'aggregate_response'):
                    aggregated_response = await aggregator.aggregate_response(
                        query=request.query,
                        session_id=request.session_id or "default",
                        context={**context, **(request.context or {})}
                    )
                else:
                    # Fallback to forward method
                    router_results = {
                        'query': request.query,
                        'query_context': context
                    }
                    result = aggregator(router_results)
                    aggregated_response = {
                        'response': result.get('final_response', 'Hello from BabyBIONN!'),
                        'sources': result.get('aggregation_analysis', {}).get('vni_contributions', []),
                        'confidence': result.get('confidence_metrics', {}).get('overall_confidence', 0.5)
                    }
                
                return {
                    "response": aggregated_response.get("response", "Hello from BabyBIONN!"),
                    "session_id": request.session_id or "default",
                    "activated_vnis": aggregated_response.get("sources", ["general-vni"]),
                    "confidence": aggregated_response.get("confidence", 0.5),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # NO FALLBACK TO MESH_CORE - log error instead
                logger.error("❌ Aggregator not available in app state!")
                return {
                    "response": "System is initializing. Please try again in a moment.",
                    "session_id": request.session_id or "default",
                    "activated_vnis": [],
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            return {
                "response": "I encountered an error processing your request. Please try again.",
                "session_id": request.session_id or "default",
                "activated_vnis": [],
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    @base_app.get("/chat")
    async def serve_chat_page():
        """Alias for chat interface"""
        return await serve_chat_interface()

    # System status endpoint using EnhancedNeuralMeshCore directly
    @base_app.get("/system/status", response_model=SystemStatus)
    async def get_system_status():
        """Get comprehensive system status"""
        try:
            # Debug: Check mesh_core
            print(f"🔍 DEBUG /system/status: mesh_core type: {type(mesh_core)}")
            print(f"🔍 DEBUG /system/status: has vni_manager: {hasattr(mesh_core, 'vni_manager')}")
            
            if hasattr(mesh_core, 'vni_manager'):
                print(f"🔍 DEBUG /system/status: vni_instances count: {len(mesh_core.vni_manager.vni_instances)}")
            
            mesh_status = mesh_core.get_mesh_status()
            print(f"🔍 DEBUG /system/status: mesh_status: {mesh_status}")
                    
            task_stats = mesh_core.get_task_statistics() if hasattr(mesh_core, 'get_task_statistics') else {}
            print(f"🔍 DEBUG /system/status: task_stats: {task_stats}")

            vni_counts = get_vni_counts_by_domain()
            print(f"🔍 DEBUG /system/status: vni_counts: {vni_counts}")

            return SystemStatus(
                system_id=f"babybionn_{uuid.uuid4().hex[:8]}",
                status="operational",
                vni_manager={
                    "total_vnis": len(mesh_core.vni_manager.vni_instances),
                    "vni_by_domain": vni_counts,
                    "active_vnis": len([v for v in mesh_core.vni_manager.vni_instances.values()
                                      if hasattr(v, 'is_active') and v.is_active])
                },
                neural_mesh=mesh_status,
                autonomous_features={
                    "learning_enabled": True,
                    "neural_pathways": len(mesh_core.vni_manager.neural_pathways) if hasattr(mesh_core.vni_manager, 'neural_pathways') else 0,
                    "collaboration_patterns": len(mesh_core.collaboration_tracker.patterns) if hasattr(mesh_core, 'collaboration_tracker') else 0,
                    "total_tasks": task_stats.get('total_tasks', 0)
                }
            )
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    @base_app.get("/system/vnis")
    async def get_vni_status():
        """Get VNI status and types"""
        try:
            type_map = {}
            for vni_id, vni in mesh_core.vni_manager.vni_instances.items():
                vni_type = getattr(vni, 'vni_type', 'unknown')
                if vni_type not in type_map:
                    type_map[vni_type] = []

                type_map[vni_type].append({
                    'id': vni_id,
                    'generation_enabled': getattr(vni, 'generation_enabled', False),
                    'knowledge_size': len(vni.knowledge_base.get('concepts', {}))
                                    if hasattr(vni, 'knowledge_base') else 0,
                    'learned_responses': len(vni.learned_responses)
                                      if hasattr(vni, 'learned_responses') else 0,
                    'is_active': getattr(vni, 'is_active', True)
                })

            return type_map
        except Exception as e:
            logger.error(f"Error getting VNI types: {e}")
            return {"error": str(e)}

    # Note: Session history functionality moved to EnhancedNeuralMeshCore task history
    @base_app.get("/session/{session_id}/history")
    async def get_session_history(session_id: str, limit: int = 20):
        """Get conversation history for a session from task history"""
        try:
            if hasattr(mesh_core, 'get_recent_tasks'):
                recent_tasks = mesh_core.get_recent_tasks(limit=limit)
                session_tasks = [task for task in recent_tasks
                               if task.get('session_id') == session_id]

                return {
                    "session_id": session_id,
                    "history": session_tasks,
                    "total_tasks": len(session_tasks),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "session_id": session_id,
                    "history": [],
                    "message": "Task history not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @base_app.get("/mesh/visualization")
    async def get_mesh_visualization():
        """Get neural mesh visualization data with error handling"""
        try:
            mesh_status = mesh_core.get_mesh_status()

            # Extract visualization data safely
            nodes = []
            if hasattr(mesh_core, 'mesh_nodes'):
                for node_id, node in mesh_core.mesh_nodes.items():
                    nodes.append({
                        "id": node_id,
                        "type": getattr(node, 'node_type', 'unknown'),
                        "activation": getattr(node, 'current_activation', 0),
                        "connections": len(getattr(node, 'axons', [])) + len(getattr(node, 'dendrites', []))
                    })

            return {
                "nodes": nodes[:100],  # Limit for performance
                "total_patterns": mesh_status.get('synaptic_patterns', 0),
                "total_edges": mesh_status.get('total_synapses', 0),
                "active_nodes": mesh_status.get('active_nodes', 0)
            }
        except Exception as e:
            logger.error(f"Error getting mesh visualization: {e}")
            return {"error": "Could not generate visualization"}

    @base_app.post("/system/learn")
    async def trigger_learning(request: Request):
        """Trigger active learning from conversation"""
        try:
            data = await request.json()
            query = data.get("query", "")
            response = data.get("response", "")

            # Learning happens automatically in EnhancedNeuralMeshCore
            return {
                "status": "learning_active",
                "query": query[:100],
                "learning_targets": ["all_vnis"],
                "timestamp": datetime.now().isoformat(),
                "message": "EnhancedNeuralMeshCore continuously learns from interactions"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Health endpoint with dependencies
    @base_app.get("/health")
    async def health_check():
        """Enhanced health check"""
        try:
            mesh_status = mesh_core.get_mesh_status()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system_id": f"babybionn_{uuid.uuid4().hex[:8]}",
                "components": {
                    "vni_manager": len(mesh_core.vni_manager.vni_instances) > 0,
                    "neural_mesh": "total_nodes" in mesh_status,
                    "learning_active": True
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    @base_app.get("/debug-files")
    async def debug_files():
        """Debug endpoint to list files"""
        files = []
        for root, dirs, filenames in os.walk(PROJECT_ROOT):
            for filename in filenames:
                if filename.endswith('.html'):
                    files.append(os.path.join(root, filename))

        return {
            "project_root": str(PROJECT_ROOT),
            "html_files": files,
            "current_dir_files": os.listdir(PROJECT_ROOT)
        }

    # Serve static files if directory exists (from main.py)
    static_dirs = [
        PROJECT_ROOT / "static",
        PROJECT_ROOT / "new" / "static",
        PROJECT_ROOT / "babybionn" / "static"
    ]

    for static_dir in static_dirs:
        if static_dir.exists():
            base_app.mount("/static", StaticFiles(directory=static_dir), name="static")
            logger.info(f"📁 Mounted static files from: {static_dir}")
            break

    # ========== ADD MISSING API ENDPOINTS FOR FRONTEND ==========
    @base_app.get("/api/health")
    async def api_health():
        return {
            "status": "healthy",
            "service": "BabyBIONN API",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }

    @base_app.get("/api/config/llm-provider")
    async def get_llm_provider_config():
        try:
            # Get provider info from aggregator, not babybionn
            if aggregator and hasattr(aggregator, 'llm_gateway'):
                current_provider = getattr(aggregator.llm_gateway, 'current_provider', 'unknown')
                available_providers = list(aggregator.llm_gateway.clients.keys())
            else:
                # Fallback to env vars
                current_provider = os.getenv('LLM_PROVIDER', 'mock')
                available_providers = ['mock', 'deepseek', 'openai', 'ollama']
            
            return {
                "provider": current_provider,
                "available_providers": available_providers,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to get LLM provider config: {str(e)}")
    
    @base_app.post("/api/config/llm-provider")
    async def update_llm_provider_config(request: Request):
        """POST method to update LLM provider configuration"""
        try:
            data = await request.json()
            provider = data.get('provider', 'mock')
            
            # Update via aggregator
            if aggregator and hasattr(aggregator, 'llm_gateway'):
                available_providers = list(aggregator.llm_gateway.clients.keys())
                
                if provider not in available_providers:
                    raise HTTPException(400, f"Invalid provider. Available: {available_providers}")
                
                # Update the provider in the gateway
                if hasattr(aggregator.llm_gateway, 'set_primary_provider'):
                    aggregator.llm_gateway.set_primary_provider(provider)
                    base_app.state.llm_provider = provider
                
                logger.info(f"LLM provider changed to: {provider}")
                
                return {
                    "success": True,
                    "provider": provider,
                    "message": f"LLM provider changed to {provider}",
                    "active_clients": len(aggregator.llm_gateway.clients),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback - just store in app state
                base_app.state.llm_provider = provider
                return {
                    "success": True,
                    "provider": provider,
                    "message": f"LLM provider preference saved (gateway not available)",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to update LLM provider: {e}")
            raise HTTPException(500, f"Failed to update LLM provider: {str(e)}")
        
    @base_app.post("/api/chat-with-image")
    async def api_chat_with_image(
        message: str = Form(None),
        image: UploadFile = File(None),
        session_id: str = Form(...),
        admin_token: str = Form(None)
    ):
        try:
            response_text = f"Message: {message or 'No text'}"

            if image:
                contents = await image.read()
                response_text += f"\nImage: {image.filename} ({len(contents)} bytes)"

            return {
                "response": response_text,
                "session_id": session_id,
                "image_analysis": {
                    "filename": image.filename if image else None,
                    "size": len(contents) if image else 0
                },
                "activated_vnis": ["core-vni", "general-vni"]
            }
        except Exception as e:
            raise HTTPException(500, f"Image error: {str(e)}")

    @base_app.post("/api/admin/login")
    async def admin_login(login: AdminLogin, response: Response):
        """Authenticate admin and set session cookie."""
        if login.password == ADMIN_PASSWORD:
            session_token = secrets.token_urlsafe(32)
            active_sessions[session_token] = True  # In production, add expiry
            response.set_cookie(
                key="admin_session",
                value=session_token,
                httponly=True,
                max_age=3600,          # 1 hour
                secure=False,           # Set to True if using HTTPS
                samesite="lax"
            )
            return {"success": True, "message": "Logged in successfully"}
        raise HTTPException(status_code=401, detail="Invalid password")

    async def verify_admin(admin_session: str = Cookie(None)):
        """Dependency to protect admin routes."""
        if not admin_session or admin_session not in active_sessions:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return True

    @base_app.post("/api/admin/pretrain")
    async def api_pretrain(
        domain: str = Form(...),
        file: UploadFile = File(...),
        session_id: str = Form(...),
        authenticated: bool = Depends(verify_admin)   # <-- new dependency
    ):
        # Debug logging
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG: /api/admin/pretrain called")
        print(f"   domain: {domain}")
        print(f"   filename: {file.filename}")
        print(f"   content_type: {file.content_type}")
        print(f"   session_id: {session_id}")
        print(f"{'='*60}\n")
        
        try:
            # Read and parse the uploaded JSON
            print(f"📖 DEBUG: Reading file contents...")
            contents = await file.read()
            print(f"   File size: {len(contents)} bytes")
            
            # Show first 500 chars of content for debugging
            preview = contents[:500].decode('utf-8', errors='ignore')
            print(f"   File preview: {preview}...")
            
            print(f"🔄 DEBUG: Parsing JSON...")
            try:
                data = json.loads(contents)
                print(f"   ✅ JSON parsed successfully")
                print(f"   Top-level keys: {list(data.keys())}")
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parse error: {e}")
                print(f"   Error at line {e.lineno}, col {e.colno}")
                print(f"   Error text: {e.msg}")
                raise HTTPException(400, f"Invalid JSON format: {str(e)}")
    
            # Get the VNI manager
            print(f"🧠 DEBUG: Getting VNI manager...")
            vni_manager = mesh_core.vni_manager
            print(f"   Current VNIs: {list(vni_manager.vni_instances.keys())}")
    
            # For technical domain, use the existing VNI
            if domain == 'technical':
                vni_id = 'technical_vni_001'
                print(f"   Using existing technical VNI: {vni_id}")
                
                # Check if it exists
                if vni_id not in vni_manager.vni_instances:
                    print(f"   ⚠️ Technical VNI not found, creating new one")
                    vni_manager.create_vni(domain, vni_id, False)
            else:
                # For other domains, create a new pretrained VNI with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                vni_id = f"{domain}_pretrained_{timestamp}"
                print(f"   Creating VNI with ID: {vni_id}")
        
                # Create VNI using VNIManager
                try:
                    vni_manager.create_vni(domain, vni_id, False)
                    print(f"   ✅ VNI created successfully")
                except Exception as e:
                    print(f"   ❌ Failed to create VNI: {e}")
                    raise HTTPException(500, f"Failed to create VNI: {str(e)}")
        
            # Get the created VNI
            vni = vni_manager.vni_instances.get(vni_id)
            print(f"   Retrieved VNI: {vni}")
            
            if not vni:
                print(f"   ❌ VNI not found after creation!")
                raise HTTPException(500, f"Failed to create VNI for domain {domain}")
    
            # Check VNI capabilities
            print(f"   VNI type: {type(vni).__name__}")
            print(f"   Has knowledge_base: {hasattr(vni, 'knowledge_base')}")
            if hasattr(vni, 'knowledge_base'):
                print(f"   knowledge_base type: {type(vni.knowledge_base)}")
            
            print(f"   Has add_knowledge: {hasattr(vni, 'add_knowledge')}")
    
            # === Process the data ===
            concepts_added = 0
            frameworks_loaded = []
            
            # Check if this is a comprehensive framework
            if "babybionn_cognitive_framework" in data:
                print(f"✅ DEBUG: Detected comprehensive cognitive framework")
                framework = data["babybionn_cognitive_framework"]
                print(f"   Framework keys: {list(framework.keys())}")
                
                # Store the entire framework in VNI's knowledge base
                if hasattr(vni, 'knowledge_base'):
                    if isinstance(vni.knowledge_base, dict):
                        print(f"   ✅ Storing framework in knowledge_base")
                        if 'cognitive_framework' not in vni.knowledge_base:
                            vni.knowledge_base['cognitive_framework'] = {}
                        
                        # Store each section
                        for key, value in framework.items():
                            vni.knowledge_base['cognitive_framework'][key] = value
                            frameworks_loaded.append(key)
                            print(f"      - Stored section: {key}")
                
                # Extract training examples from ethical_moral_foundations
                if 'ethical_moral_foundations' in framework:
                    ethical = framework['ethical_moral_foundations']
                    print(f"   Found ethical_moral_foundations")
                    
                    if 'conversational_training_examples' in ethical:
                        conv_examples = ethical['conversational_training_examples']
                        print(f"      Found conversational_training_examples")
                        
                        # Extract greeting sequences
                        if 'basic_interaction_patterns' in conv_examples:
                            patterns = conv_examples['basic_interaction_patterns']
                            if 'greeting_sequences' in patterns:
                                for seq in patterns['greeting_sequences']:
                                    if 'user' in seq and 'babybionn' in seq:
                                        if hasattr(vni, 'add_knowledge'):
                                            vni.add_knowledge(seq['user'], seq['babybionn'])
                                            concepts_added += 1
                                            print(f"      ✅ Added greeting: '{seq['user'][:30]}...'")
                            
                            if 'symptom_inquiry_flows' in patterns:
                                for flow in patterns['symptom_inquiry_flows']:
                                    if 'user' in flow and 'babybionn' in flow:
                                        if hasattr(vni, 'add_knowledge'):
                                            vni.add_knowledge(flow['user'], flow['babybionn'])
                                            concepts_added += 1
                                            print(f"      ✅ Added symptom inquiry: '{flow['user'][:30]}...'")
                        
                        # Extract complex scenarios
                        if 'complex_scenario_dialogues' in conv_examples:
                            complex_scenarios = conv_examples['complex_scenario_dialogues']
                            if 'emotional_support_scenarios' in complex_scenarios:
                                for scenario in complex_scenarios['emotional_support_scenarios']:
                                    if 'user' in scenario and 'babybionn' in scenario:
                                        if hasattr(vni, 'add_knowledge'):
                                            vni.add_knowledge(scenario['user'], scenario['babybionn'])
                                            concepts_added += 1
                                            print(f"      ✅ Added emotional support: '{scenario['user'][:30]}...'")
                            
                            if 'safety_critical_dialogues' in complex_scenarios:
                                for scenario in complex_scenarios['safety_critical_dialogues']:
                                    if 'user' in scenario and 'babybionn' in scenario:
                                        if hasattr(vni, 'add_knowledge'):
                                            vni.add_knowledge(scenario['user'], scenario['babybionn'])
                                            concepts_added += 1
                                            print(f"      ✅ Added safety critical: '{scenario['user'][:30]}...'")
                
                # Extract practical ethical scenarios
                if 'practical_ethical_scenarios' in framework:
                    practical = framework['practical_ethical_scenarios']
                    print(f"   Found practical_ethical_scenarios: {list(practical.keys())}")
                    for key, scenario in practical.items():
                        if isinstance(scenario, dict):
                            if 'ethical_empathetic_response' in scenario:
                                situation = scenario.get('situation', '')
                                response = scenario.get('ethical_empathetic_response', '')
                                if situation and response:
                                    if hasattr(vni, 'add_knowledge'):
                                        vni.add_knowledge(situation, response)
                                        concepts_added += 1
                                        print(f"      ✅ Added ethical scenario: '{situation[:30]}...'")
                            
                            if 'extended_dialogue_example' in scenario:
                                dialogue_key = f"dialogue_{key}"
                                if hasattr(vni, 'knowledge_base') and isinstance(vni.knowledge_base, dict):
                                    if 'dialogues' not in vni.knowledge_base:
                                        vni.knowledge_base['dialogues'] = {}
                                    vni.knowledge_base['dialogues'][key] = scenario['extended_dialogue_example']
                                    print(f"      ✅ Added extended dialogue: {key}")
                
                # Extract cognitive foundations reasoning scenarios
                if 'cognitive_foundations' in framework:
                    cognitive = framework['cognitive_foundations']
                    print(f"   Found cognitive_foundations")
                    if 'practical_reasoning_scenarios' in cognitive:
                        reasoning = cognitive['practical_reasoning_scenarios']
                        print(f"      Found practical_reasoning_scenarios: {list(reasoning.keys())}")
                        for key, scenario in reasoning.items():
                            if 'user_input' in scenario:
                                if hasattr(vni, 'knowledge_base') and isinstance(vni.knowledge_base, dict):
                                    if 'reasoning_examples' not in vni.knowledge_base:
                                        vni.knowledge_base['reasoning_examples'] = {}
                                    vni.knowledge_base['reasoning_examples'][key] = {
                                        'input': scenario.get('user_input', ''),
                                        'reasoning': scenario.get('reasoning_process', []),
                                        'learning_points': scenario.get('learning_points', [])
                                    }
                                    concepts_added += 1
                                    print(f"      ✅ Added reasoning example: {key}")
            
            # Also handle simple training_data format if present
            elif "training_data" in data:
                print(f"✅ DEBUG: Detected simple training_data format")
                training_items = data.get("training_data", [])
                print(f"   Found {len(training_items)} training items")
                
                for i, item in enumerate(training_items):
                    input_text = item.get("input", "")
                    output_text = item.get("output", "")
                    
                    if input_text and output_text:
                        if hasattr(vni, 'add_knowledge'):
                            vni.add_knowledge(input_text, output_text)
                            concepts_added += 1
                            print(f"      ✅ Added item {i+1}: '{input_text[:30]}...'")
            
            else:
                print(f"⚠️ DEBUG: Unknown data format - no recognized structure")
                print(f"   Available top-level keys: {list(data.keys())}")
    
            print(f"\n{'='*60}")
            print(f"✅ DEBUG: Pretraining completed")
            print(f"   Concepts added: {concepts_added}")
            print(f"   Frameworks loaded: {frameworks_loaded}")
            print(f"{'='*60}\n")
    
            return {
                "success": True,
                "message": f"Pretraining data uploaded for {domain}",
                "vni_id": vni_id,
                "analytics": {
                    "domain": domain,
                    "concepts_added": concepts_added,
                    "frameworks_loaded": frameworks_loaded,
                    "file_size": len(contents),
                    "vni_created": True,
                    "vni_manager_total": len(vni_manager.vni_instances),
                    "timestamp": datetime.now().isoformat()
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"\n❌ DEBUG: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n")
            raise HTTPException(400, f"Pretraining failed: {str(e)}")
                
    @base_app.get("/api/admin/knowledge-status")
    async def api_knowledge_status(authenticated: bool = Depends(verify_admin)):

        try:
            # Get the VNI manager
            vni_manager = mesh_core.vni_manager
            domains = ["medical", "technical", "legal", "general", "core"]
            knowledge_bases = {}

            for domain in domains:
                # Find VNIs for this domain
                domain_vnis = []
                for vni_id, vni in vni_manager.vni_instances.items():
                    vni_type = getattr(vni, 'vni_type', '')
                    if vni_type == domain:
                        domain_vnis.append(vni)

                if domain_vnis:
                    # Use the first VNI for stats
                    vni = domain_vnis[0]
                    concepts = 0
                    patterns = 0

                    # Count concepts and patterns
                    if hasattr(vni, 'knowledge_base'):
                        kb = vni.knowledge_base
                        if isinstance(kb, dict):
                            concepts = len(kb.get('concepts', {}))
                            patterns = len(kb.get('patterns', {}))
                        elif hasattr(kb, '__len__'):
                            concepts = len(kb)

                    knowledge_bases[domain] = {
                        "concepts": concepts,
                        "patterns": patterns,
                        "vni_count": len(domain_vnis),
                        "vni_ids": [vni_id for vni_id, v in vni_manager.vni_instances.items()
                                   if getattr(v, 'vni_type', '') == domain],
                        "last_updated": datetime.now().timestamp(),
                        "status": "active"
                    }
                else:
                    knowledge_bases[domain] = {
                        "concepts": 0,
                        "patterns": 0,
                        "vni_count": 0,
                        "vni_ids": [],
                        "last_updated": 0,
                        "status": "not_initialized"
                    }

            return {
                "success": True,
                "knowledge_bases": knowledge_bases,
                "total_vnis": len(vni_manager.vni_instances),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Knowledge status error: {e}")
            raise HTTPException(500, f"Error getting knowledge status: {str(e)}")

    return base_app

def create_basic_chat_interface():
    """Create a basic chat interface if not present"""
    html_file = PROJECT_ROOT / "bionn_demo_chatbot.html"

    # Check if file already exists
    if html_file.exists():
        logger.info(f"✅ Chat interface already exists: {html_file}")
        return

    # Use the HTML from main_b.py but make it configurable
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BabyBIONN Autonomous System</title>
    <style>
        /* CSS from main_b.py - could be externalized */
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               color: white; }
        .container { max-width: 800px; margin: 0 auto;
                     background: rgba(255, 255, 255, 0.1);
                     backdrop-filter: blur(10px); border-radius: 20px;
                     padding: 30px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }
        /* ... rest of the CSS ... */
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 BabyBIONN Autonomous System</h1>
        <div class="status-indicator">
            <div class="status-dot"></div>
            <span>Enhanced Hybrid Architecture - System Active</span>
        </div>
        <!-- Chat interface content -->
    </div>
    <script>
        // JavaScript from main_b.py
        // ... rest of the JavaScript ...
    </script>
</body>
</html>"""

    try:
        with open(html_file, 'w') as f:
            f.write(html_content)
        logger.info(f"✅ Created basic chat interface: {html_file}")
    except Exception as e:
        logger.error(f"❌ Failed to create chat interface: {e}")

async def main():
    """Enhanced main entry point with proper error handling"""
    # ADD THIS DEBUGGING LINE AT THE VERY BEGINNING:
    print(f"🔍 DEBUG: Starting main(), get_global_app() = {get_global_app()}")    
    # ADD THIS IMMEDIATELY - before any other code:
    print("=" * 60)
    print("🔍 DEBUG: Starting BabyBIONN with HTML file check")
    print(f"Project root: {PROJECT_ROOT}")

    # Check HTML file immediately
    html_file = PROJECT_ROOT / "bionn_demo_chatbot.html"
    print(f"Looking for HTML at: {html_file}")
    print(f"HTML exists: {html_file.exists()}")

    if html_file.exists():
        print("✅ HTML file found!")
    else:
        print("❌ HTML file NOT FOUND!")
        print("Listing files in /app:")
        for file in os.listdir(PROJECT_ROOT):
            print(f"  - {file}")

    print("=" * 60)

    # Setup logging first (original code continues...)
    try:
        setup_logging()
        logger = logging.getLogger("BabyBIONN")
    except Exception as e:
        print(f"❌ Failed to setup logging: {e}", file=sys.stderr)
        return 1

    # Validate environment with enhanced checking
    if not validate_environment():
        return 1

    try:
        logger.info("=" * 60)
        logger.info("🚀 Starting Enhanced Hybrid BabyBIONN System")
        logger.info("✨ Features: Autonomous + Clean Architecture")
        logger.info(f"📁 Project root: {PROJECT_ROOT}")
        logger.info(f"🐍 Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        logger.info("=" * 60)

        # Initialize BabyBIONN System with LLM configs
        logger.info("🌐 Initializing BabyBIONN System...")
        
        # Load API keys from environment
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not deepseek_key:
            logger.warning("⚠️ DEEPSEEK_API_KEY not found in environment")
        
        llm_configs = {
            "deepseek": {
                "api_key": deepseek_key,
                "base_url": "https://api.deepseek.com/v1"
            },
            "mock": {
                "responses": {
                    "medical": "Mock medical response (DeepSeek not configured)",
                    "general": "Mock general response"
                }
            }
        }
        try:
            babybionn = BabyBIONNSystem(llm_configs=llm_configs)
            # Debug: Check if LLM Gateway initialized
            if hasattr(babybionn, 'llm_gateway') and babybionn.llm_gateway:
                logger.info(f"✅ LLM Gateway initialized with providers: {list(babybionn.llm_gateway.clients.keys())}")
            else:
                logger.error("❌ LLM Gateway missing or None!")
                # Log what attributes babybionn DOES have
                logger.info(f"🔍 BabyBIONN attributes: {dir(babybionn)}")
        except Exception as e:
            logger.error(f"❌ Exception during BabyBIONN initialization: {e}")
            import traceback
            traceback.print_exc()
            babybionn = None
            
        logger.info("✅ BabyBIONN System initialization attempt complete")

        # Initialize EnhancedNeuralMeshCore directly
        logger.info("🧠 Initializing Enhanced Neural Mesh Core...")
        vni_manager = VNIManager(enable_generation=True)
        # Create core VNIs for each domain
        vni_manager.create_vni('medical', 'medical_vni_001')
        vni_manager.create_vni('legal', 'legal_vni_001')
        vni_manager.create_vni('technical', 'technical_vni_001')
        vni_manager.create_vni('general', 'general_vni_001')
        logger.info(f"✅ Created {len(vni_manager.vni_instances)} core VNIs: {list(vni_manager.vni_instances.keys())}")
        
        # ==================== MEDICAL VNI DEBUG START ====================
        print("\n" + "="*70)
        print("🧪 MEDICAL VNI DEBUG - STARTING")
        print("="*70)
        
        # 1. Check initial VNI Manager state
        print("📊 STEP 1: Checking initial VNI Manager state...")
        print(f"   VNI Manager created: {type(vni_manager).__name__}")
        print(f"   Has vni_instances: {hasattr(vni_manager, 'vni_instances')}")
        if hasattr(vni_manager, 'vni_instances'):
            initial_count = len(vni_manager.vni_instances)
            print(f"   Initial VNI count: {initial_count}")
            if initial_count > 0:
                print("   Initial VNIs found:")
                for vni_id, vni in vni_manager.vni_instances.items():
                    print(f"   - {vni_id}: {type(vni).__name__}")
            else:
                print("   No VNIs found initially (expected)")
        
        # 2. Test direct MedicalVNI import
        print("\n🔬 STEP 2: Testing direct MedicalVNI import...")
        try:
            from enhanced_vni_classes.domains.medical import MedicalVNI
            print("   ✅ SUCCESS: MedicalVNI imported")
            
            # Test instantiation
            print("   Testing MedicalVNI instantiation...")
            medical = MedicalVNI(
                vni_id="medical_debug_001",
                name="Medical Debug",
                enable_biological_systems=True
            )
            print(f"   ✅ MedicalVNI instantiated: {medical.vni_id}")
            print(f"   - Domain: {getattr(medical, 'domain', 'unknown')}")
            print(f"   - Has process(): {hasattr(medical, 'process')}")
            
            # Quick test of process method
            print("   Testing process() method...")
            test_result = medical.process("test: headache")
            print(f"   ✅ process() executed, returned: {type(test_result).__name__}")
            if isinstance(test_result, dict):
                print(f"   - Keys: {list(test_result.keys())}")
                if 'response' in test_result:
                    preview = str(test_result['response'])[:60]
                    print(f"   - Response preview: {preview}...")
                
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Test VNIManager.create_vni for medical
        print("\n⚙️ STEP 3: Testing VNIManager.create_vni('medical')...")
        try:
            medical_via_manager = vni_manager.create_vni(
                domain='medical',
                instance_id='medical_test_001'
            )
            print(f"   ✅ SUCCESS: VNIManager.create_vni('medical') worked")
            print(f"   - Created: {type(medical_via_manager).__name__}")
            print(f"   - Instance ID: {getattr(medical_via_manager, 'instance_id', 'unknown')}")
            
            # Check what VNIs are now in manager
            current_count = len(vni_manager.vni_instances)
            print(f"   - Now has {current_count} VNI(s) in manager")
            
            if current_count > 0:
                print("   Current VNIs in manager:")
                for vni_id, vni in vni_manager.vni_instances.items():
                    vni_type = getattr(vni, 'vni_type', 'unknown')
                    print(f"   - {vni_id}: {type(vni).__name__} (type: {vni_type})")
                    
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*70)
        print("🧪 MEDICAL VNI DEBUG - COMPLETE")
        print("="*70 + "\n")

        # ==================== MEDICAL VNI DEBUG END ====================
        mesh_core = EnhancedNeuralMeshCore(vni_manager)
        logger.info("🤝 Initializing Enhanced Aggregator...")

        from neuron.shared.synaptic_config import SynapticConfig
        # Create config with Hebbian learning ENABLED
        aggregator_config = SynapticConfig(
            aggregator_id="unified_aggregator",
            # === HEBBIAN LEARNING (ENABLED) ===
            enable_hebbian_learning=True,
            enable_biological_routing=False,  # Disable biological, use pure Hebbian
            learning_rate=0.1,
            decay_rate=0.01,
            strengthening_threshold=0.7,
            weakening_threshold=0.4,
            pruning_threshold=0.1,
            
            # === AUTO SPAWNING ===
            enable_auto_spawning=True,
            max_clusters=10,
            
            # === SESSION MANAGEMENT ===
            session_timeout_hours=2,
            history_size=1000,
            session_history_size=50
        )
        
        # Pass config to aggregator
        aggregator = ResponseAggregator(
            vni_manager=vni_manager,
            config=aggregator_config  # ← CRITICAL!
        )

        # Create FastAPI app with both mesh_core AND aggregator
        logger.info("🌐 Creating Enhanced FastAPI Application...")
        app = create_main_app(mesh_core, aggregator)
        # Store the babybionn instance in app state
        app.state.babybionn = babybionn
        logger.info("✅ FastAPI app created with EnhancedNeuralMeshCore and UnifiedAggregator")
        # logger.info(f"✅ BabyBIONN stored in app state with {len(babybionn.llm_gateway.clients)} providers")

        # ========== SET GLOBAL APP ==========
        set_global_app(app)
        logger.info("✅ Global app reference updated")
        print(f"🔍 DEBUG: set_global_app() called, get_global_app() = {get_global_app()}")

        config = uvicorn.Config(
            app,  # ← Use the app from create_main_app, NOT the module-level app
            host=Config.HOST,
            port=Config.PORT,
            reload=Config.RELOAD,
            log_level="info",
            access_log=True
        )
        # Log system details
        mesh_status = mesh_core.get_mesh_status()
        logger.info(f"✅ Neural Mesh: {mesh_status.get('total_nodes', 0)} nodes")
        logger.info(f"✅ Synapses: {mesh_status.get('total_synapses', 0)}")
        logger.info(f"✅ Core VNIs: {len(vni_manager.vni_instances)}")

        # Test the system
        test_response = await mesh_core.process_query(
            query="Hello, BabyBIONN! Are you ready?",
            session_id="test_session"
        )
        logger.info(f"✅ System test successful. Confidence: {test_response.get('confidence', 0):.2f}")

        # Display server info (clean format from main.py)
        logger.info("=" * 60)
        logger.info("🌍 Server Information:")
        logger.info(f"  • Host: {Config.HOST}")
        logger.info(f"  • Port: {Config.PORT}")
        logger.info(f"  • Reload: {Config.RELOAD}")
        logger.info("=" * 60)
        logger.info("📡 Available Endpoints:")
        logger.info(f"  • Chat Interface:    http://{Config.HOST}:{Config.PORT}/")
        logger.info(f"  • API Docs:          http://{Config.HOST}:{Config.PORT}/docs")
        logger.info(f"  • System Status:     http://{Config.HOST}:{Config.PORT}/system/status")
        logger.info(f"  • Health Check:      http://{Config.HOST}:{Config.PORT}/health")
        logger.info("=" * 60)

        # Check for HTML interface with multiple fallbacks
        html_file = PROJECT_ROOT / "bionn_demo_chatbot.html"
        if not html_file.exists():
            logger.warning(f"⚠️  Chat interface HTML not found, creating basic version")
            create_basic_chat_interface()

        # Configure and start server
        config = uvicorn.Config(
            app,
            host=Config.HOST,
            port=Config.PORT,
            reload=Config.RELOAD,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)
        logger.info("🎯 Server starting...")
        await server.serve()

    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal (Ctrl+C)")
        logger.info("🛑 Shutting down gracefully...")
        return 0

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ Failed to start Enhanced BabyBIONN: {e}")
        logger.exception("Full traceback:")
        logger.error("=" * 60)
        return 1

    return 0

# ========== MAIN ENTRY POINT ==========
if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
