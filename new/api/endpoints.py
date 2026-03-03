"""
FastAPI Endpoints - All API routes
"""
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

from enhanced_neural_mesh import EnhancedNeuralMeshCore
from ..utils.helpers import verify_admin_token, get_vni_description
from ..utils.analytics import LearningAnalytics, SynapticVisualizer
from ..models.vni_loader import model_manager

router = APIRouter(prefix="/api", tags=["api"])

# Global orchestrator reference
_ORCHESTRATOR = None

def set_global_orchestrator(orchestrator):
    """Set the global orchestrator (called from app.py)"""
    global _ORCHESTRATOR
    _ORCHESTRATOR = orchestrator

def get_orchestrator() -> EnhancedNeuralMeshCore:
    """Get orchestrator from global reference"""
    if _ORCHESTRATOR is None:
        # Try to get from app state as fallback
        try:
            from fastapi import Request
            import inspect
            # This is a hack - in production, ensure set_global_orchestrator is called
            raise RuntimeError("Orchestrator not initialized. Check app setup.")
        except:
            raise RuntimeError("Orchestrator not available")
    return _ORCHESTRATOR

@router.get("/")
async def read_root():
    """Root endpoint with API information"""
    return {
        "message": "BabyBIONN API Server",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/api/chat (POST)",
            "analytics": "/api/analytics",
            "health": "/health",
            "chat_interface": "/chat"
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    orchestrator = get_orchestrator()
    
    # Safe attribute checking for EnhancedNeuralMeshCore
    initialized = True  # If we can call get_orchestrator(), it's initialized
    
    # Get VNI count
    vni_count = 0
    if hasattr(orchestrator, 'vni_manager') and hasattr(orchestrator.vni_manager, 'vni_instances'):
        vni_count = len(orchestrator.vni_manager.vni_instances)
    
    # Get interaction count
    interactions = 0
    if hasattr(orchestrator, 'task_history'):
        interactions = len(orchestrator.task_history)
    
    return {
        "status": "healthy",
        "initialized": initialized,
        "vni_instances": vni_count,
        "total_interactions": interactions
    }

@router.post("/chat")
async def chat_endpoint(request: Dict[str, Any]):
    """Enhanced chat endpoint"""
    try:
        orchestrator = get_orchestrator()
        message = request.get("message", "")
        session_id = request.get("session_id", "default")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        response = await orchestrator.process_query(
            message, 
            {"type": "chat", "session_id": session_id}, 
            session_id
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(feedback_data: Dict[str, Any]):
    """Feedback endpoint for learning"""
    try:
        orchestrator = get_orchestrator()
        await orchestrator.learn_from_feedback(feedback_data)
        return {"status": "success", "message": "Feedback processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_analytics():
    """Get learning analytics"""
    try:
        orchestrator = get_orchestrator()
        # Placeholder - integrate with analytics module
        return {
            "vni_count": len(orchestrator.vni_instances),
            "connections": len(orchestrator.synaptic_connections),
            "conversations": len(orchestrator.conversation_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-status")
async def get_model_status():
    """Get model loading status"""
    try:
        return {
            "status": "success",
            "model_status": model_manager.get_model_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generation-status")
async def get_generation_status():
    """Get text generation capabilities status"""
    try:
        orchestrator = get_orchestrator()
        status = orchestrator.get_generation_status()
        return {
            "status": "success",
            "generation_enabled": orchestrator.generation_enabled,
            "vni_generation_status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/verify/vni-fix")
async def verify_vni_fix():
    """Verify VNI classifier fix and routing"""
    try:
        orchestrator = get_orchestrator()
        
        test_phrase = "I need help on medical and health advise"
        
        # Test medical VNI
        medical_vni = orchestrator.vni_instances.get('medical_0')
        if not medical_vni:
            return {"error": "Medical VNI not found"}
        
        # Test should_handle
        should_handle_result = False
        if hasattr(medical_vni, 'should_handle'):
            try:
                should_handle_result = medical_vni.should_handle(test_phrase)
            except Exception as e:
                should_handle_result = f"Error: {str(e)}"
        
        return {
            "status": "checking",
            "test_phrase": test_phrase,
            "should_handle": should_handle_result,
            "vni_info": {
                "vni_id": "medical_0",
                "type": getattr(medical_vni, 'vni_type', 'unknown'),
                "has_classifier": hasattr(medical_vni, 'classifier'),
                "generation_enabled": getattr(medical_vni, 'generation_enabled', False)
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

# Dashboard endpoints
@router.get("/dashboard/metrics")
async def get_dashboard_metrics(admin_token: str = Depends(verify_admin_token)):
    """Get dashboard metrics (admin only)"""
    orchestrator = get_orchestrator()
    
    active_vnis = len([vni for vni in orchestrator.vni_instances.values() 
                      if getattr(vni, 'is_active', False)])
    
    return {
        "active_vnis": active_vnis,
        "total_vnis": len(orchestrator.vni_instances),
        "synaptic_connections": len(orchestrator.synaptic_connections),
        "conversation_history": len(orchestrator.conversation_history),
        "autonomy_level": orchestrator.autonomy_level
    }

# Autonomous endpoints
@router.post("/autonomy/set-level")
async def set_autonomy_level(request: Dict[str, Any]):
    """Set autonomy level"""
    try:
        orchestrator = get_orchestrator()
        level = request.get("level", 0.5)
        await orchestrator.set_autonomy_level(float(level))
        return {
            "status": "success",
            "autonomy_level": orchestrator.autonomy_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/autonomy/metrics")
async def get_autonomy_metrics():
    """Get autonomy metrics"""
    try:
        orchestrator = get_orchestrator()
        metrics = await orchestrator.get_autonomy_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat interface
@router.get("/chat-interface", response_class=HTMLResponse)
async def serve_chat_interface():
    """Serve chatbot interface"""
    chatbot_path = os.path.join(os.path.dirname(__file__), "../static/chatbot.html")
    if os.path.exists(chatbot_path):
        return FileResponse(chatbot_path)
    return HTMLResponse(content="<h1>BabyBIONN Chat Interface</h1><p>Interface file not found</p>") 
