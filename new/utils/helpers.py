"""
Helper Functions
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import HTTPException, status

def verify_admin_token(admin_token: str) -> str:
    """Verify admin token for protected endpoints"""
    if admin_token != "babybionn_admin_2024":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )
    return admin_token

def get_vni_description(domain: str) -> str:
    """Get human-readable description for VNI domains"""
    descriptions = {
        "medical": "Healthcare and medical knowledge",
        "legal": "Legal knowledge and compliance", 
        "general": "General knowledge and conversation",
        "technical": "Programming and technical expertise",
        "core": "System operations and coordination"
    }
    return descriptions.get(domain, f"{domain} knowledge")

def format_timestamp(dt: datetime) -> str:
    """Format datetime for API responses"""
    return dt.isoformat()

def calculate_confidence_scores(responses: List[Dict]) -> Dict[str, float]:
    """Calculate confidence scores from multiple responses"""
    if not responses:
        return {"average": 0.0, "max": 0.0, "min": 0.0}
    
    confidences = [r.get('confidence', 0.0) for r in responses]
    
    return {
        "average": sum(confidences) / len(confidences),
        "max": max(confidences),
        "min": min(confidences),
        "count": len(confidences)
    }

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text"""
    words = text.lower().split()
    # Filter out short words and common stop words
    stop_words = {'the', 'and', 'but', 'for', 'with', 'this', 'that', 'these', 'those'}
    return [word for word in words if len(word) >= min_length and word not in stop_words]

def create_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "success": False
    }

def get_system_health(orchestrator) -> Dict[str, Any]:
    """Get system health status"""
    if not hasattr(orchestrator, 'vni_instances'):
        return {
            "status": "initializing",
            "active_vnis": 0,
            "total_vnis": 0,
            "health_score": 0
        }
    
    active_vnis = len([vni for vni in orchestrator.vni_instances.values() 
                      if getattr(vni, 'is_active', False)])
    total_vnis = len(orchestrator.vni_instances)
    
    health_score = 0
    if total_vnis > 0:
        health_score = min(100, (active_vnis / total_vnis) * 100 + 50)
    
    status_text = "healthy" if health_score > 80 else "degraded" if health_score > 50 else "poor"
    
    return {
        "status": status_text,
        "active_vnis": active_vnis,
        "total_vnis": total_vnis,
        "health_score": round(health_score, 1)
    } 
