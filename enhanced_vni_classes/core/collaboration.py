# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# enhanced_vni_classes/core/collaboration.py
"""Collaboration request system for Virtual Networked Individuals (VNIs)"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, ClassVar, List
from datetime import datetime
from enum import Enum
import uuid

class CollaborationType(Enum):
    """Types of collaboration between VNIs"""
    KNOWLEDGE_SHARE = "knowledge_share"
    TASK_DECOMPOSITION = "task_decomposition"
    CONFLICT_RESOLUTION = "conflict_resolution"
    LEARNING_TRANSFER = "learning_transfer"
    CONSENSUS_BUILDING = "consensus_building"
    
    @classmethod
    def from_string(cls, value: str) -> 'CollaborationType':
        """Safely convert string to CollaborationType"""
        try:
            return cls(value)
        except ValueError:
            # Fallback to default or raise with better error message
            raise ValueError(
                f"Invalid collaboration type: {value}. "
                f"Valid options are: {[t.value for t in cls]}"
            )

class CollaborationStatus(Enum):
    """Status of collaboration requests"""
    PENDING = "pending"
    ACCEPTED = "accepted" 
    REJECTED = "rejected"
    COMPLETED = "completed"
    EXPIRED = "expired"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    
    @classmethod
    def from_string(cls, value: str) -> 'CollaborationStatus':
        """Safely convert string to CollaborationStatus"""
        try:
            return cls(value)
        except ValueError:
            # Return PENDING as default for unknown status
            return cls.PENDING

@dataclass
class CollaborationRequest:
    """Request for collaboration between VNIs
    Attributes:
        request_id: Unique identifier for the request
        source_id: ID of the VNI making the request
        target_id: ID of the VNI receiving the request
        query: The collaboration query/question
        collaboration_type: Type of collaboration requested
        timestamp: When the request was created
        priority: Request priority (1=lowest, 5=highest)
        metadata: Additional request metadata
        response: Response from the target VNI (if any)
        status: Current status of the request
        expires_at: Optional expiration time for the request"""
    
    # Class constants
    MIN_PRIORITY: ClassVar[int] = 1
    MAX_PRIORITY: ClassVar[int] = 5
    DEFAULT_PRIORITY: ClassVar[int] = 1
    
    source_id: str
    target_id: str
    query: str
    collaboration_type: CollaborationType
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = DEFAULT_PRIORITY
    metadata: Dict[str, Any] = field(default_factory=dict)
    response: Optional['CollaborationResponse'] = None
    status: CollaborationStatus = field(default=CollaborationStatus.PENDING)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and normalize fields after initialization"""
        # Normalize collaboration_type if it's a string
        if isinstance(self.collaboration_type, str):
            self.collaboration_type = CollaborationType.from_string(self.collaboration_type)
        
        # Normalize status if it's a string
        if isinstance(self.status, str):
            self.status = CollaborationStatus.from_string(self.status)
        
        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}
        
        # Validate priority range
        if not (self.MIN_PRIORITY <= self.priority <= self.MAX_PRIORITY):
            raise ValueError(
                f"Priority must be between {self.MIN_PRIORITY} and {self.MAX_PRIORITY}"
            )
        
        # Validate source and target are not the same
        if self.source_id == self.target_id:
            raise ValueError("Source and target VNIs cannot be the same")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "request_id": self.request_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "query": self.query,
            "collaboration_type": self.collaboration_type.value,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "metadata": self.metadata,
            "status": self.status.value
        }
        
        # Add response if it exists
        if self.response:
            result["response"] = self.response.to_dict()
        
        # Add expires_at if it exists
        if self.expires_at:
            result["expires_at"] = self.expires_at.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollaborationRequest':
        """Create a CollaborationRequest from a dictionary"""
        
        # Handle optional fields with defaults
        request_id = data.get("request_id", str(uuid.uuid4()))
        priority = data.get("priority", cls.DEFAULT_PRIORITY)
        metadata = data.get("metadata", {})
        status = data.get("status", CollaborationStatus.PENDING.value)
        
        # Parse timestamp
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.now()
        
        # Parse expires_at if present
        expires_at = None
        if "expires_at" in data and data["expires_at"]:
            expires_at = datetime.fromisoformat(data["expires_at"])
        
        # Parse response if present
        response = None
        if "response" in data and data["response"]:
            response = CollaborationResponse.from_dict(data["response"])
        
        return cls(
            request_id=request_id,
            source_id=data["source_id"],
            target_id=data["target_id"],
            query=data["query"],
            collaboration_type=CollaborationType.from_string(data["collaboration_type"]),
            timestamp=timestamp,
            priority=priority,
            metadata=metadata,
            response=response,
            status=CollaborationStatus.from_string(status),
            expires_at=expires_at
        )
    
    def is_expired(self) -> bool:
        """Check if the request has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def accept(self) -> None:
        """Mark the request as accepted"""
        if self.is_expired():
            raise ValueError("Cannot accept an expired request")
        if self.status == CollaborationStatus.REJECTED:
            raise ValueError("Cannot accept a rejected request")
        self.status = CollaborationStatus.ACCEPTED
    
    def reject(self, reason: Optional[str] = None) -> None:
        """Mark the request as rejected"""
        if self.is_expired():
            raise ValueError("Cannot reject an expired request")
        self.status = CollaborationStatus.REJECTED
        if reason:
            self.metadata["rejection_reason"] = reason
    
    def complete(self, response_data: Dict[str, Any]) -> 'CollaborationResponse':
        """Mark the request as completed with a response"""
        if self.status != CollaborationStatus.ACCEPTED:
            raise ValueError("Request must be accepted before completion")
        
        # Create response object
        self.response = CollaborationResponse(
            request_id=self.request_id,
            responder_id=self.target_id,
            response_data=response_data,
            confidence_score=response_data.get("confidence_score", 1.0)
        )
        
        self.status = CollaborationStatus.COMPLETED
        return self.response
    
    def cancel(self) -> None:
        """Cancel the request"""
        self.status = CollaborationStatus.CANCELLED
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the request"""
        return {
            "request_id": self.request_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "collaboration_type": self.collaboration_type.value,
            "priority": self.priority,
            "status": self.status.value,
            "is_expired": self.is_expired(),
            "has_response": self.response is not None
        }

@dataclass
class CollaborationResponse:
    """Response to a collaboration request
    Attributes:
        request_id: ID of the request being responded to
        responder_id: ID of the VNI responding
        response_data: The actual response content
        timestamp: When the response was created
        confidence_score: Confidence in the response (0.0 to 1.0)
        metadata: Additional response metadata
        alternatives: Alternative responses (if any)"""
    
    request_id: str
    responder_id: str
    response_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate fields after initialization"""
        if self.confidence_score < 0 or self.confidence_score > 1:
            raise ValueError("confidence_score must be between 0 and 1")
        
        if self.metadata is None:
            self.metadata = {}
        
        if self.alternatives is None:
            self.alternatives = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "responder_id": self.responder_id,
            "response_data": self.response_data,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
            "alternatives": self.alternatives
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollaborationResponse':
        """Create from dictionary"""
        return cls(
            request_id=data["request_id"],
            responder_id=data["responder_id"],
            response_data=data["response_data"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            confidence_score=data.get("confidence_score", 1.0),
            metadata=data.get("metadata", {}),
            alternatives=data.get("alternatives", [])
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the response"""
        return {
            "request_id": self.request_id,
            "responder_id": self.responder_id,
            "confidence_score": self.confidence_score,
            "response_keys": list(self.response_data.keys()),
            "has_alternatives": len(self.alternatives) > 0
        }
