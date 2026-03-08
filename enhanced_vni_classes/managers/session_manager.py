# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# enhanced_vni_classes/managers/session_manager.py
"""
Session management for VNI interactions
"""
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionMessage:
    """Message in a session"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class Session:
    """A user session with VNIs"""
    
    def __init__(self, 
                 session_id: str,
                 user_id: str,
                 initial_vni_id: str = None,
                 ttl_hours: int = 24):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(hours=ttl_hours)
        self.active_vni_id = initial_vni_id
        self.vni_history: List[str] = []
        self.messages: List[SessionMessage] = []
        self.context: Dict[str, Any] = {}
        
        if initial_vni_id:
            self.vni_history.append(initial_vni_id)
        
        logger.info(f"Created session {session_id} for user {user_id}")
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> SessionMessage:
        """Add message to session"""
        message = SessionMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        return message
    
    def switch_vni(self, vni_id: str) -> bool:
        """Switch active VNI"""
        self.active_vni_id = vni_id
        if vni_id not in self.vni_history:
            self.vni_history.append(vni_id)
        
        # Add context switch message
        self.add_message(
            "system",
            f"Switched to VNI: {vni_id}",
            {"action": "vni_switch", "vni_id": vni_id}
        )
        
        logger.info(f"Session {self.session_id} switched to VNI {vni_id}")
        return True
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        recent_messages = self.messages[-limit:] if limit else self.messages
        return [msg.to_dict() for msg in recent_messages]
    
    def get_context(self) -> Dict[str, Any]:
        """Get session context"""
        return {
            **self.context,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "active_vni_id": self.active_vni_id,
            "vni_history": self.vni_history,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }
    
    def update_context(self, key: str, value: Any):
        """Update session context"""
        self.context[key] = value
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.now() > self.expires_at
    
    def extend(self, hours: int = 1):
        """Extend session expiry"""
        self.expires_at += timedelta(hours=hours)
        logger.info(f"Extended session {self.session_id} by {hours} hours")


class SessionManager:
    """Manages user sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
    
    def create_session(self, 
                      user_id: str,
                      initial_vni_id: str = None,
                      ttl_hours: int = 24) -> Session:
        """Create a new session"""
        session_id = str(uuid.uuid4())[:8]
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            initial_vni_id=initial_vni_id,
            ttl_hours=ttl_hours
        )
        
        self.sessions[session_id] = session
        
        # Track user's sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        logger.info(f"Created new session {session_id} for user {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session and session.is_expired():
            self.delete_session(session_id)
            return None
        return session
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user"""
        session_ids = self.user_sessions.get(user_id, [])
        sessions = []
        
        for session_id in session_ids[:]:
            session = self.get_session(session_id)
            if session:
                sessions.append(session)
            else:
                # Remove expired session from user's list
                session_ids.remove(session_id)
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Remove from user's session list
            user_id = session.user_id
            if user_id in self.user_sessions and session_id in self.user_sessions[user_id]:
                self.user_sessions[user_id].remove(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
            
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        
        return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired sessions"""
        expired_count = 0
        session_ids = list(self.sessions.keys())
        
        for session_id in session_ids:
            session = self.sessions[session_id]
            if session.is_expired():
                self.delete_session(session_id)
                expired_count += 1
        
        if expired_count:
            logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        self.cleanup_expired()
        
        return {
            "total_sessions": len(self.sessions),
            "total_users": len(self.user_sessions),
            "active_sessions_by_user": {
                user_id: len(sessions) 
                for user_id, sessions in self.user_sessions.items()
            }
        } 
