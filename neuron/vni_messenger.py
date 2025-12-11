"""
Messaging system with thumbdrive persistence for message history
"""
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from vni_storage import StorageManager

logger = logging.getLogger("vni_messenger")

@dataclass
class VNIMessage:
    """Message structure for VNI communication"""
    sender_id: str
    receiver_id: str  # "broadcast", "topic:topic_name", or specific VNI ID
    message_type: str  # "activation", "data", "response", "query", "collaboration"
    content: Dict[str, Any]
    timestamp: float
    message_id: str
    priority: int = 1  # 1=low, 5=critical
    requires_response: bool = False
    response_to: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

class VNIMessenger:
    """
    Messenger with thumbdrive persistence for message history
    """
    
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        
        # In-memory message queues per VNI
        self.queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        # Topic subscriptions: topic -> [vni_ids]
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)
        # Message history for persistence
        self.message_history: List[VNIMessage] = []
        
        # Load existing messages from thumbdrive
        self._load_message_history()
        
        # Auto-save settings
        self.save_counter = 0
        self.save_threshold = 50  # Save every 50 messages
        
        logger.info("VNI Messenger initialized with thumbdrive persistence")
    
    def _load_message_history(self):
        """Load message history from thumbdrive"""
        # We don't load all messages into queues (that would replay old messages)
        # Just load for record-keeping
        all_vnis = ["baseVNI", "transVNI", "operAction_general", "operAction_medical", "operAction_legal", "aggregator"]
        
        for vni_id in all_vnis:
            messages = self.storage.load_messages(vni_id)
            # Convert to VNIMessage objects
            for msg_dict in messages:
                try:
                    message = VNIMessage.from_dict(msg_dict)
                    self.message_history.append(message)
                except Exception as e:
                    logger.error(f"Failed to load message: {e}")
        
        logger.info(f"Loaded {len(self.message_history)} historical messages")
    
    def _save_message_history(self, message: VNIMessage):
        """Save message to thumbdrive"""
        self.save_counter += 1
        
        # Save message to sender's history
        sender_messages = [message.to_dict()]
        self.storage.save_messages(message.sender_id, sender_messages)
        
        # Also save to receiver if not broadcast
        if message.receiver_id not in ["broadcast", f"topic:{message.receiver_id}"]:
            self.storage.save_messages(message.receiver_id, sender_messages)
        
        # Bulk save all history periodically
        if self.save_counter >= self.save_threshold:
            # Save aggregated history
            all_messages = [msg.to_dict() for msg in self.message_history[-1000:]]
            self.storage.save_messages("system_history", all_messages)
            self.save_counter = 0
    
    def register_vni(self, vni_id: str):
        """Register a VNI to receive messages"""
        if vni_id not in self.queues:
            self.queues[vni_id] = asyncio.Queue()
            logger.debug(f"Registered VNI: {vni_id}")
    
    def subscribe_to_topic(self, vni_id: str, topic: str):
        """Subscribe VNI to a topic"""
        if vni_id not in self.subscriptions[topic]:
            self.subscriptions[topic].append(vni_id)
            logger.debug(f"VNI {vni_id} subscribed to topic: {topic}")
    
    async def send_message(self, message: VNIMessage):
        """Send message to specific VNI or broadcast"""
        
        # Store in history
        self.message_history.append(message)
        if len(self.message_history) > 10000:  # Keep last 10k messages
            self.message_history.pop(0)
        
        # Save to thumbdrive
        self._save_message_history(message)
        
        logger.debug(f"Message {message.message_id}: {message.sender_id} -> {message.receiver_id} [{message.message_type}]")
        
        # Broadcast to all VNIs
        if message.receiver_id == "broadcast":
            for vni_id, queue in self.queues.items():
                if vni_id != message.sender_id:  # Don't send to self
                    await queue.put(message)
            return True
        
        # Send to specific VNI
        if message.receiver_id in self.queues:
            await self.queues[message.receiver_id].put(message)
            return True
        
        # Send to topic subscribers
        if message.receiver_id.startswith("topic:"):
            topic = message.receiver_id[6:]  # Remove "topic:" prefix
            for subscriber_id in self.subscriptions.get(topic, []):
                if subscriber_id != message.sender_id:
                    await self.queues[subscriber_id].put(message)
            return True
        
        logger.warning(f"No receiver found for message: {message.receiver_id}")
        return False
    
    async def receive_message(self, vni_id: str, timeout: float = 1.0) -> Optional[VNIMessage]:
        """Receive a message for a VNI (non-blocking)"""
        try:
            if vni_id in self.queues:
                return await asyncio.wait_for(self.queues[vni_id].get(), timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving message for {vni_id}: {e}")
        return None
    
    def get_queue_size(self, vni_id: str) -> int:
        """Get number of pending messages for a VNI"""
        if vni_id in self.queues:
            return self.queues[vni_id].qsize()
        return 0
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get messaging statistics"""
        storage_info = self.storage.get_storage_info()
        
        return {
            "active_vnis": len(self.queues),
            "total_messages_sent": len(self.message_history),
            "queue_sizes": {vni_id: q.qsize() for vni_id, q in self.queues.items()},
            "subscriptions": {topic: len(vnis) for topic, vnis in self.subscriptions.items()},
            "storage_info": storage_info
        }
    
    def save_all_messages(self):
        """Force save all messages to thumbdrive"""
        all_messages = [msg.to_dict() for msg in self.message_history]
        self.storage.save_messages("full_backup", all_messages)
        logger.info(f"Saved {len(all_messages)} messages to thumbdrive") 
