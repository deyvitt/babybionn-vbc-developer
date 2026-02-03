# enhanced_vni_classes/core/registry.py
"""
VNI registry for managing VNI instances
"""
from typing import Any, Dict, List, Optional
from ..core.base_vni import EnhancedBaseVNI
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VNIRegistry:
    """
    Registry for tracking all VNI instances
    """
    
    def __init__(self):
        self._vnies: Dict[str, EnhancedBaseVNI] = {}
        self._domain_index: Dict[str, List[str]] = {}
    
    def register(self, vni: EnhancedBaseVNI) -> bool:
        """
        Register a VNI instance
        """
        if vni.instance_id in self._vnies:
            logger.warning(f"VNI {vni.instance_id} already registered")
            return False
        
        self._vnies[vni.instance_id] = vni
        
        # Add to domain index
        domain = vni.domain
        if domain not in self._domain_index:
            self._domain_index[domain] = []
        self._domain_index[domain].append(vni.instance_id)
        
        logger.info(f"Registered VNI: {vni.instance_id} ({domain})")
        return True
    
    def unregister(self, vni_id: str) -> bool:
        """
        Unregister a VNI instance
        """
        if vni_id not in self._vnies:
            return False
        
        vni = self._vnies[vni_id]
        
        # Remove from domain index
        domain = vni.domain
        if domain in self._domain_index and vni_id in self._domain_index[domain]:
            self._domain_index[domain].remove(vni_id)
            if not self._domain_index[domain]:
                del self._domain_index[domain]
        
        del self._vnies[vni_id]
        logger.info(f"Unregistered VNI: {vni_id}")
        return True
    
    def get(self, vni_id: str) -> Optional[EnhancedBaseVNI]:
        """Get VNI by ID"""
        return self._vnies.get(vni_id)
    
    def get_by_domain(self, domain: str) -> List[EnhancedBaseVNI]:
        """Get all VNIs in a domain"""
        vni_ids = self._domain_index.get(domain, [])
        return [self._vnies[vni_id] for vni_id in vni_ids if vni_id in self._vnies]
    
    def get_all(self) -> List[EnhancedBaseVNI]:
        """Get all VNIs"""
        return list(self._vnies.values())
    
    def get_domains(self) -> List[str]:
        """Get all registered domains"""
        return list(self._domain_index.keys())
    
    def find_similar(self, 
                    query: str,
                    threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find VNIs similar to query
        """
        results = []
        for vni_id, vni in self._vnies.items():
            # Simple domain matching - can be enhanced with embedding similarity
            if query.lower() in vni.domain.lower():
                results.append({
                    "vni_id": vni_id,
                    "domain": vni.domain,
                    "confidence": 0.7,
                    "status": vni.get_status()
                })
        
        return sorted(results, key=lambda x: x["confidence"], reverse=True) 
