"""
Adapter classes for orchestrator integration - COMPLETE FIX
"""
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class Stimulus:
    """Data class representing a stimulus for VNIs to process"""
    content: str
    stimulus_type: str = "query"
    metadata: Dict[str, Any] = None
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stimulus to dictionary"""
        return {
            'content': self.content,
            'stimulus_type': self.stimulus_type,
            'metadata': self.metadata,
            'source': self.source
        }

class OrchestratorToVNIManagerAdapter:
    """Adapter to make orchestrator compatible with VNI learning system"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        # Define expected VNI method names in order of preference
        self.vni_methods_to_try = ['process_query', 'generate_response', 'respond_to_query', 'process']
    
    def get_all_vnis(self) -> List[Any]:
        """Get all VNI instances"""
        if hasattr(self.orchestrator, 'vni_instances'):
            vni_instances = self.orchestrator.vni_instances
            if isinstance(vni_instances, dict):
                return list(vni_instances.values())
            elif isinstance(vni_instances, list):
                return vni_instances
            else:
                logger.warning(f"vni_instances is not dict or list: {type(vni_instances)}")
                return []
        logger.warning("Orchestrator has no vni_instances attribute")
        return []
    
    def get_vni_by_id(self, vni_id: str) -> Optional[Any]:
        """Get VNI by ID"""
        if hasattr(self.orchestrator, 'vni_instances'):
            vni_instances = self.orchestrator.vni_instances
            if isinstance(vni_instances, dict):
                return vni_instances.get(vni_id)
            elif isinstance(vni_instances, list):
                # Try to find by various attributes
                for vni in vni_instances:
                    if hasattr(vni, 'instance_id') and getattr(vni, 'instance_id', None) == vni_id:
                        return vni
                    if hasattr(vni, 'id') and getattr(vni, 'id', None) == vni_id:
                        return vni
                    if hasattr(vni, 'name') and getattr(vni, 'name', None) == vni_id:
                        return vni
                    if hasattr(vni, '__class__'):
                        class_name = vni.__class__.__name__
                        if vni_id.lower() in class_name.lower():
                            return vni
        logger.warning(f"VNI {vni_id} not found in orchestrator")
        return None
    
    def get_all_vnis_with_ids(self) -> Dict[str, Any]:
        """Get all VNIs as a dictionary with their IDs as keys"""
        if not hasattr(self.orchestrator, 'vni_instances'):
            logger.warning("Orchestrator has no vni_instances attribute")
            return {}
        
        vni_instances = self.orchestrator.vni_instances
        
        if isinstance(vni_instances, dict):
            # Already a dict with IDs as keys
            return vni_instances.copy()
        
        # Convert list to dict
        vni_dict = {}
        for i, vni in enumerate(vni_instances):
            vni_id = None
            
            # Try to get ID from various attributes
            if hasattr(vni, 'instance_id'):
                vni_id = getattr(vni, 'instance_id')
            elif hasattr(vni, 'id'):
                vni_id = getattr(vni, 'id')
            elif hasattr(vni, 'name'):
                vni_id = getattr(vni, 'name')
            elif hasattr(vni, '__class__'):
                class_name = vni.__class__.__name__
                if 'EnhancedMedicalVNI' in class_name:
                    vni_id = f"medical_{i}"
                elif 'EnhancedLegalVNI' in class_name:
                    vni_id = f"legal_{i}"
                elif 'EnhancedGeneralVNI' in class_name:
                    vni_id = f"general_{i}"
                else:
                    vni_id = f"{class_name.lower().replace('vni', '').replace('enhanced', '').strip('_')}_{i}"
            else:
                vni_id = f"vni_{i}"
            
            vni_dict[vni_id] = vni
        
        return vni_dict
    
    def spawn_vni(self, vni_type: str, instance_id: str) -> Any:
        """Spawn a new VNI instance"""
        try:
            # Check if orchestrator has vni_classes
            if not hasattr(self.orchestrator, 'vni_classes'):
                logger.error("Orchestrator has no vni_classes attribute")
                return None
            
            vni_classes = self.orchestrator.vni_classes
            if not isinstance(vni_classes, dict):
                logger.error(f"vni_classes is not a dictionary: {type(vni_classes)}")
                return None
            
            # Map VNI types to class names
            vni_class_map = {
                "medical": 'EnhancedMedicalVNI',
                "legal": 'EnhancedLegalVNI', 
                "general": 'EnhancedGeneralVNI',
                "enhanced_medical": 'EnhancedMedicalVNI',
                "enhanced_legal": 'EnhancedLegalVNI',
                "enhanced_general": 'EnhancedGeneralVNI'
            }
            
            class_name = vni_class_map.get(vni_type.lower(), vni_type)
            
            if class_name in vni_classes:
                vni_class = vni_classes[class_name]
                instance = vni_class(instance_id)
                
                # Ensure orchestrator has vni_instances dict
                if not hasattr(self.orchestrator, 'vni_instances'):
                    self.orchestrator.vni_instances = {}
                elif not isinstance(self.orchestrator.vni_instances, dict):
                    self.orchestrator.vni_instances = {}
                
                self.orchestrator.vni_instances[instance_id] = instance
                logger.info(f"➕ Spawned {instance_id} of type {vni_type} via adapter")
                return instance
            else:
                available = list(vni_classes.keys())
                logger.warning(f"VNI class '{class_name}' not found. Available: {available}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to spawn {instance_id} ({vni_type}): {e}", exc_info=True)
            return None
    
    def update_vni_connection_strength(self, source_id: str, target_id: str, new_strength: float):
        """Update synaptic connection strength"""
        if not hasattr(self.orchestrator, 'synaptic_connections'):
            logger.warning("Orchestrator has no synaptic_connections attribute")
            return
        
        connection_id = f"{source_id}→{target_id}"
        if connection_id in self.orchestrator.synaptic_connections:
            self.orchestrator.synaptic_connections[connection_id].strength = new_strength
            logger.info(f"🔄 Updated connection {connection_id} to strength {new_strength}")
        else:
            logger.warning(f"Connection {connection_id} not found")
    
    def log_interaction(self, vni_id: str, query: str, response: str, confidence: float):
        """Log VNI interaction for learning"""
        logger.info(f"📝 Interaction logged for {vni_id}: {confidence:.2f} confidence")
        # Could store this in a learning log if needed
        if hasattr(self.orchestrator, 'interaction_log'):
            log_entry = {
                'vni_id': vni_id,
                'query': query,
                'response': response,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            self.orchestrator.interaction_log.append(log_entry)

    def process_stimulus(self, *args, **kwargs):
        """Flexible process_stimulus that can handle different call signatures"""
        if len(args) == 2:  # Called as process_stimulus(vni_id, stimulus)
            vni_id, stimulus = args[0], args[1]
            logger.warning(f"⚠️ process_stimulus called with vni_id={vni_id}, using single VNI mode")
            # Call the new single VNI method
            return self._process_stimulus_single_vni(vni_id, stimulus)
        elif len(args) == 1:  # Called as process_stimulus(stimulus)
            stimulus = args[0]
            # Call the original batch processing method
            return self._process_stimulus_batch(stimulus)
        else:
            raise ValueError(f"Invalid arguments to process_stimulus: {args}")

    def _process_stimulus_batch(self, stimulus: Union[Stimulus, Dict[str, Any], str]) -> Dict[str, Any]:
        """Process stimulus for learning orchestrator - COMPLETE FIX"""
        # Handle different stimulus input types
        if isinstance(stimulus, str):
            stimulus_obj = Stimulus(content=stimulus, stimulus_type="query")
        elif isinstance(stimulus, dict):
            stimulus_obj = Stimulus(
                content=stimulus.get('content', ''),
                stimulus_type=stimulus.get('stimulus_type', 'query'),
                metadata=stimulus.get('metadata', {})
            )
        elif isinstance(stimulus, Stimulus):
            stimulus_obj = stimulus
        else:
            logger.error(f"Unsupported stimulus type: {type(stimulus)}")
            return {
                'error': f"Unsupported stimulus type: {type(stimulus)}",
                'vni_responses': {},
                'stimulus_id': '',
                'timestamp': datetime.now().isoformat(),
                'processed_by': 'orchestrator_adapter'
            }
        
        logger.info(f"🎯 Processing stimulus: {stimulus_obj.stimulus_type}")
        
        # Get all VNIs with their IDs
        vni_dict = self.get_all_vnis_with_ids()
        logger.info(f"🔍 Found {len(vni_dict)} VNIs: {list(vni_dict.keys())}")
        
        if not vni_dict:
            logger.warning("⚠️ No VNI instances available to process stimulus")
            return {
                'vni_responses': {},
                'stimulus_id': str(abs(hash(stimulus_obj.content)))[:8],
                'timestamp': datetime.now().isoformat(),
                'processed_by': 'orchestrator_adapter',
                'original_stimulus': stimulus_obj.to_dict()
            }
        
        # Process with all available VNIs
        vni_responses = {}
        for vni_id, vni_instance in vni_dict.items():
            logger.info(f"🔍 Processing with VNI: {vni_id}")
            logger.debug(f"🔍 VNI type: {type(vni_instance)}")
            logger.debug(f"🔍 VNI attributes: {[attr for attr in dir(vni_instance) if not attr.startswith('_')][:10]}")
            
            # Try different method names
            response = None
            method_used = None
            
            for method_name in self.vni_methods_to_try:
                if hasattr(vni_instance, method_name):
                    try:
                        method = getattr(vni_instance, method_name)
                        logger.debug(f"🔍 Trying method {method_name} on {vni_id}")
                        
                        # Call with appropriate parameters
                        if method_name in ['process_query', 'generate_response']:
                            response = method(stimulus_obj.content, {})
                        else:
                            response = method(stimulus_obj.content)
                        
                        method_used = method_name
                        logger.debug(f"✅ Method {method_name} succeeded for {vni_id}")
                        break
                    except Exception as e:
                        logger.debug(f"❌ Method {method_name} failed for {vni_id}: {e}")
                        continue
            
            if response is not None:
                # Extract response type from VNI ID or class
                response_type = "unknown"
                if hasattr(vni_instance, '__class__'):
                    class_name = vni_instance.__class__.__name__.lower()
                    if 'medical' in class_name:
                        response_type = 'medical'
                    elif 'legal' in class_name:
                        response_type = 'legal'
                    elif 'general' in class_name:
                        response_type = 'general'
                    else:
                        # Extract type from class name
                        response_type = class_name.replace('vni', '').replace('enhanced', '').strip('_')
                
                vni_responses[vni_id] = {
                    'content': str(response),
                    'confidence': 0.7,  # Default confidence
                    'response_type': response_type,
                    'method_used': method_used,
                    'stimulus_type': stimulus_obj.stimulus_type,
                    'vni_class': vni_instance.__class__.__name__
                }
                logger.info(f"✅ Got response from {vni_id} using {method_used}")
                
                # Log the interaction
                self.log_interaction(vni_id, stimulus_obj.content, str(response), 0.7)
            else:
                logger.warning(f"⚠️ No response from {vni_id} after trying all methods")
                logger.debug(f"🔍 Available methods on {vni_id}: {[m for m in dir(vni_instance) if not m.startswith('_')]}")
        
        # Create stimulus ID from content hash
        stimulus_id = str(abs(hash(stimulus_obj.content)))[:8]
        
        logger.info(f"📊 Returning {len(vni_responses)} VNI responses")
        return {
            'vni_responses': vni_responses,
            'stimulus_id': stimulus_id,
            'timestamp': datetime.now().isoformat(),
            'processed_by': 'orchestrator_adapter',
            'original_stimulus': stimulus_obj.to_dict()
        }

    def _process_stimulus_single_vni(self, vni_id: str, stimulus: Union[Stimulus, Dict[str, Any], str]) -> Dict[str, Any]:
        """Process stimulus with a specific VNI only"""
        # Convert stimulus input to Stimulus object
        if isinstance(stimulus, str):
            stimulus_obj = Stimulus(content=stimulus, stimulus_type="query")
        elif isinstance(stimulus, dict):
            stimulus_obj = Stimulus(
                content=stimulus.get('content', ''),
                stimulus_type=stimulus.get('stimulus_type', 'query'),
                metadata=stimulus.get('metadata', {})
            )
        elif isinstance(stimulus, Stimulus):
            stimulus_obj = stimulus
        else:
            logger.error(f"Unsupported stimulus type: {type(stimulus)}")
            return {
                'error': f"Unsupported stimulus type: {type(stimulus)}",
                'vni_responses': {},
                'stimulus_id': '',
                'timestamp': datetime.now().isoformat(),
                'processed_by': 'orchestrator_adapter'
            }
    
        # Get the specific VNI
        vni_instance = self.get_vni_by_id(vni_id)
        if not vni_instance:
            logger.warning(f"VNI {vni_id} not found")
            return {
                'vni_responses': {},
                'stimulus_id': str(abs(hash(stimulus_obj.content)))[:8],
                'timestamp': datetime.now().isoformat(),
                'processed_by': 'orchestrator_adapter',
                'error': f"VNI {vni_id} not found"
            }
    
        logger.info(f"🔍 Processing with specific VNI: {vni_id}")
    
        # Try different method names
        response = None
        method_used = None
    
        for method_name in self.vni_methods_to_try:
            if hasattr(vni_instance, method_name):
                try:
                    method = getattr(vni_instance, method_name)
                    # Call with appropriate parameters
                    if method_name in ['process_query', 'generate_response']:
                        response = method(stimulus_obj.content, {})
                    else:
                        response = method(stimulus_obj.content)
                    method_used = method_name
                    break
                except Exception as e:
                    logger.debug(f"Method {method_name} failed for {vni_id}: {e}")
                    continue
    
        vni_responses = {}
        if response is not None:
            # Extract response type from VNI ID or class
            response_type = "unknown"
            if hasattr(vni_instance, '__class__'):
                class_name = vni_instance.__class__.__name__.lower()
                if 'medical' in class_name:
                    response_type = 'medical'
                elif 'legal' in class_name:
                    response_type = 'legal'
                elif 'general' in class_name:
                    response_type = 'general'
                else:
                    response_type = class_name.replace('vni', '').replace('enhanced', '').strip('_')
        
            vni_responses[vni_id] = {
                'content': str(response),
                'confidence': 0.7,
                'response_type': response_type,
                'method_used': method_used,
                'stimulus_type': stimulus_obj.stimulus_type
            }
            logger.info(f"✅ Got response from {vni_id} using {method_used}")
        else:
            logger.warning(f"⚠️ No response from {vni_id}")
    
        stimulus_id = str(abs(hash(stimulus_obj.content)))[:8]
    
        return {
            'vni_responses': vni_responses,
            'stimulus_id': stimulus_id,
            'timestamp': datetime.now().isoformat(),
            'processed_by': 'orchestrator_adapter',
            'original_stimulus': stimulus_obj.to_dict()
        }

    def process_stimulus_with_learning(self, stimulus: Union[Stimulus, Dict, str]) -> Dict[str, Any]:
        """Process stimulus with learning updates"""
        # First, process normally
        result = self.process_stimulus(stimulus)
    
        # Add learning logic here
        self._apply_learning_updates(result)
    
        return result

    def _apply_learning_updates(self, processing_result: Dict[str, Any]):
        """Apply learning updates based on processing results"""
        try:
            vni_responses = processing_result.get('vni_responses', {})
            stimulus_data = processing_result.get('original_stimulus', {})
        
            if not vni_responses:
                logger.debug("No responses to learn from")
                return
        
            # 1. UPDATE CONNECTION STRENGTHS BASED ON COLLABORATION
            self._update_connections_from_interactions(vni_responses, stimulus_data)
        
            # 2. ADJUST VNI CONFIDENCE SCORES
            self._adjust_vni_confidence(vni_responses, stimulus_data)
        
            # 3. LEARN STIMULUS PATTERNS
            self._learn_stimulus_patterns(stimulus_data, vni_responses)
        
            # 4. UPDATE SYNAPTIC WEIGHTS IF APPLICABLE
            self._update_synaptic_weights(processing_result)
        
        except Exception as e:
            logger.error(f"Error in learning updates: {e}", exc_info=True)

    def _update_connections_from_interactions(self, vni_responses: Dict, stimulus_data: Dict):
        """Update connection strengths when VNIs collaborate"""
        vni_ids = list(vni_responses.keys())
    
        # For every pair of VNIs that responded
        for i in range(len(vni_ids)):
            for j in range(len(vni_ids)):
                if i == j:
                    continue
                
                source_id = vni_ids[i]
                target_id = vni_ids[j]
            
                # Check if this collaboration was useful
                source_response = vni_responses[source_id]['content']
                target_response = vni_responses[target_id]['content']
            
                # Simple collaboration detection - if responses reference each other
                collaboration_score = self._calculate_collaboration_score(
                    source_response, target_response
                )
            
                if collaboration_score > 0.3:  # Threshold for meaningful collaboration
                    # Strengthen the connection
                    connection_id = f"{source_id}→{target_id}"
                    if hasattr(self.orchestrator, 'synaptic_connections'):
                        if connection_id in self.orchestrator.synaptic_connections:
                            current_strength = self.orchestrator.synaptic_connections[connection_id].strength
                            new_strength = min(1.0, current_strength + 0.05)  # Increment
                            self.update_vni_connection_strength(source_id, target_id, new_strength)
                        else:
                            # Create new connection if it doesn't exist
                            self._create_synaptic_connection(source_id, target_id, 0.5)
                        
                    logger.debug(f"Updated connection {source_id}→{target_id} due to collaboration")

    def _calculate_collaboration_score(self, response1: str, response2: str) -> float:
        """Calculate how much two responses collaborate"""
        # Simple implementation - check for references
        score = 0.0
    
        # Convert to lowercase for comparison
        r1_lower = response1.lower()
        r2_lower = response2.lower()
    
        # Check if responses complement each other
        unique_words1 = set(r1_lower.split())
        unique_words2 = set(r2_lower.split())
    
        # Shared concepts
        shared_words = unique_words1.intersection(unique_words2)
        if len(shared_words) > 3:  # They talk about similar things
            score += 0.2
    
        # Check if one response references concepts from the other
        if len(r1_lower) > 50 and len(r2_lower) > 50:
            # Simple overlap check
            overlap = self._calculate_text_overlap(r1_lower, r2_lower)
            score += overlap * 0.3
    
        return min(score, 1.0)

    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap between two strings"""
        words1 = set(text1.split())
        words2 = set(text2.split())
    
        if not words1 or not words2:
            return 0.0
    
        intersection = words1.intersection(words2)
        union = words1.union(words2)
    
        return len(intersection) / len(union) if union else 0.0

    def _adjust_vni_confidence(self, vni_responses: Dict, stimulus_data: Dict):
        """Adjust VNI confidence scores based on performance"""
    
        for vni_id, response_data in vni_responses.items():
            vni_instance = self.get_vni_by_id(vni_id)
            if not vni_instance:
                continue
            
            current_confidence = response_data.get('confidence', 0.5)
            response_content = response_data.get('content', '')
            response_type = response_data.get('response_type', 'unknown')
            stimulus_type = stimulus_data.get('stimulus_type', 'query')
        
            # Calculate performance metrics
            performance_score = self._evaluate_response_performance(
                response_content, stimulus_data, response_type, stimulus_type
            )
        
            # Update VNI's internal confidence if it has such capability
            if hasattr(vni_instance, 'update_confidence'):
                try:
                    vni_instance.update_confidence(performance_score)
                    logger.debug(f"Updated confidence for {vni_id}: +{performance_score:.3f}")
                except Exception as e:
                    logger.debug(f"Could not update confidence for {vni_id}: {e}")
        
            # Track in orchestrator's learning stats
            if hasattr(self.orchestrator, 'learning_stats'):
                if vni_id not in self.orchestrator.learning_stats:
                    self.orchestrator.learning_stats[vni_id] = {
                        'total_responses': 0,
                        'avg_confidence': 0.5,
                        'performance_history': []
                    }
            
                stats = self.orchestrator.learning_stats[vni_id]
                stats['total_responses'] += 1
                stats['performance_history'].append(performance_score)
                # Update moving average
                stats['avg_confidence'] = (
                    stats['avg_confidence'] * 0.9 + performance_score * 0.1
                )

    def _evaluate_response_performance(self, response: str, stimulus_data: Dict, 
                                       response_type: str, stimulus_type: str) -> float:
        """Evaluate how good a response was"""
        score = 0.5  # Base score
    
        # Length heuristic - appropriate length responses are better
        response_length = len(response.split())
        if 10 <= response_length <= 500:  # Reasonable response length
            score += 0.1
    
        # Specificity check - responses with domain terms might be better
        domain_keywords = {
            'medical': ['patient', 'symptom', 'diagnosis', 'treatment', 'medication', 'health', 'doctor', 'hospital'],
            'legal': ['law', 'statute', 'regulation', 'case', 'precedent', 'legal', 'contract', 'court'],
            'general': ['information', 'explanation', 'context', 'overview', 'help', 'assist', 'answer']
        }
    
        if response_type in domain_keywords:
            keywords = domain_keywords[response_type]
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in response.lower())
            if keyword_count > 0:
                score += min(0.2, keyword_count * 0.05)
    
        # Stimulus type matching - if VNI type matches stimulus type
        if response_type in stimulus_type.lower():
            score += 0.15
    
        return min(max(score, 0.1), 1.0)  # Clamp between 0.1 and 1.0

    def _learn_stimulus_patterns(self, stimulus_data: Dict, vni_responses: Dict):
        """Learn patterns from successful stimulus-response pairs"""
    
        if not hasattr(self.orchestrator, 'pattern_library'):
            self.orchestrator.pattern_library = {}
    
        stimulus_content = stimulus_data.get('content', '')
        stimulus_type = stimulus_data.get('stimulus_type', 'query')
    
        # Extract keywords from stimulus
        keywords = self._extract_keywords(stimulus_content)
    
        # Store which VNIs responded well
        successful_vnis = []
        for vni_id, response_data in vni_responses.items():
            if response_data.get('confidence', 0) > 0.6:
                successful_vnis.append(vni_id)
    
        if successful_vnis and keywords:
            pattern_key = f"{stimulus_type}:{','.join(sorted(keywords[:3]))}"
        
            if pattern_key not in self.orchestrator.pattern_library:
                self.orchestrator.pattern_library[pattern_key] = {
                    'successful_vnis': successful_vnis,
                    'count': 1,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat()
                }
            else:
                self.orchestrator.pattern_library[pattern_key]['successful_vnis'] = list(
                    set(self.orchestrator.pattern_library[pattern_key]['successful_vnis'] + successful_vnis)
                )
                self.orchestrator.pattern_library[pattern_key]['count'] += 1
                self.orchestrator.pattern_library[pattern_key]['last_seen'] = datetime.now().isoformat()
        
            logger.debug(f"Learned pattern: {pattern_key} → {successful_vnis}")

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text (simple implementation)"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
        words = text.lower().split()
        keywords = []
    
        for word in words:
            # Basic filtering
            if (len(word) > 3 and 
                word not in stop_words and
                word.isalpha()):
                keywords.append(word)
    
        # Return most frequent keywords
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(max_keywords)]

    def _update_synaptic_weights(self, processing_result: Dict[str, Any]):
        """Update synaptic weights based on overall processing success"""
    
        if not hasattr(self.orchestrator, 'synaptic_connections'):
            return
    
        total_responses = len(processing_result.get('vni_responses', {}))
        if total_responses == 0:
            return
    
        # Calculate average confidence
        confidences = [r.get('confidence', 0) for r in processing_result['vni_responses'].values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
        # Boost all active connections if processing was successful
        if avg_confidence > 0.6:
            for connection_id, connection in self.orchestrator.synaptic_connections.items():
                if hasattr(connection, 'strength'):
                    # Slight reinforcement
                    connection.strength = min(1.0, connection.strength + 0.01)
                    logger.debug(f"Reinforced connection {connection_id}")

    def _create_synaptic_connection(self, source_id: str, target_id: str, initial_strength: float = 0.5):
        """Create a new synaptic connection between VNIs"""
        if not hasattr(self.orchestrator, 'synaptic_connections'):
            self.orchestrator.synaptic_connections = {}
    
        connection_id = f"{source_id}→{target_id}"
    
        # Create a simple connection object
        class SynapticConnection:
            def __init__(self, source, target, strength):
                self.source = source
                self.target = target
                self.strength = strength
                self.created_at = datetime.now().isoformat()
                self.last_updated = datetime.now().isoformat()
            
            def __repr__(self):
                return f"SynapticConnection({self.source}→{self.target}, strength={self.strength})"
    
        connection = SynapticConnection(source_id, target_id, initial_strength)
    
        self.orchestrator.synaptic_connections[connection_id] = connection
        logger.info(f"Created new synaptic connection: {connection_id} with strength {initial_strength}")
    
    def debug_vni_access(self) -> Dict[str, Any]:
        """Debug method to check VNI access"""
        debug_info = {
            'orchestrator_has_vni_instances': hasattr(self.orchestrator, 'vni_instances'),
            'vni_instances_type': None,
            'vni_count': 0,
            'vni_details': {}
        }
        
        if hasattr(self.orchestrator, 'vni_instances'):
            vni_instances = self.orchestrator.vni_instances
            debug_info['vni_instances_type'] = type(vni_instances).__name__
            
            if isinstance(vni_instances, dict):
                debug_info['vni_count'] = len(vni_instances)
                for vni_id, vni in vni_instances.items():
                    debug_info['vni_details'][vni_id] = {
                        'type': vni.__class__.__name__,
                        'has_process_query': hasattr(vni, 'process_query'),
                        'has_generate_response': hasattr(vni, 'generate_response'),
                        'has_respond_to_query': hasattr(vni, 'respond_to_query'),
                        'has_process': hasattr(vni, 'process'),
                        'methods': [m for m in dir(vni) if not m.startswith('_')][:10]
                    }
            elif isinstance(vni_instances, list):
                debug_info['vni_count'] = len(vni_instances)
                for i, vni in enumerate(vni_instances):
                    vni_id = f"vni_{i}"
                    debug_info['vni_details'][vni_id] = {
                        'type': vni.__class__.__name__,
                        'has_process_query': hasattr(vni, 'process_query'),
                        'has_generate_response': hasattr(vni, 'generate_response'),
                        'methods': [m for m in dir(vni) if not m.startswith('_')][:10]
                    }
        
        return debug_info
