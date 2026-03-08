# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# neuron/reinforcement_learning/training/training_pipeline.py
"""
BabyBIONN Complete Training Pipeline
Integrates pretraining, reinforcement learning, and analytics
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Import your existing components
from pretraining_processor import BabyBIONNPretrainer, PretrainingConfig, create_pretrainer
from learning_analytics import LearningAnalytics

logger = logging.getLogger("babybionn_trainer")

@dataclass
class TrainingConfig:
    """Configuration for the complete training pipeline"""
    # Pretraining settings
    pretraining_epochs: int = 3
    domain_knowledge_path: str = "knowledge_bases"
    
    # RL Training settings
    training_episodes: int = 1000
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    batch_size: int = 32
    
    # Evaluation settings
    evaluation_interval: int = 100
    validation_set_size: int = 50
    
    # Analytics settings
    save_interval: int = 50
    model_checkpoint_path: str = "checkpoints"


class DomainKnowledgeLoader:
    """Loads and manages domain knowledge for training - UPDATED FOR YOUR STRUCTURE"""
    
    def __init__(self, knowledge_path: str):
        self.knowledge_path = Path(knowledge_path)
        self.knowledge_path.mkdir(exist_ok=True)
    
    def load_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """Load domain knowledge from JSON files - UPDATED FOR YOUR NAMING"""
        # Try multiple possible file patterns
        possible_files = [
            self.knowledge_path / f"knowledge_{domain}_{domain}_0.json",  # existing pattern
            self.knowledge_path / f"knowledge_{domain}_{domain}_1.json",  # existing pattern  
            self.knowledge_path / f"{domain}_knowledge.json",             # New pattern
            self.knowledge_path / f"{domain}_knowledge_base.json"         # Pretrainer pattern
        ]
        
        for domain_file in possible_files:
            if domain_file.exists():
                try:
                    with open(domain_file, 'r') as f:
                        knowledge_data = json.load(f)
                    logger.info(f"Loaded domain knowledge from: {domain_file}")
                    return self._convert_to_pretraining_format(knowledge_data, domain)
                except Exception as e:
                    logger.error(f"Error loading {domain_file}: {e}")
                    continue
        
        logger.warning(f"No knowledge file found for domain: {domain}")
        return self._create_default_knowledge(domain)
    
    def _convert_to_pretraining_format(self, knowledge_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Convert your existing knowledge format to pretraining format"""
        # Your existing files might have different structure than what pretrainer expects
        # The pretrainer expects: {'concepts': {}, 'reasoning_patterns': {}, 'response_templates': {}}
        
        # If it's already in pretraining format, return as-is
        if 'concepts' in knowledge_data or 'synaptic_patterns' in knowledge_data:
            return knowledge_data
        
        # Otherwise convert from your VNI knowledge format
        converted_data = {
            'concepts': {},
            'reasoning_patterns': {},
            'response_templates': {},
            'domain': domain
        }
        
        # Extract concepts from your knowledge structure
        if 'knowledge' in knowledge_data:
            for concept_name, concept_info in knowledge_data['knowledge'].items():
                converted_data['concepts'][concept_name] = {
                    'response': concept_info.get('response', ''),
                    'confidence': concept_info.get('confidence', 0.7),
                    'usage_count': concept_info.get('usage_count', 0)
                }
        
        return converted_data
    
    def _create_default_knowledge(self, domain: str) -> Dict[str, Any]:
        """Create default knowledge structure for a domain"""
        return {
            "domain": domain,
            "concepts": {},
            "reasoning_patterns": {},
            "response_templates": {},
            "metadata": {"created": time.time(), "source": "default"}
        }
    
    def get_all_domains(self) -> List[str]:
        """Get list of all available domains from your knowledge bases"""
        domains = set()
        
        # Look for your existing file patterns
        for file_pattern in ["knowledge_*_*_0.json", "knowledge_*_*_1.json", "*_knowledge.json"]:
            for file in self.knowledge_path.glob(file_pattern):
                # Extract domain from filename patterns like "knowledge_medical_medical_0.json"
                if file.name.startswith("knowledge_"):
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        domains.add(parts[1])  # Second part is domain
                else:
                    # For patterns like "medical_knowledge.json"
                    domain = file.name.replace('_knowledge.json', '')
                    domains.add(domain)
        
        return list(domains)

class TrainingEpisode:
    """Represents a single training episode"""
    
    def __init__(self, episode_id: int, domain: str, query: str, expected_response: str):
        self.episode_id = episode_id
        self.domain = domain
        self.query = query
        self.expected_response = expected_response
        self.actual_response = None
        self.confidence = 0.0
        self.reward = 0.0
        self.vni_used = None
        self.timestamp = time.time()
    
    def calculate_reward(self, similarity_threshold: float = 0.7) -> float:
        """Calculate reward based on response quality"""
        if not self.actual_response:
            return -1.0
        
        # Calculate semantic similarity (simplified)
        similarity = self._calculate_similarity(
            self.expected_response, 
            self.actual_response
        )
        
        # Base reward on similarity
        base_reward = similarity * 2 - 1  # Convert to -1 to 1 range
        
        # Bonus for high confidence when correct
        if similarity > similarity_threshold and self.confidence > 0.8:
            base_reward += 0.5
        
        # Penalty for high confidence when wrong
        if similarity < 0.3 and self.confidence > 0.7:
            base_reward -= 0.5
            
        return max(min(base_reward, 1.0), -1.0)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simplified text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class ReinforcementTrainer:
    """Handles reinforcement learning aspect of training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.q_table = {}  # Simplified Q-table
        self.training_history = []
    
    def update_policy(self, episode: TrainingEpisode):
        """Update RL policy based on episode results"""
        state = self._get_state_representation(episode)
        action = episode.vni_used
        
        # Initialize Q-values if needed
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Simple Q-learning update
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.config.learning_rate * (
            episode.reward + self.config.discount_factor * self._get_max_future_q(state) - current_q
        )
        
        # Record training data
        self.training_history.append({
            'episode_id': episode.episode_id,
            'state': state,
            'action': action,
            'reward': episode.reward,
            'q_value': self.q_table[state][action],
            'timestamp': time.time()
        })
    
    def _get_state_representation(self, episode: TrainingEpisode) -> str:
        """Convert episode to state representation"""
        query_len = len(episode.query.split())
        length_category = "short" if query_len < 5 else "medium" if query_len < 10 else "long"
        
        return f"{episode.domain}_{length_category}"
    
    def _get_max_future_q(self, state: str) -> float:
        """Get maximum Q-value for future state"""
        if state not in self.q_table or not self.q_table[state]:
            return 0.0
        return max(self.q_table[state].values())
    
    def select_vni(self, domain: str, query: str) -> str:
        """Select VNI based on current policy"""
        state = self._get_state_representation(
            TrainingEpisode(0, domain, query, "")
        )
        
        # Exploration vs exploitation
        if np.random.random() < self.config.exploration_rate:
            # Explore: random VNI selection
            vnis = [f"VNI_{domain}_001", "VNI_general_001", "VNI_technical_001"]
            return np.random.choice(vnis)
        else:
            # Exploit: best known VNI
            if state in self.q_table and self.q_table[state]:
                return max(self.q_table[state].items(), key=lambda x: x[1])[0]
            else:
                return f"VNI_{domain}_001"

class BabyBIONNTrainingPipeline:
    """
    Complete training pipeline integrating all components
    """
    
    def __init__(self, vni_manager, rl_system, config: TrainingConfig = None):
        self.vni_manager = vni_manager
        self.rl_system = rl_system
        self.config = config or TrainingConfig()
        
        # Initialize your existing pretrainer
        self.pretrainer = create_pretrainer(vni_manager, rl_system)
        self.knowledge_loader = DomainKnowledgeLoader(self.config.domain_knowledge_path)
        self.rl_trainer = ReinforcementTrainer(self.config)
        self.learning_analytics = LearningAnalytics()
        
        # Training state
        self.current_episode = 0
        self.is_training = False
        self.training_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'average_reward': 0.0,
            'domain_performance': {},
            'start_time': None
        }
        
        logger.info("BabyBIONN Training Pipeline initialized")
    
    def run_complete_training(self, domains: List[str] = None):
        """
        Run complete training pipeline including pretraining and RL
        """
        if not domains:
            domains = self.knowledge_loader.get_all_domains()
        
        logger.info(f"Starting complete training for domains: {domains}")
        
        # Phase 1: Pretraining
        self._run_pretraining_phase(domains)
        
        # Phase 2: Reinforcement Learning
        self._run_rl_training_phase(domains)
        
        # Phase 3: Final Evaluation
        final_metrics = self._run_final_evaluation(domains)
        
        logger.info("Complete training finished")
        return final_metrics
    
    def _run_pretraining_phase(self, domains: List[str]):
        """Run pretraining phase for all domains"""
        logger.info("Starting pretraining phase")
        
        pretraining_results = {}
        for domain in domains:
            logger.info(f"Pretraining domain: {domain}")
            
            # Load domain knowledge
            domain_knowledge = self.knowledge_loader.load_domain_knowledge(domain)
            
            # Run pretraining using your existing pretrainer
            result = self.pretrainer.pretrain_domain(domain, domain_knowledge)
            pretraining_results[domain] = result
            
            # Record in analytics
            self.learning_analytics.record_interaction(
                session_id=f"pretrain_{domain}",
                vni_type=domain,
                confidence=result.get('success', False),
                feedback="positive" if result.get('success') else "negative"
            )
        
        logger.info("Pretraining phase completed")
        return pretraining_results
    
    def _run_rl_training_phase(self, domains: List[str]):
        """Run reinforcement learning training phase"""
        logger.info("Starting RL training phase")
        
        self.is_training = True
        self.training_stats['start_time'] = time.time()
        
        for episode_num in range(self.config.training_episodes):
            self.current_episode = episode_num
            
            # Select random domain and create training episode
            domain = np.random.choice(domains)
            episode = self._create_training_episode(episode_num, domain)
            
            if episode:
                # Process episode and learn
                self._process_training_episode(episode)
            
            # Periodic evaluation and checkpointing
            if episode_num % self.config.evaluation_interval == 0:
                self._run_periodic_evaluation(episode_num)
            
            if episode_num % self.config.save_interval == 0:
                self._save_training_checkpoint(episode_num)
        
        self.is_training = False
        logger.info("RL training phase completed")
    
    def _create_training_episode(self, episode_id: int, domain: str) -> Optional[TrainingEpisode]:
        """Create a training episode with sample data"""
        try:
            # Load training samples for domain
            training_samples = self._load_training_samples(domain)
            if not training_samples:
                logger.warning(f"No training samples for domain: {domain}")
                return None
            
            # Select random sample
            sample = np.random.choice(training_samples)
            
            return TrainingEpisode(
                episode_id=episode_id,
                domain=domain,
                query=sample['query'],
                expected_response=sample['expected_response']
            )
        except Exception as e:
            logger.error(f"Error creating training episode: {e}")
            return None
    
    def _load_training_samples(self, domain: str) -> List[Dict[str, str]]:
        """Load training samples for a domain"""
        samples_file = Path(self.config.domain_knowledge_path) / f"{domain}_samples.json"
        
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                return json.load(f).get('training_samples', [])
        
        # Generate simple samples if none exist
        return self._generate_sample_training_data(domain)
    
    def _generate_sample_training_data(self, domain: str) -> List[Dict[str, str]]:
        """Generate sample training data for a domain"""
        domain_samples = {
            'medical': [
                {
                    'query': 'What are the symptoms of flu?',
                    'expected_response': 'Common flu symptoms include fever, cough, sore throat, body aches, and fatigue.'
                },
                {
                    'query': 'How to treat a headache?',
                    'expected_response': 'For headaches, rest, hydration, and over-the-counter pain relievers may help.'
                }
            ],
            'legal': [
                {
                    'query': 'What is a contract?',
                    'expected_response': 'A contract is a legally binding agreement between two or more parties.'
                }
            ],
            'technical': [
                {
                    'query': 'How does a computer work?',
                    'expected_response': 'Computers process instructions using CPUs and store data in memory.'
                }
            ]
        }
        
        return domain_samples.get(domain, [])
    
    def _process_training_episode(self, episode: TrainingEpisode):
        """Process a single training episode"""
        try:
            # Select VNI using RL policy
            selected_vni = self.rl_trainer.select_vni(episode.domain, episode.query)
            episode.vni_used = selected_vni
            
            # Simulate VNI processing (replace with actual VNI call)
            episode.actual_response = self._simulate_vni_response(selected_vni, episode.query)
            episode.confidence = np.random.uniform(0.5, 0.95)  # Simulated confidence
            
            # Calculate reward
            episode.reward = episode.calculate_reward()
            
            # Update RL policy
            self.rl_trainer.update_policy(episode)
            
            # Update training statistics
            self._update_training_stats(episode)
            
            # Record in analytics using your existing LearningAnalytics
            self.learning_analytics.record_interaction(
                session_id=f"training_{episode.episode_id}",
                vni_type=episode.domain,
                confidence=episode.confidence,
                feedback="positive" if episode.reward > 0 else "negative"
            )
            
        except Exception as e:
            logger.error(f"Error processing training episode {episode.episode_id}: {e}")
    
    def _simulate_vni_response(self, vni_id: str, query: str) -> str:
        """Simulate VNI response (replace with actual VNI processing)"""
        # In real implementation, this would call your actual VNI system
        return f"Simulated response from {vni_id} for query: {query}"
    
    def _update_training_stats(self, episode: TrainingEpisode):
        """Update training statistics"""
        self.training_stats['total_episodes'] += 1
        
        if episode.reward > 0:
            self.training_stats['successful_episodes'] += 1
        
        # Update domain performance
        domain = episode.domain
        if domain not in self.training_stats['domain_performance']:
            self.training_stats['domain_performance'][domain] = {
                'total_episodes': 0,
                'successful_episodes': 0,
                'total_reward': 0.0
            }
        
        domain_stats = self.training_stats['domain_performance'][domain]
        domain_stats['total_episodes'] += 1
        domain_stats['total_reward'] += episode.reward
        
        if episode.reward > 0:
            domain_stats['successful_episodes'] += 1
        
        # Update overall average reward
        total_reward = sum(stats['total_reward'] for stats in self.training_stats['domain_performance'].values())
        total_domain_episodes = sum(stats['total_episodes'] for stats in self.training_stats['domain_performance'].values())
        
        if total_domain_episodes > 0:
            self.training_stats['average_reward'] = total_reward / total_domain_episodes
    
    def _run_periodic_evaluation(self, episode_num: int):
        """Run periodic evaluation during training"""
        logger.info(f"Running periodic evaluation at episode {episode_num}")
        
        # Generate performance report using your existing LearningAnalytics
        report = self.learning_analytics.generate_learning_report()
        logger.info(f"Training Progress Report:\n{report}")
        
        # Plot confidence trends
        self.learning_analytics.plot_confidence_trend()
        
        # Save current state
        self._save_training_state(episode_num)
    
    def _run_final_evaluation(self, domains: List[str]) -> Dict[str, Any]:
        """Run final evaluation after training"""
        logger.info("Running final evaluation")
        
        evaluation_results = {
            'timestamp': time.time(),
            'domains_tested': domains,
            'performance_metrics': {},
            'training_summary': self.training_stats
        }
        
        # Test each domain
        for domain in domains:
            domain_performance = self._evaluate_domain_performance(domain)
            evaluation_results['performance_metrics'][domain] = domain_performance
        
        # Generate final analytics report
        evaluation_results['analytics_report'] = self.learning_analytics.generate_learning_report()
        
        # Save final model and results
        self._save_final_model(evaluation_results)
        
        logger.info("Final evaluation completed")
        return evaluation_results
    
    def _evaluate_domain_performance(self, domain: str) -> Dict[str, Any]:
        """Evaluate performance on a specific domain"""
        test_samples = self._load_training_samples(domain)
        if not test_samples:
            return {'error': 'No test samples available'}
        
        # Use a subset for evaluation
        evaluation_samples = test_samples[:min(len(test_samples), self.config.validation_set_size)]
        
        correct_responses = 0
        total_confidence = 0.0
        
        for sample in evaluation_samples:
            # Use trained policy to select VNI
            selected_vni = self.rl_trainer.select_vni(domain, sample['query'])
            
            # Simulate response
            response = self._simulate_vni_response(selected_vni, sample['query'])
            confidence = np.random.uniform(0.6, 0.98)  # Simulated
            
            # Check if response is reasonable (simplified)
            if self._is_response_reasonable(response, sample['expected_response']):
                correct_responses += 1
            
            total_confidence += confidence
        
        accuracy = correct_responses / len(evaluation_samples) if evaluation_samples else 0
        avg_confidence = total_confidence / len(evaluation_samples) if evaluation_samples else 0
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'samples_tested': len(evaluation_samples),
            'correct_responses': correct_responses
        }
    
    def _is_response_reasonable(self, actual: str, expected: str) -> bool:
        """Check if response is reasonable (simplified)"""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Simple keyword matching
        expected_keywords = set(expected_lower.split()[:5])
        actual_keywords = set(actual_lower.split())
        
        common_keywords = expected_keywords.intersection(actual_keywords)
        return len(common_keywords) >= 2
    
    def _save_training_checkpoint(self, episode_num: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.model_checkpoint_path)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_data = {
            'episode': episode_num,
            'training_stats': self.training_stats,
            'q_table': self.rl_trainer.q_table,
            'timestamp': time.time()
        }
        
        checkpoint_file = checkpoint_dir / f"checkpoint_episode_{episode_num}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def _save_training_state(self, episode_num: int):
        """Save current training state"""
        state_file = Path(self.config.model_checkpoint_path) / "training_state.json"
        
        state_data = {
            'current_episode': episode_num,
            'training_stats': self.training_stats,
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def _save_final_model(self, evaluation_results: Dict[str, Any]):
        """Save final trained model"""
        model_dir = Path(self.config.model_checkpoint_path)
        model_dir.mkdir(exist_ok=True)
        
        model_data = {
            'training_completed': True,
            'final_episode': self.current_episode,
            'training_stats': self.training_stats,
            'q_table': self.rl_trainer.q_table,
            'evaluation_results': evaluation_results,
            'completion_time': time.time()
        }
        
        model_file = model_dir / "final_trained_model.json"
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Final model saved: {model_file}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'current_episode': self.current_episode,
            'total_episodes': self.config.training_episodes,
            'progress_percentage': (self.current_episode / self.config.training_episodes) * 100,
            'training_stats': self.training_stats,
            'average_reward': self.training_stats['average_reward']
        }

# Factory function
def create_training_pipeline(vni_manager, rl_system) -> BabyBIONNTrainingPipeline:
    """Create and configure training pipeline"""
    config = TrainingConfig(
        pretraining_epochs=3,
        training_episodes=500,
        learning_rate=0.01,
        evaluation_interval=50,
        save_interval=25,
        domain_knowledge_path="knowledge_bases"        
    )
    
    return BabyBIONNTrainingPipeline(vni_manager, rl_system, config) 
