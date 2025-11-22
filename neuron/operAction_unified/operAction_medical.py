# operAction_medical.py
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger("operAction_medical")

class OperActionPerformance:
    def __init__(self):
        self.response_quality_history = []
        self.response_time_history = []
        self.domain_specific_metrics = {}
        self.success_patterns = []
    
    def record_outcome(self, query, response, human_feedback=None):
        """Track performance for continuous improvement"""
        quality_score = self.assess_response_quality(response, human_feedback)
        self.response_quality_history.append(quality_score)
        
        # Learn from high-quality responses
        if quality_score > 0.8:
            self.learn_from_success(query, response)
        
        return quality_score
    
    def assess_response_quality(self, response, human_feedback=None):
        """Assess response quality automatically or via feedback"""
        if human_feedback is not None:
            # Use explicit human feedback
            return human_feedback
        
        # Automatic quality assessment for medical responses
        quality_score = 0.5  # Base score
        
        # Check for comprehensive medical analysis
        if 'medical_analysis' in response:
            analysis = response['medical_analysis']
            if analysis.get('diagnoses') and len(analysis['diagnoses']) > 0:
                quality_score += 0.3
            if analysis.get('treatments') and len(analysis['treatments']) > 0:
                quality_score += 0.2
            if response.get('confidence_score', 0) > 0.7:
                quality_score += 0.1
        
        return min(quality_score, 1.0)  # Cap at 1.0
    
    def learn_from_success(self, query, response):
        """Extract patterns from successful medical responses"""
        success_pattern = {
            'query_pattern': self.extract_medical_query_pattern(query),
            'response_pattern': self.extract_medical_response_pattern(response),
            'timestamp': time.time(),
            'quality_score': self.response_quality_history[-1]
        }
        self.success_patterns.append(success_pattern)
        
        # Keep only recent patterns (last 100)
        if len(self.success_patterns) > 100:
            self.success_patterns = self.success_patterns[-100:]
    
    def extract_medical_query_pattern(self, query):
        """Extract medical-specific query patterns"""
        if isinstance(query, dict) and 'abstraction_data' in query:
            cognitive_data = query['abstraction_data'].get('cognitive', {})
            concepts = cognitive_data.get('concepts', [])
            return {
                'symptom_count': len([c for c in concepts if any(symptom in c.lower() for symptom in ['pain', 'fever', 'cough', 'headache'])]),
                'condition_indicators': len([c for c in concepts if any(cond in c.lower() for cond in ['disease', 'infection', 'disorder'])]),
                'key_symptoms': [c for c in concepts if any(symptom in c.lower() for symptom in ['fever', 'cough', 'pain', 'nausea', 'vomiting'])]
            }
        return {'raw_query': str(query)[:200]}
    
    def extract_medical_response_pattern(self, response):
        """Extract medical response patterns"""
        pattern = {
            'diagnoses_made': 0,
            'treatments_suggested': 0,
            'urgency_assessed': False,
            'confidence_level': response.get('confidence_score', 0)
        }
        
        if 'medical_analysis' in response:
            analysis = response['medical_analysis']
            pattern['diagnoses_made'] = len(analysis.get('diagnoses', []))
            pattern['treatments_suggested'] = len(analysis.get('treatments', []))
            pattern['urgency_assessed'] = any(d.get('urgency') for d in analysis.get('diagnoses', []))
        
        return pattern
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.response_quality_history:
            return {
                'average_quality': 0,
                'total_responses': 0,
                'recent_trend': 0
            }
        
        return {
            'average_quality': np.mean(self.response_quality_history),
            'total_responses': len(self.response_quality_history),
            'recent_trend': np.mean(self.response_quality_history[-10:]) if len(self.response_quality_history) >= 10 else np.mean(self.response_quality_history),
            'success_patterns_stored': len(self.success_patterns)
        }

@dataclass
class MedicalOperActionConfig:
    """Configuration for medical operAction VNI"""
    vni_id: str = "operAction_medical"
    reasoning_depth: str = "comprehensive"  # basic, intermediate, comprehensive
    medical_knowledge_base: Dict[str, Any] = None
    confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.medical_knowledge_base is None:
            self.medical_knowledge_base = {
                'symptom_patterns': {
                    'fever_cough': ['respiratory_infection', 'pneumonia', 'covid'],
                    'chest_pain_shortness_breath': ['heart_attack', 'angina', 'pulmonary_embolism'],
                    'headache_nausea': ['migraine', 'concussion', 'hypertension'],
                    'abdominal_pain_vomiting': ['gastroenteritis', 'appendicitis', 'food_poisoning']
                },
                'disease_treatments': {
                    'respiratory_infection': ['antibiotics', 'rest', 'fluids'],
                    'pneumonia': ['antibiotics', 'hospitalization', 'oxygen_therapy'],
                    'covid': ['antiviral_medication', 'isolation', 'symptomatic_treatment'],
                    'heart_attack': ['emergency_care', 'aspirin', 'cardiac_intervention']
                },
                'risk_factors': {
                    'heart_disease': ['hypertension', 'high_cholesterol', 'smoking', 'diabetes'],
                    'respiratory_disease': ['smoking', 'asthma', 'environmental_exposure'],
                    'neurological_disorders': ['family_history', 'head_trauma', 'age']
                }
            }

class MedicalKnowledgeGraph:
    def __init__(self):
        self.symptom_disease_graph = self.build_medical_ontology()
        self.drug_interactions = self.load_drug_database()
        self.clinical_guidelines = self.load_guidelines()
    
    def build_medical_ontology(self):
        """Build proper medical relationships"""
        G = {
            'symptoms': {
                'fever': {'related_diseases': ['covid', 'flu', 'pneumonia'], 'severity': 0.7},
                'cough': {'related_diseases': ['covid', 'bronchitis', 'pneumonia'], 'severity': 0.5}
            },
            'diseases': {
                'covid': {
                    'common_symptoms': ['fever', 'cough', 'fatigue'],
                    'treatments': ['rest', 'hydration', 'antiviral_meds'],
                    'urgency': 'high'
                }
            }
        }
        return G
    
    def diagnose_from_symptoms(self, symptoms, patient_context=None):
        """Use graph-based diagnosis instead of pattern matching"""
        candidates = []
        for symptom in symptoms:
            if symptom in self.symptom_disease_graph['symptoms']:
                diseases = self.symptom_disease_graph['symptoms'][symptom]['related_diseases']
                for disease in diseases:
                    # Calculate evidence score
                    evidence = self.calculate_evidence(disease, symptoms)
                    candidates.append({
                        'disease': disease,
                        'evidence_score': evidence,
                        'supporting_symptoms': self.get_supporting_symptoms(disease, symptoms)
                    })
        
        return sorted(candidates, key=lambda x: x['evidence_score'], reverse=True)

class MedicalReasoningEngine(nn.Module):
    """Medical reasoning and diagnosis engine"""
    def __init__(self, config: MedicalOperActionConfig):
        super().__init__()
        self.config = config
        self.knowledge_graph = MedicalKnowledgeGraph()       
        # Medical reasoning networks
        self.symptom_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        self.diagnosis_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Sigmoid()
        )
        
        self.treatment_recommender = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.Sigmoid()
        )
        
        # Risk assessment
        self.risk_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8 risk categories
            nn.Sigmoid()
        )
        
    def extract_medical_concepts(self, abstraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medical concepts from abstraction data"""
        medical_concepts = {
            'symptoms': [],
            'conditions': [],
            'treatments': [],
            'risk_factors': [],
            'body_parts': []
        }
        
        # Extract from cognitive level
        if 'cognitive' in abstraction_data:
            cognitive = abstraction_data['cognitive']
            concepts = cognitive.get('concepts', [])
            
            # Medical concept mapping
            symptom_keywords = ['fever', 'cough', 'pain', 'headache', 'nausea', 'vomiting', 
                              'shortness', 'breath', 'chest', 'abdominal']
            condition_keywords = ['cancer', 'diabetes', 'covid', 'pneumonia', 'infection',
                                'disease', 'disorder', 'syndrome']
            treatment_keywords = ['medicine', 'treatment', 'therapy', 'surgery', 'medication']
            risk_keywords = ['risk', 'factor', 'history', 'genetic', 'family']
            body_keywords = ['heart', 'lung', 'brain', 'liver', 'kidney', 'stomach']
            
            for concept in concepts:
                concept_lower = concept.lower()
                if any(keyword in concept_lower for keyword in symptom_keywords):
                    medical_concepts['symptoms'].append(concept)
                elif any(keyword in concept_lower for keyword in condition_keywords):
                    medical_concepts['conditions'].append(concept)
                elif any(keyword in concept_lower for keyword in treatment_keywords):
                    medical_concepts['treatments'].append(concept)
                elif any(keyword in concept_lower for keyword in risk_keywords):
                    medical_concepts['risk_factors'].append(concept)
                elif any(keyword in concept_lower for keyword in body_keywords):
                    medical_concepts['body_parts'].append(concept)
        
        return medical_concepts
    
    def analyze_symptom_patterns(self, symptoms: List[str]) -> Dict[str, float]:
        """Analyze symptoms to identify potential conditions"""
        symptom_patterns = self.config.medical_knowledge_base['symptom_patterns']
        condition_scores = {}
        
        # Convert symptoms to pattern key
        symptom_key = '_'.join(sorted([s.lower() for s in symptoms]))
        
        # Exact pattern matching
        for pattern, conditions in symptom_patterns.items():
            pattern_symptoms = set(pattern.split('_'))
            input_symptoms = set(s.lower() for s in symptoms)
            
            # Calculate overlap
            overlap = len(pattern_symptoms.intersection(input_symptoms))
            if overlap > 0:
                score = overlap / len(pattern_symptoms)
                for condition in conditions:
                    condition_scores[condition] = max(condition_scores.get(condition, 0), score)
        
        return condition_scores
    
    def generate_differential_diagnosis(self, medical_concepts: Dict[str, Any], 
                                      cognitive_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate differential diagnosis based on symptoms and context"""
        
        diagnoses = []
        symptoms = medical_concepts.get('symptoms', [])
        
        if symptoms:
            # Pattern-based analysis
            pattern_scores = self.analyze_symptom_patterns(symptoms)
            
            # Neural network analysis
            symptom_embedding = self.symptom_analyzer(cognitive_tensor)
            diagnosis_probs = self.diagnosis_predictor(symptom_embedding)
            
            # Combine pattern and neural network results
            for condition, pattern_score in pattern_scores.items():
                neural_score = diagnosis_probs[0][hash(condition) % 16].item()  # Simplified
                combined_score = 0.6 * pattern_score + 0.4 * neural_score
                
                if combined_score > self.config.confidence_threshold:
                    diagnoses.append({
                        'condition': condition,
                        'confidence': combined_score,
                        'supporting_symptoms': symptoms,
                        'recommended_tests': self.suggest_diagnostic_tests(condition),
                        'urgency': self.assess_urgency(condition, symptoms)
                    })
        
        # Sort by confidence
        diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
        return diagnoses[:3]  # Return top 3 diagnoses
    
    def suggest_diagnostic_tests(self, condition: str) -> List[str]:
        """Suggest diagnostic tests for a condition"""
        test_mapping = {
            'respiratory_infection': ['chest_xray', 'blood_test', 'sputum_culture'],
            'pneumonia': ['chest_xray', 'ct_scan', 'blood_cultures'],
            'covid': ['pcr_test', 'rapid_antigen_test', 'chest_ct'],
            'heart_attack': ['ecg', 'troponin_test', 'echocardiogram'],
            'gastroenteritis': ['stool_test', 'blood_test', 'abdominal_ultrasound']
        }
        return test_mapping.get(condition, ['general_physical_exam', 'basic_blood_work'])
    
    def assess_urgency(self, condition: str, symptoms: List[str]) -> str:
        """Assess urgency of medical condition"""
        emergency_conditions = ['heart_attack', 'pulmonary_embolism', 'stroke', 'appendicitis']
        urgent_conditions = ['pneumonia', 'covid', 'severe_infection']
        
        if condition in emergency_conditions:
            return 'emergency'
        elif condition in urgent_conditions:
            return 'urgent'
        else:
            return 'routine'
    
    def recommend_treatments(self, diagnosis: Dict[str, Any], 
                           medical_concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend treatments based on diagnosis"""
        treatments = []
        condition = diagnosis['condition']
        
        # Knowledge base recommendations
        knowledge_treatments = self.config.medical_knowledge_base['disease_treatments'].get(condition, [])
        
        for treatment in knowledge_treatments:
            treatments.append({
                'treatment': treatment,
                'type': self.classify_treatment_type(treatment),
                'evidence_level': 'established',
                'rationale': f"Standard treatment for {condition}"
            })
        
        # Neural network recommendations
        diagnosis_embedding = torch.tensor([diagnosis['confidence']] * 64)  # Simplified
        treatment_scores = self.treatment_recommender(diagnosis_embedding.unsqueeze(0))
        
        # Additional considerations
        if 'allergy' in medical_concepts.get('risk_factors', []):
            treatments.append({
                'treatment': 'allergy_testing',
                'type': 'diagnostic',
                'evidence_level': 'precautionary',
                'rationale': 'Recommended due to allergy history'
            })
        
        return treatments
    
    def classify_treatment_type(self, treatment: str) -> str:
        """Classify treatment type"""
        medication_keywords = ['antibiotic', 'antiviral', 'medication', 'drug']
        procedure_keywords = ['surgery', 'therapy', 'intervention']
        lifestyle_keywords = ['rest', 'diet', 'exercise']
        
        treatment_lower = treatment.lower()
        if any(keyword in treatment_lower for keyword in medication_keywords):
            return 'medication'
        elif any(keyword in treatment_lower for keyword in procedure_keywords):
            return 'procedure'
        elif any(keyword in treatment_lower for keyword in lifestyle_keywords):
            return 'lifestyle'
        else:
            return 'general'
    
    def assess_risks(self, medical_concepts: Dict[str, Any], 
                    structural_tensor: torch.Tensor) -> Dict[str, Any]:
        """Assess medical risks based on symptoms and context"""
        risks = {}
        
        # Neural network risk assessment
        risk_scores = self.risk_assessor(structural_tensor)
        
        risk_categories = [
            'cardiovascular_risk', 'respiratory_risk', 'neurological_risk',
            'infectious_risk', 'metabolic_risk', 'trauma_risk',
            'environmental_risk', 'genetic_risk'
        ]
        
        for i, category in enumerate(risk_categories):
            if i < len(risk_scores[0]):
                risks[category] = risk_scores[0][i].item()
        
        # Knowledge-based risk factors
        symptoms = medical_concepts.get('symptoms', [])
        if 'chest_pain' in [s.lower() for s in symptoms]:
            risks['cardiovascular_risk'] = max(risks.get('cardiovascular_risk', 0), 0.8)
        
        if 'fever' in [s.lower() for s in symptoms]:
            risks['infectious_risk'] = max(risks.get('infectious_risk', 0), 0.7)
        
        return risks
    
    def generate_medical_advice(self, diagnoses: List[Dict[str, Any]], 
                              treatments: List[Dict[str, Any]],
                              risks: Dict[str, Any]) -> str:
        """Generate comprehensive medical advice"""
        
        if not diagnoses:
            return "Insufficient information for specific medical advice. Please consult a healthcare professional."
        
        primary_diagnosis = diagnoses[0]
        advice_parts = []
        
        # Diagnosis summary
        advice_parts.append(f"Based on the symptoms described, the most likely condition is {primary_diagnosis['condition']} (confidence: {primary_diagnosis['confidence']:.1%}).")
        
        # Urgency assessment
        urgency = primary_diagnosis.get('urgency', 'routine')
        if urgency == 'emergency':
            advice_parts.append("This appears to be a medical emergency. Seek immediate medical attention.")
        elif urgency == 'urgent':
            advice_parts.append("This condition requires prompt medical evaluation.")
        
        # Treatment recommendations
        if treatments:
            advice_parts.append("Recommended approaches:")
            for treatment in treatments[:3]:  # Top 3 treatments
                advice_parts.append(f"- {treatment['treatment']} ({treatment['type']})")
        
        # Risk factors
        high_risks = {k: v for k, v in risks.items() if v > 0.7}
        if high_risks:
            advice_parts.append("Notable risk factors identified:")
            for risk, score in high_risks.items():
                advice_parts.append(f"- {risk.replace('_', ' ')} (score: {score:.1%})")
        
        return " ".join(advice_parts)

class OperActionMedical(nn.Module):
    """Medical reasoning operAction VNI"""
    
    def __init__(self, config: MedicalOperActionConfig = None):
        super().__init__()
        self.config = config or MedicalOperActionConfig()
        self.reasoning_engine = MedicalReasoningEngine(self.config)
        
        logger.info(f"Medical operAction VNI initialized with ID: {self.config.vni_id}")
    
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process medical reasoning request"""
        
        try:
            abstraction_data = input_data.get('abstraction_data', {})
            
            # Extract medical concepts
            medical_concepts = self.reasoning_engine.extract_medical_concepts(abstraction_data)
            
            # Get cognitive tensor for neural processing
            cognitive_tensor = abstraction_data.get('cognitive', {}).get('tensor', torch.zeros(256))
            structural_tensor = abstraction_data.get('structural', {}).get('tensor', torch.zeros(256))
            
            # Generate differential diagnosis
            diagnoses = self.reasoning_engine.generate_differential_diagnosis(
                medical_concepts, cognitive_tensor
            )
            
            # Recommend treatments
            treatments = []
            if diagnoses:
                treatments = self.reasoning_engine.recommend_treatments(
                    diagnoses[0], medical_concepts
                )
            
            # Assess risks
            risks = self.reasoning_engine.assess_risks(medical_concepts, structural_tensor)
            
            # Generate final advice
            medical_advice = self.reasoning_engine.generate_medical_advice(
                diagnoses, treatments, risks
            )
            
            # Compile results
            results = {
                'medical_analysis': {
                    'diagnoses': diagnoses,
                    'treatments': treatments,
                    'risk_assessment': risks,
                    'identified_concepts': medical_concepts
                },
                'medical_advice': medical_advice,
                'confidence_score': diagnoses[0]['confidence'] if diagnoses else 0.0,
                'processing_metadata': {
                    'symptoms_analyzed': len(medical_concepts.get('symptoms', [])),
                    'conditions_considered': len(diagnoses),
                    'risk_categories_assessed': len(risks)
                }
            }
            
            # Add VNI metadata
            results['vni_metadata'] = {
                'vni_id': self.config.vni_id,
                'vni_type': 'operAction_medical',
                'processing_stages': ['concept_extraction', 'diagnosis_generation', 
                                    'treatment_recommendation', 'risk_assessment'],
                'success': True,
                'domain': 'medical'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Medical operAction processing failed: {str(e)}")
            return self._generate_error_output(str(e))
    
    def _generate_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Generate error output"""
        return {
            'medical_analysis': {},
            'medical_advice': f"Medical analysis unavailable: {error_msg}",
            'vni_metadata': {
                'vni_id': self.config.vni_id,
                'vni_type': 'operAction_medical',
                'success': False,
                'error': error_msg,
                'domain': 'medical'
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return VNI capabilities description"""
        return {
            'vni_type': 'operAction_medical',
            'description': 'Medical diagnosis and treatment reasoning VNI',
            'capabilities': [
                'Symptom pattern analysis',
                'Differential diagnosis generation',
                'Treatment recommendation',
                'Medical risk assessment',
                'Urgency evaluation'
            ],
            'input_types': ['medical_abstraction_data'],
            'output_types': ['medical_analysis', 'medical_advice', 'risk_assessment'],
            'domain': 'medical'
        }

# Demonstration and testing
def test_medical_operAction():
    """Test the medical operAction with sample inputs"""
    
    # Initialize medical operAction
    config = MedicalOperActionConfig(vni_id="operAction_medical_test_001")
    medical_vni = OperActionMedical(config)
    
    # Create test input data
    test_input = {
        'abstraction_data': {
            'cognitive': {
                'tensor': torch.randn(256),
                'concepts': ['patient', 'fever', 'cough', 'chest pain', 'breathing difficulty'],
                'intent': 'diagnosis'
            },
            'structural': {
                'tensor': torch.randn(256),
                'logical_flow': 'descriptive'
            },
            'signal': {
                'tensor': torch.randn(256)
            }
        },
        'metadata': {
            'source_topics': ['medical'],
            'cross_domain': False
        }
    }
    
    print("=== Medical OperAction Demo Test ===\n")
    
    with torch.no_grad():
        results = medical_vni(test_input)
    
    # Display results
    print("Medical Analysis Results:")
    print(f"Primary Diagnosis: {results['medical_analysis']['diagnoses'][0]['condition']}")
    print(f"Confidence: {results['medical_analysis']['diagnoses'][0]['confidence']:.1%}")
    print(f"Urgency: {results['medical_analysis']['diagnoses'][0]['urgency']}")
    
    print("\nRecommended Treatments:")
    for treatment in results['medical_analysis']['treatments'][:2]:
        print(f"- {treatment['treatment']} ({treatment['type']})")
    
    print(f"\nMedical Advice: {results['medical_advice']}")
    
    return medical_vni

if __name__ == "__main__":
    # Run demonstration
    test_medical_operAction() 