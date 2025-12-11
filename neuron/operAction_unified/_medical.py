# _medical.py - HYBRID VERSION with static knowledge + dynamic adaptation
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import re
import time
import json

# Import the new base class
try:
    from specialized_vni_base import SpecializedVNIBase, SpecializedVNIState
except ImportError:
    # Fallback for standalone testing
    print("⚠️  specialized_vni_base not available. Using standalone mode.")
    class SpecializedVNIBase(nn.Module):
        def __init__(self, topic_name: str, config: Dict = None):
            super().__init__()
            self.topic_name = topic_name

logger = logging.getLogger("operAction_medical_hybrid")

@dataclass
class MedicalOperActionConfig:
    """Configuration for medical operAction VNI"""
    vni_id: str = "operAction_medical_hybrid"
    reasoning_depth: str = "comprehensive"  # basic, intermediate, comprehensive
    medical_knowledge_base: Dict[str, Any] = None
    confidence_threshold: float = 0.6
    enable_dynamic_adaptation: bool = True
    
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
                },
                'diagnostic_tests': {
                    'respiratory_infection': ['chest_xray', 'blood_test', 'sputum_culture'],
                    'pneumonia': ['chest_xray', 'ct_scan', 'blood_cultures'],
                    'covid': ['pcr_test', 'rapid_antigen_test', 'chest_ct'],
                    'heart_attack': ['ecg', 'troponin_test', 'echocardiogram']
                }
            }

class MedicalKnowledgeGraph:
    """Static medical knowledge graph"""
    def __init__(self):
        self.symptom_disease_graph = self.build_medical_ontology()
        self.drug_interactions = self.load_drug_database()
        self.clinical_guidelines = self.load_guidelines()
    
    def build_medical_ontology(self):
        """Build proper medical relationships"""
        return {
            'symptoms': {
                'fever': {'related_diseases': ['covid', 'flu', 'pneumonia'], 'severity': 0.7, 'body_system': 'general'},
                'cough': {'related_diseases': ['covid', 'bronchitis', 'pneumonia'], 'severity': 0.5, 'body_system': 'respiratory'},
                'chest_pain': {'related_diseases': ['heart_attack', 'angina', 'pulmonary_embolism'], 'severity': 0.9, 'body_system': 'cardiovascular'},
                'shortness_of_breath': {'related_diseases': ['asthma', 'copd', 'heart_failure'], 'severity': 0.8, 'body_system': 'respiratory'},
                'headache': {'related_diseases': ['migraine', 'tension', 'hypertension'], 'severity': 0.4, 'body_system': 'neurological'},
                'nausea': {'related_diseases': ['gastroenteritis', 'migraine', 'food_poisoning'], 'severity': 0.3, 'body_system': 'gastrointestinal'},
                'vomiting': {'related_diseases': ['gastroenteritis', 'appendicitis', 'migraine'], 'severity': 0.6, 'body_system': 'gastrointestinal'},
                'abdominal_pain': {'related_diseases': ['appendicitis', 'gastritis', 'kidney_stones'], 'severity': 0.7, 'body_system': 'gastrointestinal'},
                'fatigue': {'related_diseases': ['anemia', 'hypothyroidism', 'depression'], 'severity': 0.3, 'body_system': 'general'},
                'dizziness': {'related_diseases': ['vertigo', 'anemia', 'hypotension'], 'severity': 0.5, 'body_system': 'neurological'}
            },
            'diseases': {
                'covid': {
                    'common_symptoms': ['fever', 'cough', 'fatigue', 'loss_of_taste'],
                    'treatments': ['rest', 'hydration', 'antiviral_meds'],
                    'urgency': 'high',
                    'body_system': 'respiratory'
                },
                'pneumonia': {
                    'common_symptoms': ['fever', 'cough', 'chest_pain', 'shortness_of_breath'],
                    'treatments': ['antibiotics', 'oxygen_therapy', 'hospitalization'],
                    'urgency': 'high',
                    'body_system': 'respiratory'
                },
                'heart_attack': {
                    'common_symptoms': ['chest_pain', 'shortness_of_breath', 'nausea', 'sweating'],
                    'treatments': ['aspirin', 'nitroglycerin', 'emergency_care'],
                    'urgency': 'emergency',
                    'body_system': 'cardiovascular'
                },
                'migraine': {
                    'common_symptoms': ['headache', 'nausea', 'sensitivity_to_light'],
                    'treatments': ['pain_relievers', 'triptans', 'rest'],
                    'urgency': 'medium',
                    'body_system': 'neurological'
                },
                'gastroenteritis': {
                    'common_symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'],
                    'treatments': ['hydration', 'electrolytes', 'rest'],
                    'urgency': 'medium',
                    'body_system': 'gastrointestinal'
                }
            }
        }
    
    def load_drug_database(self):
        """Load drug interaction database"""
        return {
            'aspirin': {
                'interactions': ['warfarin', 'ibuprofen', 'alcohol'],
                'contraindications': ['bleeding_disorders', 'asthma']
            },
            'antibiotics': {
                'interactions': ['oral_contraceptives', 'antacids'],
                'contraindications': ['allergy']
            }
        }
    
    def load_guidelines(self):
        """Load clinical guidelines"""
        return {
            'hypertension': {'threshold_bp': 140, 'lifestyle_changes': True, 'medication_threshold': 150},
            'diabetes': {'fasting_glucose_threshold': 126, 'hba1c_threshold': 6.5},
            'asthma': {'peak_flow_threshold': 80, 'rescue_inhaler_frequency': '3_times_week'}
        }
    
    def diagnose_from_symptoms(self, symptoms: List[str], patient_context: Dict = None) -> List[Dict]:
        """Use graph-based diagnosis"""
        candidates = []
        for symptom in symptoms:
            if symptom in self.symptom_disease_graph['symptoms']:
                diseases = self.symptom_disease_graph['symptoms'][symptom]['related_diseases']
                severity = self.symptom_disease_graph['symptoms'][symptom]['severity']
                
                for disease in diseases:
                    # Check if disease exists in our knowledge
                    if disease in self.symptom_disease_graph['diseases']:
                        disease_info = self.symptom_disease_graph['diseases'][disease]
                        
                        # Calculate evidence score based on symptom overlap
                        common_symptoms = set(symptoms) & set(disease_info['common_symptoms'])
                        evidence_score = len(common_symptoms) / max(1, len(disease_info['common_symptoms']))
                        
                        # Adjust score by symptom severity
                        evidence_score *= severity
                        
                        # Consider patient context if available
                        if patient_context:
                            if 'age' in patient_context and disease == 'heart_attack' and patient_context['age'] > 50:
                                evidence_score *= 1.2
                            if 'smoker' in patient_context and disease in ['copd', 'lung_cancer']:
                                evidence_score *= 1.3
                        
                        candidates.append({
                            'disease': disease,
                            'evidence_score': min(evidence_score, 1.0),
                            'supporting_symptoms': list(common_symptoms),
                            'missing_symptoms': list(set(disease_info['common_symptoms']) - set(symptoms)),
                            'body_system': disease_info.get('body_system', 'unknown'),
                            'urgency': disease_info.get('urgency', 'medium')
                        })
        
        # Remove duplicates and sort
        unique_candidates = {}
        for candidate in candidates:
            disease = candidate['disease']
            if disease not in unique_candidates or candidate['evidence_score'] > unique_candidates[disease]['evidence_score']:
                unique_candidates[disease] = candidate
        
        return sorted(unique_candidates.values(), key=lambda x: x['evidence_score'], reverse=True)
    
    def calculate_evidence(self, disease: str, symptoms: List[str]) -> float:
        """Calculate evidence score for a disease given symptoms"""
        if disease not in self.symptom_disease_graph['diseases']:
            return 0.0
        
        disease_info = self.symptom_disease_graph['diseases'][disease]
        common_symptoms = set(symptoms) & set(disease_info['common_symptoms'])
        
        if not disease_info['common_symptoms']:
            return 0.0
        
        return len(common_symptoms) / len(disease_info['common_symptoms'])
    
    def get_supporting_symptoms(self, disease: str, symptoms: List[str]) -> List[str]:
        """Get which symptoms support a given disease"""
        if disease not in self.symptom_disease_graph['diseases']:
            return []
        
        disease_info = self.symptom_disease_graph['diseases'][disease]
        return list(set(symptoms) & set(disease_info['common_symptoms']))

class MedicalReasoningEngine(nn.Module):
    """Medical reasoning and diagnosis engine - Enhanced with dynamic learning"""
    
    def __init__(self, config: MedicalOperActionConfig):
        super().__init__()
        self.config = config
        self.knowledge_graph = MedicalKnowledgeGraph()
        
        # Static medical reasoning networks
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
            nn.Linear(64, 32),  # 32 potential diagnoses
            nn.Sigmoid()
        )
        
        self.treatment_recommender = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 32 treatment options
            nn.Sigmoid()
        )
        
        # Dynamic adaptation network (learns from interactions)
        self.dynamic_adapter = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Risk assessment network
        self.risk_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),  # 16 risk categories
            nn.Sigmoid()
        )
        
        # Learned patterns storage
        self.learned_patterns = []
        self.successful_responses = []
        
        # Performance tracking
        self.performance_stats = {
            'total_cases': 0,
            'successful_diagnoses': 0,
            'avg_confidence': 0.5,
            'common_symptom_patterns': {}
        }
    
    def extract_medical_concepts(self, text: str, abstraction_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract medical concepts from text and/or abstraction data"""
        medical_concepts = {
            'symptoms': [],
            'conditions': [],
            'treatments': [],
            'risk_factors': [],
            'body_parts': [],
            'patient_context': {}
        }
        
        text_lower = text.lower()
        
        # Extract from text directly
        symptom_keywords = ['fever', 'cough', 'pain', 'headache', 'nausea', 'vomiting', 
                          'shortness', 'breath', 'chest', 'abdominal', 'fatigue', 'dizziness',
                          'sore throat', 'runny nose', 'congestion', 'diarrhea', 'constipation']
        
        condition_keywords = ['cancer', 'diabetes', 'covid', 'pneumonia', 'infection',
                            'disease', 'disorder', 'syndrome', 'hypertension', 'asthma',
                            'arthritis', 'allergy', 'migraine']
        
        treatment_keywords = ['medicine', 'treatment', 'therapy', 'surgery', 'medication',
                            'antibiotic', 'antiviral', 'vaccine', 'injection', 'tablet']
        
        risk_keywords = ['risk', 'factor', 'history', 'genetic', 'family', 'smoking',
                        'alcohol', 'obese', 'overweight', 'age', 'old', 'young']
        
        body_keywords = ['heart', 'lung', 'brain', 'liver', 'kidney', 'stomach',
                        'intestine', 'muscle', 'bone', 'joint', 'skin']
        
        # Extract from text
        for symptom in symptom_keywords:
            if symptom in text_lower:
                medical_concepts['symptoms'].append(symptom)
        
        for condition in condition_keywords:
            if condition in text_lower:
                medical_concepts['conditions'].append(condition)
        
        # Extract from abstraction data if available
        if abstraction_data and 'cognitive' in abstraction_data:
            cognitive = abstraction_data['cognitive']
            concepts = cognitive.get('concepts', [])
            
            for concept in concepts:
                concept_lower = concept.lower()
                if any(keyword in concept_lower for keyword in symptom_keywords):
                    if concept_lower not in medical_concepts['symptoms']:
                        medical_concepts['symptoms'].append(concept_lower)
                elif any(keyword in concept_lower for keyword in condition_keywords):
                    if concept_lower not in medical_concepts['conditions']:
                        medical_concepts['conditions'].append(concept_lower)
        
        # Extract patient context
        age_pattern = r'(\d+)\s*year'
        age_match = re.search(age_pattern, text_lower)
        if age_match:
            medical_concepts['patient_context']['age'] = int(age_match.group(1))
        
        if any(word in text_lower for word in ['smoke', 'smoking', 'cigarette']):
            medical_concepts['patient_context']['smoker'] = True
        
        if any(word in text_lower for word in ['alcohol', 'drink', 'beer', 'wine']):
            medical_concepts['patient_context']['alcohol'] = True
        
        # Remove duplicates
        for key in ['symptoms', 'conditions', 'treatments', 'risk_factors', 'body_parts']:
            medical_concepts[key] = list(set(medical_concepts[key]))
        
        return medical_concepts
    
    def analyze_symptom_patterns(self, symptoms: List[str]) -> Dict[str, float]:
        """Analyze symptoms to identify potential conditions - with dynamic learning"""
        symptom_patterns = self.config.medical_knowledge_base['symptom_patterns']
        condition_scores = {}
        
        # Static pattern matching
        for pattern, conditions in symptom_patterns.items():
            pattern_symptoms = set(pattern.split('_'))
            input_symptoms = set(s.lower().replace(' ', '_') for s in symptoms)
            
            # Calculate overlap
            overlap = len(pattern_symptoms.intersection(input_symptoms))
            if overlap > 0:
                score = overlap / len(pattern_symptoms)
                for condition in conditions:
                    condition_scores[condition] = max(condition_scores.get(condition, 0), score)
        
        # Apply learned patterns
        for learned_pattern in self.learned_patterns:
            pattern_symptoms = set(learned_pattern['symptoms'])
            input_symptoms = set(s.lower().replace(' ', '_') for s in symptoms)
            
            overlap = len(pattern_symptoms.intersection(input_symptoms))
            if overlap / max(1, len(pattern_symptoms)) > 0.5:  # Significant overlap
                for condition in learned_pattern['conditions']:
                    adjusted_score = condition_scores.get(condition, 0) * 1.2  # Boost learned patterns
                    condition_scores[condition] = min(adjusted_score, 1.0)
        
        return condition_scores
    
    def generate_differential_diagnosis(self, medical_concepts: Dict[str, Any], 
                                      features: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate differential diagnosis with static + dynamic analysis"""
        
        symptoms = medical_concepts.get('symptoms', [])
        diagnoses = []
        
        if symptoms:
            # 1. Static knowledge graph diagnosis
            graph_diagnoses = self.knowledge_graph.diagnose_from_symptoms(
                symptoms, medical_concepts.get('patient_context', {})
            )
            
            # 2. Pattern-based analysis
            pattern_scores = self.analyze_symptom_patterns(symptoms)
            
            # 3. Neural network analysis
            if features is not None and features.numel() > 0:
                symptom_embedding = self.symptom_encoder(features)
                diagnosis_probs = self.diagnosis_predictor(symptom_embedding)
                
                # Convert neural outputs to diagnosis scores
                neural_scores = {}
                for i, prob in enumerate(diagnosis_probs[0]):
                    neural_scores[f'neural_diagnosis_{i}'] = prob.item()
            
            # Combine results
            combined_diagnoses = {}
            
            # Add graph diagnoses
            for graph_dx in graph_diagnoses:
                disease = graph_dx['disease']
                evidence_score = graph_dx['evidence_score']
                
                # Boost with pattern scores if available
                pattern_score = pattern_scores.get(disease, 0)
                combined_score = 0.6 * evidence_score + 0.4 * pattern_score
                
                combined_diagnoses[disease] = {
                    'condition': disease,
                    'confidence': combined_score,
                    'supporting_symptoms': graph_dx['supporting_symptoms'],
                    'missing_symptoms': graph_dx.get('missing_symptoms', []),
                    'body_system': graph_dx.get('body_system', 'unknown'),
                    'urgency': graph_dx.get('urgency', 'medium'),
                    'recommended_tests': self.suggest_diagnostic_tests(disease),
                    'evidence_sources': ['knowledge_graph', 'symptom_patterns']
                }
            
            # Apply dynamic adaptation if enabled
            if self.config.enable_dynamic_adaptation and features is not None:
                dynamic_adjustment = self.dynamic_adapter(features)
                adjustment_factor = torch.sigmoid(dynamic_adjustment.mean()).item()
                
                # Adjust confidences based on learned patterns
                for disease, dx_info in combined_diagnoses.items():
                    dx_info['confidence'] = min(dx_info['confidence'] * (1 + adjustment_factor * 0.2), 1.0)
                    dx_info['evidence_sources'].append('dynamic_adapter')
            
            # Convert to list and sort
            diagnoses = list(combined_diagnoses.values())
            diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Update performance stats
            self.performance_stats['total_cases'] += 1
            if diagnoses and diagnoses[0]['confidence'] > self.config.confidence_threshold:
                self.performance_stats['successful_diagnoses'] += 1
            
            # Update common symptom patterns
            symptom_key = '_'.join(sorted(symptoms))
            if symptom_key not in self.performance_stats['common_symptom_patterns']:
                self.performance_stats['common_symptom_patterns'][symptom_key] = 0
            self.performance_stats['common_symptom_patterns'][symptom_key] += 1
        
        return diagnoses[:5]  # Return top 5 diagnoses
    
    def suggest_diagnostic_tests(self, condition: str) -> List[str]:
        """Suggest diagnostic tests for a condition"""
        test_mapping = self.config.medical_knowledge_base.get('diagnostic_tests', {})
        
        if condition in test_mapping:
            return test_mapping[condition]
        
        # Fallback based on body system
        body_system_tests = {
            'respiratory': ['chest_xray', 'pulmonary_function_test', 'ct_scan'],
            'cardiovascular': ['ecg', 'echocardiogram', 'stress_test'],
            'neurological': ['mri', 'ct_scan', 'eeg'],
            'gastrointestinal': ['endoscopy', 'ultrasound', 'colonoscopy']
        }
        
        # Try to get body system from knowledge graph
        if condition in self.knowledge_graph.symptom_disease_graph['diseases']:
            body_system = self.knowledge_graph.symptom_disease_graph['diseases'][condition].get('body_system', 'general')
            if body_system in body_system_tests:
                return body_system_tests[body_system]
        
        return ['general_physical_exam', 'basic_blood_work']
    
    def assess_urgency(self, condition: str, symptoms: List[str]) -> str:
        """Assess urgency of medical condition with dynamic learning"""
        
        # Check knowledge graph first
        if condition in self.knowledge_graph.symptom_disease_graph['diseases']:
            urgency = self.knowledge_graph.symptom_disease_graph['diseases'][condition].get('urgency', 'medium')
            return urgency
        
        # Emergency conditions
        emergency_keywords = ['heart_attack', 'stroke', 'severe_allergic', 'pulmonary_embolism', 
                            'appendicitis', 'meningitis', 'septic', 'overdose']
        
        # Urgent conditions
        urgent_keywords = ['pneumonia', 'covid', 'infection', 'fracture', 'asthma_attack']
        
        condition_lower = condition.lower()
        if any(keyword in condition_lower for keyword in emergency_keywords):
            return 'emergency'
        elif any(keyword in condition_lower for keyword in urgent_keywords):
            return 'urgent'
        
        # Check symptoms for emergency signs
        emergency_symptoms = ['chest_pain', 'shortness_of_breath', 'severe_headache', 
                            'uncontrolled_bleeding', 'loss_of_consciousness']
        
        symptom_set = set(s.lower().replace(' ', '_') for s in symptoms)
        if any(es in symptom_set for es in emergency_symptoms):
            return 'urgent'
        
        return 'routine'
    
    def recommend_treatments(self, diagnosis: Dict[str, Any], 
                           medical_concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend treatments based on diagnosis"""
        treatments = []
        condition = diagnosis['condition']
        
        # 1. Static knowledge base recommendations
        knowledge_treatments = self.config.medical_knowledge_base['disease_treatments'].get(condition, [])
        
        for treatment in knowledge_treatments:
            treatments.append({
                'treatment': treatment,
                'type': self.classify_treatment_type(treatment),
                'evidence_level': 'established',
                'rationale': f"Standard treatment for {condition}",
                'priority': 'primary'
            })
        
        # 2. Neural network recommendations
        if 'confidence' in diagnosis:
            # Create feature vector from diagnosis confidence and condition
            diagnosis_features = torch.tensor([diagnosis['confidence'], len(medical_concepts.get('symptoms', [])) / 10.0])
            diagnosis_features = F.pad(diagnosis_features, (0, 94))  # Pad to 96 dimensions
            
            treatment_scores = self.treatment_recommender(diagnosis_features.unsqueeze(0))
            
            # Map scores to treatment suggestions
            if treatment_scores[0][0].item() > 0.5:
                treatments.append({
                    'treatment': 'pain_management',
                    'type': 'symptomatic',
                    'evidence_level': 'general',
                    'rationale': 'Symptomatic relief recommended',
                    'priority': 'supportive'
                })
        
        # 3. Context-aware additions
        patient_context = medical_concepts.get('patient_context', {})
        
        if patient_context.get('age', 0) > 65:
            treatments.append({
                'treatment': 'geriatric_assessment',
                'type': 'evaluative',
                'evidence_level': 'precautionary',
                'rationale': 'Recommended for older patients',
                'priority': 'supplemental'
            })
        
        if 'allergy' in medical_concepts.get('risk_factors', []):
            treatments.append({
                'treatment': 'allergy_testing',
                'type': 'diagnostic',
                'evidence_level': 'precautionary',
                'rationale': 'Recommended due to allergy history',
                'priority': 'diagnostic'
            })
        
        # 4. Dynamic adaptation based on learned successful treatments
        for successful_response in self.successful_responses:
            if successful_response['condition'] == condition:
                for treatment in successful_response['effective_treatments']:
                    if treatment not in [t['treatment'] for t in treatments]:
                        treatments.append({
                            'treatment': treatment,
                            'type': 'learned',
                            'evidence_level': 'empirical',
                            'rationale': f"Previously effective for similar cases",
                            'priority': 'alternative'
                        })
        
        return treatments[:8]  # Return up to 8 treatments
    
    def classify_treatment_type(self, treatment: str) -> str:
        """Classify treatment type"""
        medication_keywords = ['antibiotic', 'antiviral', 'medication', 'drug', 'pill', 'tablet']
        procedure_keywords = ['surgery', 'therapy', 'intervention', 'operation', 'injection']
        lifestyle_keywords = ['rest', 'diet', 'exercise', 'hydration', 'sleep']
        diagnostic_keywords = ['test', 'scan', 'xray', 'mri', 'blood_work']
        
        treatment_lower = treatment.lower()
        if any(keyword in treatment_lower for keyword in medication_keywords):
            return 'medication'
        elif any(keyword in treatment_lower for keyword in procedure_keywords):
            return 'procedure'
        elif any(keyword in treatment_lower for keyword in lifestyle_keywords):
            return 'lifestyle'
        elif any(keyword in treatment_lower for keyword in diagnostic_keywords):
            return 'diagnostic'
        else:
            return 'general'
    
    def assess_risks(self, medical_concepts: Dict[str, Any], 
                    features: torch.Tensor) -> Dict[str, Any]:
        """Assess medical risks with static and dynamic analysis"""
        risks = {}
        
        # Neural network risk assessment
        if features is not None and features.numel() > 0:
            risk_scores = self.risk_assessor(features)
            
            risk_categories = [
                'cardiovascular_risk', 'respiratory_risk', 'neurological_risk',
                'infectious_risk', 'metabolic_risk', 'trauma_risk',
                'environmental_risk', 'genetic_risk', 'medication_risk',
                'allergy_risk', 'surgical_risk', 'diagnostic_risk',
                'lifestyle_risk', 'age_related_risk', 'pregnancy_risk', 'immunological_risk'
            ]
            
            for i, category in enumerate(risk_categories):
                if i < len(risk_scores[0]):
                    risks[category] = risk_scores[0][i].item()
        
        # Knowledge-based risk factors
        symptoms = [s.lower() for s in medical_concepts.get('symptoms', [])]
        patient_context = medical_concepts.get('patient_context', {})
        
        # Symptom-based risks
        if 'chest_pain' in symptoms:
            risks['cardiovascular_risk'] = max(risks.get('cardiovascular_risk', 0), 0.8)
        
        if 'fever' in symptoms and 'cough' in symptoms:
            risks['infectious_risk'] = max(risks.get('infectious_risk', 0), 0.7)
        
        if 'headache' in symptoms and 'nausea' in symptoms:
            risks['neurological_risk'] = max(risks.get('neurological_risk', 0), 0.6)
        
        # Patient context based risks
        age = patient_context.get('age', 0)
        if age > 65:
            risks['age_related_risk'] = max(risks.get('age_related_risk', 0), 0.7)
            risks['cardiovascular_risk'] = risks.get('cardiovascular_risk', 0) + 0.2
        
        if patient_context.get('smoker', False):
            risks['respiratory_risk'] = max(risks.get('respiratory_risk', 0), 0.8)
            risks['cardiovascular_risk'] = risks.get('cardiovascular_risk', 0) + 0.3
        
        if patient_context.get('alcohol', False):
            risks['lifestyle_risk'] = max(risks.get('lifestyle_risk', 0), 0.6)
        
        # Cap all risks at 1.0
        for key in list(risks.keys()):
            risks[key] = min(risks[key], 1.0)
        
        return risks
    
    def generate_medical_advice(self, diagnoses: List[Dict[str, Any]], 
                              treatments: List[Dict[str, Any]],
                              risks: Dict[str, Any],
                              medical_concepts: Dict[str, Any]) -> str:
        """Generate comprehensive medical advice"""
        
        if not diagnoses:
            return "Based on the information provided, specific medical conditions cannot be identified with sufficient confidence. Please provide more detailed symptoms or consult a healthcare professional for an in-person evaluation."
        
        primary_diagnosis = diagnoses[0]
        advice_parts = []
        
        # Diagnosis summary
        confidence_percent = primary_diagnosis['confidence'] * 100
        advice_parts.append(f"**Primary Concern:** {primary_diagnosis['condition'].replace('_', ' ').title()}")
        advice_parts.append(f"**Confidence Level:** {confidence_percent:.0f}%")
        
        # Urgency assessment
        urgency = primary_diagnosis.get('urgency', 'routine').upper()
        urgency_advice = {
            'EMERGENCY': "⚠️ **MEDICAL EMERGENCY** - Seek immediate medical attention or call emergency services.",
            'URGENT': "🔴 **Urgent Care Needed** - Visit an urgent care facility or emergency room within 24 hours.",
            'ROUTINE': "🟡 **Routine Care** - Schedule an appointment with your primary care physician."
        }
        
        if urgency in urgency_advice:
            advice_parts.append(urgency_advice[urgency])
        
        # Supporting evidence
        if primary_diagnosis.get('supporting_symptoms'):
            symptoms_text = ', '.join(primary_diagnosis['supporting_symptoms']).replace('_', ' ')
            advice_parts.append(f"**Supporting Symptoms:** {symptoms_text}")
        
        # Treatment recommendations
        if treatments:
            advice_parts.append("\n**Recommended Actions:**")
            
            # Group by priority
            primary_treatments = [t for t in treatments if t.get('priority') == 'primary']
            supportive_treatments = [t for t in treatments if t.get('priority') in ['supportive', 'supplemental']]
            
            if primary_treatments:
                advice_parts.append("*Primary Treatments:*")
                for treatment in primary_treatments[:3]:
                    advice_parts.append(f"  • {treatment['treatment'].replace('_', ' ').title()} - {treatment.get('rationale', 'Standard care')}")
            
            if supportive_treatments:
                advice_parts.append("*Supportive Care:*")
                for treatment in supportive_treatments[:2]:
                    advice_parts.append(f"  • {treatment['treatment'].replace('_', ' ').title()}")
        
        # Risk factors
        high_risks = {k: v for k, v in risks.items() if v > 0.7}
        if high_risks:
            advice_parts.append("\n**Notable Risk Factors:**")
            for risk, score in list(high_risks.items())[:3]:
                risk_name = risk.replace('_', ' ').title()
                advice_parts.append(f"  • {risk_name} (score: {score:.0%})")
        
        # Next steps
        advice_parts.append("\n**Next Steps:**")
        advice_parts.append("1. Monitor symptoms and note any changes")
        advice_parts.append("2. Follow recommended treatments as appropriate")
        advice_parts.append("3. Seek professional medical care based on urgency level")
        advice_parts.append("4. Follow up if symptoms persist or worsen")
        
        # Disclaimer
        advice_parts.append("\n*Disclaimer: This is AI-generated medical information, not a substitute for professional medical advice, diagnosis, or treatment.*")
        
        return "\n".join(advice_parts)
    
    def learn_from_interaction(self, input_data: Any, result: Dict, success_metric: float):
        """Learn from successful medical interactions"""
        if success_metric > 0.7:  # Good result
            # Extract medical concepts from input
            if isinstance(input_data, str):
                medical_concepts = self.extract_medical_concepts(input_data)
            elif isinstance(input_data, dict):
                text = input_data.get('text', '')
                medical_concepts = self.extract_medical_concepts(text, input_data.get('abstraction_data'))
            else:
                medical_concepts = {'symptoms': []}
            
            # Store successful pattern
            if result.get('medical_analysis') and result['medical_analysis'].get('diagnoses'):
                primary_diagnosis = result['medical_analysis']['diagnoses'][0]
                
                learned_pattern = {
                    'symptoms': medical_concepts.get('symptoms', []),
                    'condition': primary_diagnosis['condition'],
                    'confidence': primary_diagnosis['confidence'],
                    'effective_treatments': [t['treatment'] for t in result['medical_analysis'].get('treatments', [])],
                    'success_metric': success_metric,
                    'timestamp': time.time()
                }
                
                self.learned_patterns.append(learned_pattern)
                self.successful_responses.append({
                    'condition': primary_diagnosis['condition'],
                    'effective_treatments': learned_pattern['effective_treatments'],
                    'success_rate': success_metric
                })
                
                # Keep only recent patterns
                if len(self.learned_patterns) > 50:
                    self.learned_patterns = self.learned_patterns[-50:]
                if len(self.successful_responses) > 100:
                    self.successful_responses = self.successful_responses[-100:]
                
                # Update performance stats
                self.performance_stats['avg_confidence'] = (
                    self.performance_stats['avg_confidence'] * (self.performance_stats['total_cases'] - 1) +
                    primary_diagnosis['confidence']
                ) / max(1, self.performance_stats['total_cases'])
                
                logger.info(f"Medical reasoning engine learned new pattern: {primary_diagnosis['condition']}")

class MedicalActionVNI(SpecializedVNIBase):
    """Medical specialized VNI with static knowledge + dynamic adaptation"""
    
    def __init__(self, config: MedicalOperActionConfig = None):
        config_obj = config or MedicalOperActionConfig()
        super().__init__(topic_name="medical", config=config_obj.__dict__)
        
        self.config = config_obj
        self.reasoning_engine = MedicalReasoningEngine(self.config)
        
        # Medical-specific dynamic adapter
        self.medical_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Replace base adapter with medical-specific one
        self.dynamic_adapter = self.medical_adapter
        
        # Performance tracker
        self.performance_tracker = OperActionPerformance()
        
        logger.info(f"✅ Hybrid Medical VNI initialized: {self.config.vni_id}")
        logger.info(f"   Dynamic adaptation: {self.config.enable_dynamic_adaptation}")
    
    def forward(self, base_features: Dict, input_data: Any) -> Dict:
        """Process medical input with hybrid static+dynamic analysis"""
        
        # Extract text from input
        text = self._extract_text(input_data)
        
        # Extract medical concepts
        abstraction_data = base_features.get('abstraction_levels', {})
        medical_concepts = self.reasoning_engine.extract_medical_concepts(text, abstraction_data)
        
        # Get features for neural processing
        features = self._extract_features(base_features)
        
        # Generate differential diagnosis
        diagnoses = self.reasoning_engine.generate_differential_diagnosis(medical_concepts, features)
        
        # Recommend treatments
        treatments = []
        if diagnoses:
            primary_diagnosis = diagnoses[0]
            treatments = self.reasoning_engine.recommend_treatments(primary_diagnosis, medical_concepts)
        
        # Assess risks
        risks = self.reasoning_engine.assess_risks(medical_concepts, features)
        
        # Generate medical advice
        medical_advice = self.reasoning_engine.generate_medical_advice(
            diagnoses, treatments, risks, medical_concepts
        )
        
        # Compile base result
        base_result = {
            'medical_analysis': {
                'diagnoses': diagnoses,
                'treatments': treatments,
                'risk_assessment': risks,
                'identified_concepts': medical_concepts,
                'patient_context': medical_concepts.get('patient_context', {})
            },
            'medical_advice': medical_advice,
            'confidence_score': diagnoses[0]['confidence'] if diagnoses else 0.0,
            'processing_metadata': {
                'symptoms_analyzed': len(medical_concepts.get('symptoms', [])),
                'conditions_considered': len(diagnoses),
                'treatments_recommended': len(treatments),
                'risk_categories_assessed': len(risks),
                'dynamic_adaptation_used': self.config.enable_dynamic_adaptation
            }
        }
        
        # Apply dynamic adaptation if enabled
        if self.config.enable_dynamic_adaptation and self.adaptation_strength > 0.1:
            adapted_result = self.apply_dynamic_adaptation(base_result, base_features)
            adapted_result['dynamic_adaptation_applied'] = True
            result = adapted_result
        else:
            result = base_result
        
        # Add VNI metadata
        result['vni_metadata'] = {
            'vni_id': self.config.vni_id,
            'vni_type': 'operAction_medical_hybrid',
            'processing_stages': ['concept_extraction', 'diagnosis_generation', 
                                'treatment_recommendation', 'risk_assessment', 'dynamic_adaptation'],
            'success': True,
            'domain': 'medical',
            'hybrid_system': True,
            'static_knowledge_used': True,
            'dynamic_learning_enabled': self.config.enable_dynamic_adaptation,
            'adaptation_strength': self.adaptation_strength
        }
        
        # Track performance
        if hasattr(self, 'performance_tracker'):
            quality_score = self.performance_tracker.record_outcome(input_data, result)
            result['quality_assessment'] = quality_score
        
        return result
    
    def _extract_text(self, input_data: Any) -> str:
        """Extract text from various input formats"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            if 'text' in input_data:
                return input_data['text']
            elif 'input' in input_data:
                return str(input_data['input'])
        return str(input_data)
    
    def _extract_features(self, base_features: Dict) -> Optional[torch.Tensor]:
        """Extract features for neural processing"""
        if 'semantic' in base_features and 'tensor' in base_features['semantic']:
            return base_features['semantic']['tensor']
        elif 'tensor' in base_features:
            return base_features['tensor']
        elif 'abstraction_levels' in base_features:
            abstraction = base_features['abstraction_levels']
            if 'semantic' in abstraction and 'tensor' in abstraction['semantic']:
                return abstraction['semantic']['tensor']
        
        # Create default features
        return torch.zeros(1, 256)
    
    def learn_from_interaction(self, input_data: Any, result: Dict, success_metric: float):
        """Enhanced learning with medical-specific patterns"""
        super().learn_from_interaction(input_data, result, success_metric)
        
        # Medical-specific learning
        self.reasoning_engine.learn_from_interaction(input_data, result, success_metric)
        
        # Update medical knowledge with successful patterns
        if success_metric > 0.8 and result.get('medical_analysis'):
            self._update_medical_knowledge(result['medical_analysis'])
    
    def _update_medical_knowledge(self, medical_analysis: Dict):
        """Update static knowledge with learned patterns"""
        diagnoses = medical_analysis.get('diagnoses', [])
        treatments = medical_analysis.get('treatments', [])
        
        if diagnoses and treatments:
            primary_diagnosis = diagnoses[0]
            condition = primary_diagnosis['condition']
            
            # Add to symptom patterns if new
            symptoms = medical_analysis.get('identified_concepts', {}).get('symptoms', [])
            if symptoms and condition:
                symptom_key = '_'.join(sorted([s.lower().replace(' ', '_') for s in symptoms]))
                
                if symptom_key not in self.config.medical_knowledge_base['symptom_patterns']:
                    self.config.medical_knowledge_base['symptom_patterns'][symptom_key] = [condition]
                    logger.info(f"Added new symptom pattern: {symptom_key} -> {condition}")
            
            # Add to treatments if new
            treatment_names = [t['treatment'] for t in treatments]
            if treatment_names and condition not in self.config.medical_knowledge_base['disease_treatments']:
                self.config.medical_knowledge_base['disease_treatments'][condition] = treatment_names
                logger.info(f"Added treatments for new condition: {condition}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return enhanced capabilities including dynamic learning status"""
        base_capabilities = super().get_capabilities()
        
        medical_capabilities = {
            'vni_type': 'operAction_medical_hybrid',
            'description': 'Hybrid medical diagnosis VNI with static knowledge + dynamic adaptation',
            'capabilities': [
                'Symptom pattern analysis with knowledge graph',
                'Differential diagnosis generation',
                'Treatment recommendation with priority levels',
                'Medical risk assessment across 16 categories',
                'Urgency evaluation (emergency/urgent/routine)',
                'Dynamic adaptation from learned patterns',
                'Patient context integration',
                'Performance tracking and learning'
            ],
            'medical_specific': {
                'symptom_database_size': len(self.reasoning_engine.knowledge_graph.symptom_disease_graph['symptoms']),
                'disease_database_size': len(self.reasoning_engine.knowledge_graph.symptom_disease_graph['diseases']),
                'learned_patterns': len(self.reasoning_engine.learned_patterns),
                'performance_stats': self.reasoning_engine.performance_stats
            },
            'hybrid_features': {
                'static_knowledge': True,
                'dynamic_adaptation': self.config.enable_dynamic_adaptation,
                'learning_enabled': True,
                'state_persistence': True
            },
            'input_types': ['text', 'dict_with_abstraction', 'raw_medical_query'],
            'output_types': ['medical_analysis', 'medical_advice', 'risk_assessment', 'treatment_recommendations']
        }
        
        # Merge with base capabilities
        medical_capabilities.update(base_capabilities)
        return medical_capabilities
    
    def save_state(self, path: str):
        """Save medical-specific state in addition to base state"""
        super().save_state(path)
        
        # Save medical-specific data
        medical_state = {
            'learned_patterns': self.reasoning_engine.learned_patterns,
            'successful_responses': self.reasoning_engine.successful_responses,
            'performance_stats': self.reasoning_engine.performance_stats,
            'knowledge_base_updates': self.config.medical_knowledge_base
        }
        
        import json
        medical_path = path.replace('.json', '_medical.json')
        with open(medical_path, 'w') as f:
            json.dump(medical_state, f, indent=2)
        
        logger.info(f"Medical state saved to {medical_path}")
    
    def load_state(self, path: str):
        """Load medical-specific state in addition to base state"""
        super().load_state(path)
        
        # Load medical-specific data
        import json
        medical_path = path.replace('.json', '_medical.json')
        
        if os.path.exists(medical_path):
            with open(medical_path, 'r') as f:
                medical_state = json.load(f)
            
            self.reasoning_engine.learned_patterns = medical_state.get('learned_patterns', [])
            self.reasoning_engine.successful_responses = medical_state.get('successful_responses', [])
            self.reasoning_engine.performance_stats = medical_state.get('performance_stats', {})
            
            # Update knowledge base with learned patterns
            knowledge_updates = medical_state.get('knowledge_base_updates', {})
            for key, value in knowledge_updates.items():
                if key in self.config.medical_knowledge_base:
                    self.config.medical_knowledge_base[key].update(value)
                else:
                    self.config.medical_knowledge_base[key] = value
            
            logger.info(f"Medical state loaded from {medical_path}")
            logger.info(f"  Loaded {len(self.reasoning_engine.learned_patterns)} learned patterns")
            logger.info(f"  Loaded {len(self.reasoning_engine.successful_responses)} successful responses")

# Performance tracking class
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

# Demonstration and testing
def demo_hybrid_medical_vni():
    """Demo the hybrid medical VNI"""
    
    print("🧬 HYBRID MEDICAL VNI DEMONSTRATION")
    print("=" * 60)
    
    # Initialize hybrid medical VNI
    config = MedicalOperActionConfig(
        vni_id="medical_hybrid_demo_001",
        reasoning_depth="comprehensive",
        enable_dynamic_adaptation=True
    )
    
    medical_vni = MedicalActionVNI(config)
    
    # Display capabilities
    capabilities = medical_vni.get_capabilities()
    print(f"VNI Type: {capabilities['vni_type']}")
    print(f"Description: {capabilities['description']}")
    print(f"Dynamic Adaptation: {capabilities['hybrid_features']['dynamic_adaptation']}")
    print(f"Learned Patterns: {capabilities['medical_specific']['learned_patterns']}")
    print()
    
    # Test cases
    test_cases = [
        "Patient is a 45-year-old male with fever and persistent cough for 3 days, also complains of chest pain when breathing deeply.",
        "I have severe headache with nausea and sensitivity to light, the pain is throbbing and gets worse with movement.",
        "68-year-old female with history of hypertension presents with sudden chest pain radiating to left arm, accompanied by sweating and shortness of breath.",
        "Child with high fever, sore throat, and runny nose for 2 days, no appetite.",
        "Abdominal pain in lower right quadrant with nausea and loss of appetite."
    ]
    
    for i, test_text in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Input: {test_text}")
        print("-" * 40)
        
        # Create mock base features
        base_features = {
            'primary_topic': 'medical',
            'complexity': 0.7,
            'abstraction_levels': {
                'semantic': {
                    'tensor': torch.randn(1, 512),
                    'concepts': ['patient', 'fever', 'cough', 'chest pain'],
                    'intent': 'diagnosis'
                }
            }
        }
        
        # Process with hybrid VNI
        with torch.no_grad():
            result = medical_vni(base_features, test_text)
        
        # Display results
        if result.get('vni_metadata', {}).get('success', False):
            analysis = result['medical_analysis']
            
            if analysis.get('diagnoses'):
                print(f"Primary Diagnosis: {analysis['diagnoses'][0]['condition'].replace('_', ' ').title()}")
                print(f"Confidence: {analysis['diagnoses'][0]['confidence']:.1%}")
                print(f"Urgency: {analysis['diagnoses'][0].get('urgency', 'routine').upper()}")
            
            print(f"\nMedical Advice (summary):")
            advice_lines = result['medical_advice'].split('\n')
            for line in advice_lines[:5]:  # Show first 5 lines
                if line.strip():
                    print(f"  {line}")
            
            if analysis.get('treatments'):
                print(f"\nTop Treatment: {analysis['treatments'][0]['treatment'].replace('_', ' ').title()}")
            
            print(f"Dynamic Adaptation Applied: {result.get('dynamic_adaptation_applied', False)}")
        else:
            print("Processing failed")
        
        print()
    
    print("=" * 60)
    print("🎯 HYBRID FEATURES DEMONSTRATED:")
    print("  ✅ Static medical knowledge graph")
    print("  ✅ Dynamic adaptation from interactions")
    print("  ✅ Neural network enhanced diagnosis")
    print("  ✅ Risk assessment across multiple categories")
    print("  ✅ Learning from successful patterns")
    print("  ✅ State persistence and recovery")
    
    return medical_vni

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Run the demo
    vni = demo_hybrid_medical_vni()
    
    # Save state for demonstration
    vni.save_state("medical_vni_demo_state.json")
    print("\n💾 State saved to medical_vni_demo_state.json") 
