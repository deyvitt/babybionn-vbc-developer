# _legal.py - HYBRID VERSION with static knowledge + dynamic adaptation
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from specialized_vni_base import SpecializedVNIBase
import re
import time

logger = logging.getLogger("operAction_legal_hybrid")

@dataclass
class LegalOperActionConfig:
    """Configuration for legal operAction VNI"""
    vni_id: str = "operAction_legal_hybrid"
    reasoning_depth: str = "comprehensive"
    legal_knowledge_base: Dict[str, Any] = None
    confidence_threshold: float = 0.6
    enable_dynamic_adaptation: bool = True
    
    def __post_init__(self):
        if self.legal_knowledge_base is None:
            self.legal_knowledge_base = {
                'legal_areas': {
                    'contract_law': {
                        'keywords': ['contract', 'agreement', 'breach', 'liability', 'enforcement'],
                        'common_issues': ['unclear_terms', 'unenforceable_clauses', 'ambiguous_obligations'],
                        'remedies': ['amendment', 'termination', 'damages_award']
                    },
                    'privacy_law': {
                        'keywords': ['privacy', 'data', 'gdpr', 'compliance', 'consent'],
                        'common_issues': ['data_breach', 'non_compliance', 'unauthorized_sharing'],
                        'remedies': ['compliance_audit', 'data_protection_measures', 'penalty_mitigation']
                    },
                    'intellectual_property': {
                        'keywords': ['copyright', 'patent', 'trademark', 'infringement', 'licensing'],
                        'common_issues': ['infringement', 'invalid_rights', 'licensing_disputes'],
                        'remedies': ['cease_and_desist', 'licensing_agreement', 'damages_recovery']
                    },
                    'employment_law': {
                        'keywords': ['employee', 'employer', 'termination', 'discrimination', 'harassment'],
                        'common_issues': ['wrongful_termination', 'discrimination_claims', 'wage_disputes'],
                        'remedies': ['settlement', 'reinstatement', 'compensation']
                    }
                },
                'compliance_frameworks': {
                    'gdpr': {
                        'requirements': ['data_protection', 'user_consent', 'privacy_by_design'],
                        'jurisdiction': 'eu',
                        'penalties': ['fines_up_to_4_percent']
                    },
                    'hipaa': {
                        'requirements': ['patient_privacy', 'data_security', 'medical_records_protection'],
                        'jurisdiction': 'us',
                        'penalties': ['civil_and_criminal_penalties']
                    },
                    'ccpa': {
                        'requirements': ['consumer_rights', 'data_transparency', 'opt_out_mechanisms'],
                        'jurisdiction': 'california',
                        'penalties': ['statutory_damages']
                    }
                },
                'risk_levels': {
                    'high': ['data_breach', 'regulatory_violation', 'major_contract_breach'],
                    'medium': ['compliance_gaps', 'contractual_ambiguities', 'potential_infringement'],
                    'low': ['procedural_issues', 'documentation_gaps', 'minor_compliance_items']
                }
            }

class LegalKnowledgeGraph:
    """Static legal knowledge graph"""
    def __init__(self):
        self.legal_ontology = self.build_legal_ontology()
        self.precedent_patterns = self.load_legal_precedents()
    
    def build_legal_ontology(self):
        """Build proper legal concept relationships"""
        return {
            'contract_law': {
                'concepts': ['offer', 'acceptance', 'consideration', 'breach', 'damages', 'remedies'],
                'relationships': {
                    'breach': ['leads_to', 'damages', 'termination'],
                    'offer': ['requires', 'acceptance', 'consideration'],
                    'damages': ['compensatory', 'punitive', 'liquidated']
                },
                'key_doctrines': ['freedom_of_contract', 'good_faith', 'unconscionability']
            },
            'privacy_law': {
                'concepts': ['consent', 'data_processing', 'right_to_be_forgotten', 'data_protection_officer'],
                'relationships': {
                    'data_processing': ['requires', 'legal_basis', 'purpose_limitation'],
                    'consent': ['must_be', 'informed', 'specific', 'unambiguous']
                },
                'key_doctrines': ['privacy_by_design', 'data_minimization', 'purpose_limitation']
            },
            'intellectual_property': {
                'concepts': ['copyright', 'patent', 'trademark', 'trade_secret', 'fair_use'],
                'relationships': {
                    'infringement': ['requires', 'unauthorized_use', 'commercial_impact'],
                    'licensing': ['grants', 'permission', 'subject_to', 'terms']
                },
                'key_doctrines': ['first_sale_doctrine', 'fair_use', 'exhaustion_of_rights']
            }
        }
    
    def load_legal_precedents(self):
        """Load patterns from legal precedents"""
        return {
            'contract_interpretation': {
                'patterns': ['strict_construction', 'plain_meaning', 'contra_proferentem'],
                'application': 'ambiguous_terms'
            },
            'data_breach_cases': {
                'patterns': ['failure_to_protect', 'lack_of_encryption', 'delayed_notification'],
                'outcomes': ['regulatory_fines', 'class_action_settlements']
            },
            'ip_infringement': {
                'patterns': ['substantial_similarity', 'commercial_use', 'market_impact'],
                'remedies': ['injunction', 'damages', 'profit_disgorgement']
            }
        }
    
    def find_relevant_precedents(self, legal_issue: str, context: Dict) -> List[Dict]:
        """Find relevant legal precedents for an issue"""
        precedents = []
        
        for category, data in self.precedent_patterns.items():
            if legal_issue in data.get('application', '') or any(pattern in legal_issue for pattern in data['patterns']):
                precedents.append({
                    'category': category,
                    'patterns': data['patterns'],
                    'typical_outcomes': data.get('outcomes', []),
                    'relevance_score': 0.7  # Base relevance
                })
        
        return precedents

class LegalReasoningEngine(nn.Module):
    """Legal reasoning engine with dynamic learning"""
    
    def __init__(self, config: LegalOperActionConfig):
        super().__init__()
        self.config = config
        self.knowledge_graph = LegalKnowledgeGraph()
        
        # Legal reasoning networks
        self.legal_concept_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        self.risk_assessor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        self.compliance_analyzer = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.Sigmoid()
        )
        
        self.remedy_generator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Softmax(dim=-1)
        )
        
        # Dynamic adaptation network
        self.dynamic_adapter = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Learned patterns
        self.learned_patterns = []
        self.successful_analyses = []
        
        # Performance tracking
        self.performance_stats = {
            'total_cases': 0,
            'high_risk_cases': 0,
            'avg_risk_score': 0.5,
            'common_issue_patterns': {}
        }
    
    def extract_legal_concepts(self, text: str, abstraction_data: Dict = None) -> Dict[str, Any]:
        """Extract legal concepts from text"""
        legal_concepts = {
            'legal_areas': [],
            'key_issues': [],
            'parties': [],
            'obligations': [],
            'risks': [],
            'jurisdiction_indicators': []
        }
        
        text_lower = text.lower()
        
        # Legal area detection
        for area, info in self.config.legal_knowledge_base['legal_areas'].items():
            if any(keyword in text_lower for keyword in info['keywords']):
                legal_concepts['legal_areas'].append({
                    'area': area,
                    'keywords_found': [k for k in info['keywords'] if k in text_lower],
                    'potential_issues': info['common_issues']
                })
        
        # Extract parties (simplified)
        party_patterns = [r'between (\w+) and (\w+)', r'(\w+) agrees to', r'(\w+) shall']
        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    legal_concepts['parties'].extend([m for m in match if len(m) > 3])
                else:
                    legal_concepts['parties'].append(match)
        
        # Extract obligations
        obligation_indicators = ['shall', 'must', 'will', 'agrees to', 'is obligated to', 'undertakes to']
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in obligation_indicators):
                legal_concepts['obligations'].append(sentence.strip())
        
        # Extract risks
        risk_indicators = ['risk', 'liability', 'penalty', 'fine', 'breach', 'violation', 'damages']
        for indicator in risk_indicators:
            if indicator in text_lower:
                legal_concepts['risks'].append(indicator)
        
        # Jurisdiction detection
        jurisdiction_indicators = {
            'eu': ['gdpr', 'european union', 'eu law', 'european commission'],
            'us': ['united states', 'federal', 'state law', 'california', 'new york'],
            'uk': ['united kingdom', 'uk law', 'english law', 'london']
        }
        
        for jurisdiction, indicators in jurisdiction_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                legal_concepts['jurisdiction_indicators'].append(jurisdiction)
        
        return legal_concepts
    
    def identify_legal_issues(self, legal_concepts: Dict, features: torch.Tensor) -> List[Dict]:
        """Identify legal issues with static + dynamic analysis"""
        issues = []
        
        # Static knowledge analysis
        for area_info in legal_concepts.get('legal_areas', []):
            area = area_info['area']
            area_data = self.config.legal_knowledge_base['legal_areas'].get(area, {})
            
            # Check for common issues in this area
            for potential_issue in area_data.get('common_issues', []):
                # Simple matching for now
                if any(keyword in str(legal_concepts).lower() for keyword in potential_issue.split('_')):
                    risk_score = self._calculate_risk_score(potential_issue, legal_concepts)
                    
                    # Apply dynamic adjustment
                    if self.config.enable_dynamic_adaptation and features is not None:
                        dynamic_adjustment = self.dynamic_adapter(features)
                        adjustment_factor = torch.sigmoid(dynamic_adjustment.mean()).item()
                        risk_score = min(risk_score * (1 + adjustment_factor * 0.3), 1.0)
                    
                    issue = {
                        'issue': potential_issue.replace('_', ' ').title(),
                        'area': area,
                        'risk_score': risk_score,
                        'description': f'Potential {potential_issue.replace("_", " ")} issue identified',
                        'urgency': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
                        'evidence_sources': ['knowledge_base']
                    }
                    
                    # Add relevant precedents
                    precedents = self.knowledge_graph.find_relevant_precedents(potential_issue, legal_concepts)
                    if precedents:
                        issue['relevant_precedents'] = precedents
                        issue['evidence_sources'].append('legal_precedents')
                    
                    issues.append(issue)
        
        # Sort by risk score
        issues.sort(key=lambda x: x['risk_score'], reverse=True)
        
        # Update stats
        self.performance_stats['total_cases'] += 1
        if issues and issues[0]['risk_score'] > 0.7:
            self.performance_stats['high_risk_cases'] += 1
        
        return issues[:5]  # Top 5 issues
    
    def _calculate_risk_score(self, issue: str, concepts: Dict) -> float:
        """Calculate risk score for a legal issue"""
        base_score = 0.5
        
        # Adjust based on issue type
        if issue in self.config.legal_knowledge_base['risk_levels']['high']:
            base_score = 0.8
        elif issue in self.config.legal_knowledge_base['risk_levels']['medium']:
            base_score = 0.6
        elif issue in self.config.legal_knowledge_base['risk_levels']['low']:
            base_score = 0.4
        
        # Adjust based on number of obligations
        obligation_count = len(concepts.get('obligations', []))
        base_score += min(obligation_count * 0.05, 0.2)
        
        # Adjust based on jurisdiction complexity
        jurisdiction_count = len(concepts.get('jurisdiction_indicators', []))
        if jurisdiction_count > 1:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def analyze_compliance(self, legal_concepts: Dict, issues: List[Dict], features: torch.Tensor) -> List[Dict]:
        """Analyze compliance requirements"""
        compliance_issues = []
        
        # Check for compliance framework indicators
        for framework, info in self.config.legal_knowledge_base['compliance_frameworks'].items():
            framework_lower = framework.lower()
            
            # Check if framework is mentioned
            if framework_lower in str(legal_concepts).lower():
                compliance_score = 0.5
                
                # Neural compliance analysis
                if features is not None:
                    compliance_output = self.compliance_analyzer(features[:96])
                    compliance_score = compliance_output.mean().item()
                
                compliance_issues.append({
                    'framework': framework.upper(),
                    'compliance_level': compliance_score,
                    'requirements': info['requirements'],
                    'jurisdiction': info['jurisdiction'],
                    'potential_penalties': info['penalties'],
                    'assessment': f'{framework.upper()} compliance considerations identified'
                })
        
        return compliance_issues
    
    def generate_remedies(self, issues: List[Dict], legal_concepts: Dict) -> List[Dict]:
        """Generate legal remedies"""
        remedies = []
        
        for issue in issues[:3]:  # Top 3 issues
            area_data = self.config.legal_knowledge_base['legal_areas'].get(issue['area'], {})
            area_remedies = area_data.get('remedies', ['Consult legal counsel'])
            
            remedy_scores = []
            if hasattr(self, 'remedy_generator'):
                # Generate remedy scores
                remedy_input = torch.tensor([issue['risk_score']] * 64)
                remedy_scores_tensor = self.remedy_generator(remedy_input.unsqueeze(0))
                remedy_scores = remedy_scores_tensor[0].tolist()
            
            # Create remedy recommendations
            issue_remedies = []
            for i, remedy in enumerate(area_remedies):
                confidence = remedy_scores[i] if i < len(remedy_scores) else 0.5
                
                issue_remedies.append({
                    'action': remedy,
                    'type': self._classify_remedy_type(remedy),
                    'confidence': confidence,
                    'priority': 'immediate' if issue['urgency'] == 'high' else 'short_term'
                })
            
            remedies.append({
                'issue': issue['issue'],
                'risk_score': issue['risk_score'],
                'recommended_actions': issue_remedies[:3],  # Top 3 actions
                'preventive_measures': [
                    'Regular legal compliance audits',
                    'Clear documentation practices',
                    'Stakeholder training programs'
                ]
            })
        
        return remedies
    
    def _classify_remedy_type(self, remedy: str) -> str:
        """Classify remedy type"""
        remedy_lower = remedy.lower()
        
        if any(word in remedy_lower for word in ['amendment', 'revision', 'modification']):
            return 'corrective'
        elif any(word in remedy_lower for word in ['termination', 'cancellation', 'rescission']):
            return 'termination'
        elif any(word in remedy_lower for word in ['damages', 'compensation', 'reimbursement']):
            return 'compensatory'
        elif any(word in remedy_lower for word in ['audit', 'review', 'assessment']):
            return 'evaluative'
        else:
            return 'general'
    
    def generate_legal_advice(self, issues: List[Dict], compliance_issues: List[Dict], 
                            remedies: List[Dict], legal_concepts: Dict) -> str:
        """Generate comprehensive legal advice"""
        
        if not issues:
            return "Based on the information provided, no significant legal issues were identified. However, general legal review is always recommended for formal agreements."
        
        primary_issue = issues[0]
        advice_parts = []
        
        # Summary
        advice_parts.append(f"**Primary Legal Concern:** {primary_issue['issue']}")
        advice_parts.append(f"**Risk Level:** {primary_issue['urgency'].upper()} (Score: {primary_issue['risk_score']:.1%})")
        advice_parts.append(f"**Legal Area:** {primary_issue['area'].replace('_', ' ').title()}")
        
        # Urgency guidance
        if primary_issue['urgency'] == 'high':
            advice_parts.append("\n⚠️ **HIGH URGENCY** - Immediate legal review recommended.")
        elif primary_issue['urgency'] == 'medium':
            advice_parts.append("\n🔶 **MEDIUM URGENCY** - Prompt legal evaluation advised.")
        
        # Compliance considerations
        if compliance_issues:
            advice_parts.append("\n**Compliance Frameworks Applicable:**")
            for comp in compliance_issues[:2]:
                status = "⚠️ Requires attention" if comp['compliance_level'] < 0.7 else "✅ Generally compliant"
                advice_parts.append(f"- {comp['framework']}: {status}")
        
        # Recommended actions
        if remedies:
            primary_remedy = remedies[0]
            advice_parts.append("\n**Recommended Actions:**")
            for action in primary_remedy['recommended_actions'][:2]:
                advice_parts.append(f"• {action['action']} ({action['type'].title()})")
        
        # Preventive measures
        advice_parts.append("\n**Preventive Measures:**")
        advice_parts.append("• Implement regular legal compliance reviews")
        advice_parts.append("• Maintain clear documentation of all agreements")
        advice_parts.append("• Conduct stakeholder legal awareness training")
        
        # Disclaimer
        advice_parts.append("\n*Disclaimer: This analysis provides general legal information and does not constitute legal advice. Consult qualified legal counsel for specific situations.*")
        
        return "\n".join(advice_parts)

class LegalActionVNI(SpecializedVNIBase):
    """Hybrid Legal VNI with static knowledge + dynamic adaptation"""
    
    def __init__(self, config: LegalOperActionConfig = None):
        config_obj = config or LegalOperActionConfig()
        super().__init__(topic_name="legal", config=config_obj.__dict__)
        
        self.config = config_obj
        self.reasoning_engine = LegalReasoningEngine(self.config)
        
        # Legal-specific dynamic adapter
        self.legal_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Replace base adapter with legal-specific one
        self.dynamic_adapter = self.legal_adapter
        
        logger.info(f"✅ Hybrid Legal VNI initialized: {self.config.vni_id}")
        logger.info(f"   Dynamic adaptation: {self.config.enable_dynamic_adaptation}")
    
    def forward(self, base_features: Dict, input_data: Any) -> Dict:
        """Process legal input with hybrid analysis"""
        
        # Extract text
        text = self._extract_text(input_data)
        
        # Extract legal concepts
        abstraction_data = base_features.get('abstraction_levels', {})
        legal_concepts = self.reasoning_engine.extract_legal_concepts(text, abstraction_data)
        
        # Get features
        features = self._extract_features(base_features)
        
        # Identify legal issues
        legal_issues = self.reasoning_engine.identify_legal_issues(legal_concepts, features)
        
        # Analyze compliance
        compliance_issues = self.reasoning_engine.analyze_compliance(legal_concepts, legal_issues, features)
        
        # Generate remedies
        remedies = self.reasoning_engine.generate_remedies(legal_issues, legal_concepts)
        
        # Generate advice
        legal_advice = self.reasoning_engine.generate_legal_advice(
            legal_issues, compliance_issues, remedies, legal_concepts
        )
        
        # Compile result
        base_result = {
            'legal_analysis': {
                'legal_issues': legal_issues,
                'compliance_issues': compliance_issues,
                'remedies': remedies,
                'identified_concepts': legal_concepts,
                'jurisdiction_indicators': legal_concepts.get('jurisdiction_indicators', [])
            },
            'legal_advice': legal_advice,
            'confidence_score': legal_issues[0]['risk_score'] if legal_issues else 0.0,
            'processing_metadata': {
                'legal_areas_identified': len(legal_concepts.get('legal_areas', [])),
                'issues_found': len(legal_issues),
                'compliance_frameworks': len(compliance_issues),
                'dynamic_adaptation_used': self.config.enable_dynamic_adaptation
            }
        }
        
        # Apply dynamic adaptation
        if self.config.enable_dynamic_adaptation and self.adaptation_strength > 0.1:
            adapted_result = self.apply_dynamic_adaptation(base_result, base_features)
            adapted_result['dynamic_adaptation_applied'] = True
            result = adapted_result
        else:
            result = base_result
        
        # Add metadata
        result['vni_metadata'] = {
            'vni_id': self.config.vni_id,
            'vni_type': 'operAction_legal_hybrid',
            'processing_stages': ['concept_extraction', 'issue_identification', 
                                'compliance_analysis', 'remedy_generation'],
            'success': True,
            'domain': 'legal',
            'hybrid_system': True,
            'static_knowledge_used': True,
            'dynamic_learning_enabled': self.config.enable_dynamic_adaptation
        }
        
        return result
    
    def _extract_text(self, input_data: Any) -> str:
        """Extract text from input"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data.get('text', str(input_data))
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
        return torch.zeros(1, 256) 
