# operAction_legal.py
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger("operAction_legal")

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
        
        # Automatic quality assessment based on response characteristics
        quality_score = 0.5  # Base score
        
        # Check for comprehensive analysis
        if 'legal_analysis' in response:
            analysis = response['legal_analysis']
            if analysis.get('legal_issues') and len(analysis['legal_issues']) > 0:
                quality_score += 0.2
            if analysis.get('compliance_issues') and len(analysis['compliance_issues']) > 0:
                quality_score += 0.2
            if response.get('confidence_score', 0) > 0.7:
                quality_score += 0.1
        
        return min(quality_score, 1.0)  # Cap at 1.0
    
    def learn_from_success(self, query, response):
        """Extract patterns from successful responses"""
        success_pattern = {
            'query_pattern': self.extract_query_pattern(query),
            'response_pattern': self.extract_response_pattern(response),
            'timestamp': time.time(),
            'quality_score': self.response_quality_history[-1]
        }
        self.success_patterns.append(success_pattern)
        
        # Keep only recent patterns (last 100)
        if len(self.success_patterns) > 100:
            self.success_patterns = self.success_patterns[-100:]
    
    def extract_query_pattern(self, query):
        """Extract key features from query for pattern matching"""
        if isinstance(query, dict) and 'abstraction_data' in query:
            # Extract from cognitive concepts
            cognitive_data = query['abstraction_data'].get('cognitive', {})
            concepts = cognitive_data.get('concepts', [])
            return {
                'concept_count': len(concepts),
                'key_concepts': concepts[:5],  # Top 5 concepts
                'domain_indicators': self.detect_domain_indicators(concepts)
            }
        return {'raw_query': str(query)[:200]}  # Truncate for storage
    
    def extract_response_pattern(self, response):
        """Extract key features from successful response"""
        pattern = {
            'analysis_depth': 0,
            'issues_identified': 0,
            'confidence_level': response.get('confidence_score', 0)
        }
        
        if 'legal_analysis' in response:
            analysis = response['legal_analysis']
            pattern['issues_identified'] = len(analysis.get('legal_issues', []))
            pattern['compliance_checks'] = len(analysis.get('compliance_issues', []))
            pattern['remedies_suggested'] = len(analysis.get('remedies', []))
        
        return pattern
    
    def detect_domain_indicators(self, concepts):
        """Detect which legal domains are indicated by concepts"""
        domains = {
            'contract_law': ['contract', 'agreement', 'breach', 'liability'],
            'privacy_law': ['privacy', 'data', 'gdpr', 'compliance'],
            'intellectual_property': ['copyright', 'patent', 'trademark'],
            'employment_law': ['employee', 'employer', 'discrimination']
        }
        
        detected_domains = []
        for domain, keywords in domains.items():
            if any(any(keyword in concept.lower() for keyword in keywords) for concept in concepts):
                detected_domains.append(domain)
        
        return detected_domains
    
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
class LegalOperActionConfig:
    """Configuration for legal operAction VNI"""
    vni_id: str = "operAction_legal"
    reasoning_depth: str = "comprehensive"
    legal_knowledge_base: Dict[str, Any] = None
    confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.legal_knowledge_base is None:
            self.legal_knowledge_base = {
                'legal_areas': {
                    'contract_law': ['breach', 'liability', 'enforcement', 'terms'],
                    'privacy_law': ['gdpr', 'compliance', 'data_protection', 'consent'],
                    'intellectual_property': ['copyright', 'patent', 'trademark', 'infringement'],
                    'employment_law': ['discrimination', 'termination', 'contract', 'rights'],
                    'healthcare_law': ['hipaa', 'malpractice', 'consent', 'regulations']
                },
                'common_clauses': {
                    'liability_limitation': 'Limits party responsibility for damages',
                    'confidentiality': 'Protects sensitive information sharing',
                    'termination': 'Conditions for ending agreement',
                    'governing_law': 'Specifies applicable jurisdiction',
                    'indemnification': 'Compensation for losses or damages'
                },
                'compliance_frameworks': {
                    'gdpr': ['data_protection', 'user_consent', 'privacy_by_design'],
                    'hipaa': ['patient_privacy', 'data_security', 'medical_records'],
                    'ccpa': ['consumer_rights', 'data_transparency', 'opt_out_requirements']
                }
            }

class LegalKnowledgeGraph:
    def __init__(self):
        self.legal_ontology = self.build_legal_ontology()
        self.case_precedents = self.load_legal_cases()
        self.statute_database = self.load_statutes()
    
    def build_legal_ontology(self):
        """Build proper legal concept relationships"""
        ontology = {
            'contract_law': {
                'concepts': ['offer', 'acceptance', 'consideration', 'breach', 'remedies'],
                'relationships': {
                    'breach': ['leads_to', 'damages', 'termination'],
                    'offer': ['requires', 'acceptance', 'consideration']
                }
            },
            'privacy_law': {
                'concepts': ['consent', 'data_processing', 'right_to_be_forgotten', 'compliance'],
                'relationships': {
                    'data_processing': ['requires', 'consent', 'legal_basis']
                }
            }
        }
        return ontology
    
    def find_related_statutes(self, legal_issue):
        """Find relevant laws and regulations"""
        # Use semantic matching instead of keyword matching
        return self.semantic_search_statutes(legal_issue)
    
    def get_legal_precedents(self, case_facts):
        """Find similar legal cases"""
        return self.case_based_reasoning(case_facts)
    
class LegalReasoningEngine(nn.Module):
    """Legal reasoning and analysis engine"""
    def __init__(self, config: LegalOperActionConfig):
        super().__init__()
        self.config = config
        self.knowledge_graph = LegalKnowledgeGraph()        
        # Legal reasoning networks
        self.legal_feature_encoder = nn.Sequential(
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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.compliance_checker = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        self.remedy_suggester = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.Softmax(dim=-1)
        )
        
    def extract_legal_concepts(self, abstraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract legal concepts from abstraction data"""
        legal_concepts = {
            'legal_areas': [],
            'key_clauses': [],
            'parties_involved': [],
            'obligations': [],
            'risks_identified': []
        }
        
        # Extract from cognitive level
        if 'cognitive' in abstraction_data:
            cognitive = abstraction_data['cognitive']
            concepts = cognitive.get('concepts', [])
            
            # Legal concept mapping
            contract_keywords = ['contract', 'agreement', 'clause', 'term', 'party']
            liability_keywords = ['liability', 'damages', 'breach', 'compensation']
            privacy_keywords = ['privacy', 'data', 'gdpr', 'compliance', 'consent']
            ip_keywords = ['copyright', 'patent', 'trademark', 'intellectual']
            employment_keywords = ['employee', 'employer', 'termination', 'discrimination']
            
            for concept in concepts:
                concept_lower = concept.lower()
                if any(keyword in concept_lower for keyword in contract_keywords):
                    legal_concepts['legal_areas'].append('contract_law')
                elif any(keyword in concept_lower for keyword in liability_keywords):
                    legal_concepts['key_clauses'].append('liability_limitation')
                elif any(keyword in concept_lower for keyword in privacy_keywords):
                    legal_concepts['legal_areas'].append('privacy_law')
                elif any(keyword in concept_lower for keyword in ip_keywords):
                    legal_concepts['legal_areas'].append('intellectual_property')
                elif any(keyword in concept_lower for keyword in employment_keywords):
                    legal_concepts['legal_areas'].append('employment_law')
                
                # Extract parties
                if 'party' in concept_lower or 'client' in concept_lower or 'company' in concept_lower:
                    legal_concepts['parties_involved'].append(concept)
                
                # Extract obligations
                if 'must' in concept_lower or 'shall' in concept_lower or 'required' in concept_lower:
                    legal_concepts['obligations'].append(concept)
                
                # Extract risks
                if 'risk' in concept_lower or 'liability' in concept_lower or 'penalty' in concept_lower:
                    legal_concepts['risks_identified'].append(concept)
        
        # Remove duplicates
        for key in legal_concepts:
            legal_concepts[key] = list(set(legal_concepts[key]))
        
        return legal_concepts
    
    def identify_legal_issues(self, legal_concepts: Dict[str, Any], 
                            cognitive_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """Identify potential legal issues"""
        legal_issues = []
        
        # Analyze legal concepts
        legal_embedding = self.legal_concept_analyzer(cognitive_tensor)
        risk_scores = self.risk_assessor(legal_embedding)
        
        # Check for specific legal issues
        legal_areas = legal_concepts.get('legal_areas', [])
        key_clauses = legal_concepts.get('key_clauses', [])
        
        # Contract law issues
        if 'contract_law' in legal_areas:
            issues = self.analyze_contract_issues(legal_concepts, risk_scores[0][0].item())
            legal_issues.extend(issues)
        
        # Privacy law issues
        if 'privacy_law' in legal_areas:
            issues = self.analyze_privacy_issues(legal_concepts, risk_scores[0][1].item())
            legal_issues.extend(issues)
        
        # IP law issues
        if 'intellectual_property' in legal_areas:
            issues = self.analyze_ip_issues(legal_concepts, risk_scores[0][2].item())
            legal_issues.extend(issues)
        
        # Employment law issues
        if 'employment_law' in legal_areas:
            issues = self.analyze_employment_issues(legal_concepts, risk_scores[0][3].item())
            legal_issues.extend(issues)
        
        # Sort by risk score
        legal_issues.sort(key=lambda x: x['risk_score'], reverse=True)
        return legal_issues
    
    def analyze_contract_issues(self, legal_concepts: Dict[str, Any], base_risk: float) -> List[Dict[str, Any]]:
        """Analyze contract law issues"""
        issues = []
        
        if 'liability_limitation' in legal_concepts.get('key_clauses', []):
            issues.append({
                'issue': 'Unlimited Liability Exposure',
                'risk_score': base_risk * 0.8,
                'area': 'contract_law',
                'description': 'Potential unlimited liability without proper limitation clauses',
                'suggested_action': 'Review and include liability limitation clauses'
            })
        
        if len(legal_concepts.get('obligations', [])) > 5:
            issues.append({
                'issue': 'Overly Burdensome Obligations',
                'risk_score': base_risk * 0.6,
                'area': 'contract_law',
                'description': 'Excessive obligations may lead to compliance difficulties',
                'suggested_action': 'Simplify or clarify obligation terms'
            })
        
        return issues
    
    def analyze_privacy_issues(self, legal_concepts: Dict[str, Any], base_risk: float) -> List[Dict[str, Any]]:
        """Analyze privacy law issues"""
        issues = []
        
        if any('data' in concept.lower() for concept in legal_concepts.get('concepts', [])):
            issues.append({
                'issue': 'Data Privacy Compliance',
                'risk_score': base_risk * 0.9,
                'area': 'privacy_law',
                'description': 'Data handling may require GDPR/HIPAA compliance',
                'suggested_action': 'Implement data protection measures and obtain consents'
            })
        
        return issues
    
    def analyze_ip_issues(self, legal_concepts: Dict[str, Any], base_risk: float) -> List[Dict[str, Any]]:
        """Analyze intellectual property issues"""
        issues = []
        
        ip_keywords = ['copyright', 'patent', 'trademark', 'intellectual']
        if any(any(keyword in concept.lower() for keyword in ip_keywords) 
               for concept in legal_concepts.get('concepts', [])):
            issues.append({
                'issue': 'Intellectual Property Protection',
                'risk_score': base_risk * 0.7,
                'area': 'intellectual_property',
                'description': 'Potential IP infringement or protection needs',
                'suggested_action': 'Conduct IP audit and implement protection strategies'
            })
        
        return issues
    
    def analyze_employment_issues(self, legal_concepts: Dict[str, Any], base_risk: float) -> List[Dict[str, Any]]:
        """Analyze employment law issues"""
        issues = []
        
        employment_keywords = ['employee', 'employer', 'termination', 'discrimination']
        if any(any(keyword in concept.lower() for keyword in employment_keywords) 
               for concept in legal_concepts.get('concepts', [])):
            issues.append({
                'issue': 'Employment Law Compliance',
                'risk_score': base_risk * 0.75,
                'area': 'employment_law',
                'description': 'Potential employment law violations or compliance requirements',
                'suggested_action': 'Review employment practices and policies'
            })
        
        return issues
    
    def check_compliance(self, legal_issues: List[Dict[str, Any]], 
                       structural_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """Check regulatory compliance requirements"""
        compliance_issues = []
        
        # Neural network compliance checking
        compliance_scores = self.compliance_checker(structural_tensor)
        
        frameworks = ['gdpr', 'hipaa', 'ccpa', 'sox', 'pci_dss']
        
        for i, framework in enumerate(frameworks):
            if i < len(compliance_scores[0]) and compliance_scores[0][i].item() > 0.5:
                compliance_issues.append({
                    'framework': framework.upper(),
                    'compliance_level': compliance_scores[0][i].item(),
                    'requirements': self.get_framework_requirements(framework),
                    'assessment': f'Potential {framework.upper()} compliance considerations'
                })
        
        return compliance_issues
    
    def get_framework_requirements(self, framework: str) -> List[str]:
        """Get requirements for compliance framework"""
        requirements_map = {
            'gdpr': ['Data Protection Impact Assessment', 'User Consent Management', 
                    'Right to be Forgotten', 'Data Breach Notification'],
            'hipaa': ['Patient Authorization', 'Medical Records Security', 
                     'Privacy Notice', 'Business Associate Agreements'],
            'ccpa': ['Consumer Data Rights', 'Opt-Out Mechanisms', 
                    'Data Transparency', 'Verifiable Requests']
        }
        return requirements_map.get(framework, ['General compliance review recommended'])
    
    def suggest_remedies(self, legal_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest legal remedies and actions"""
        remedies = []
        
        for issue in legal_issues[:3]:  # Top 3 issues
            remedy_embedding = torch.tensor([issue['risk_score']] * 96)  # Simplified
            remedy_scores = self.remedy_suggester(remedy_embedding.unsqueeze(0))
            
            remedies.append({
                'issue': issue['issue'],
                'recommended_actions': [
                    issue.get('suggested_action', 'Consult legal counsel'),
                    'Document all relevant communications',
                    'Review applicable laws and regulations'
                ],
                'urgency': 'high' if issue['risk_score'] > 0.7 else 'medium',
                'preventive_measures': [
                    'Regular legal compliance audits',
                    'Clear contract drafting and review processes',
                    'Employee training on legal requirements'
                ]
            })
        
        return remedies
    
    def generate_legal_advice(self, legal_issues: List[Dict[str, Any]],
                            compliance_issues: List[Dict[str, Any]],
                            remedies: List[Dict[str, Any]]) -> str:
        """Generate comprehensive legal advice"""
        
        if not legal_issues:
            return "No significant legal issues identified based on available information. General legal review still recommended."
        
        advice_parts = []
        
        # Summary of key issues
        primary_issue = legal_issues[0]
        advice_parts.append(f"Primary legal concern: {primary_issue['issue']} (risk score: {primary_issue['risk_score']:.1%})")
        
        # Compliance considerations
        if compliance_issues:
            advice_parts.append("Compliance frameworks to consider:")
            for comp in compliance_issues[:2]:
                advice_parts.append(f"- {comp['framework']} (level: {comp['compliance_level']:.1%})")
        
        # Recommended actions
        if remedies:
            primary_remedy = remedies[0]
            advice_parts.append("Immediate actions recommended:")
            for action in primary_remedy['recommended_actions'][:2]:
                advice_parts.append(f"- {action}")
        
        # General disclaimer
        advice_parts.append("Note: This analysis is for informational purposes only and does not constitute legal advice.")
        
        return " ".join(advice_parts)

class OperActionLegal(nn.Module):
    """Legal reasoning operAction VNI"""
    
    def __init__(self, config: LegalOperActionConfig = None):
        super().__init__()
        self.config = config or LegalOperActionConfig()
        self.reasoning_engine = LegalReasoningEngine(self.config)
        
        logger.info(f"Legal operAction VNI initialized with ID: {self.config.vni_id}")
    
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process legal reasoning request"""
        
        try:
            abstraction_data = input_data.get('abstraction_data', {})
            
            # Extract legal concepts
            legal_concepts = self.reasoning_engine.extract_legal_concepts(abstraction_data)
            
            # Get cognitive tensor for neural processing
            cognitive_tensor = abstraction_data.get('cognitive', {}).get('tensor', torch.zeros(256))
            structural_tensor = abstraction_data.get('structural', {}).get('tensor', torch.zeros(256))
            
            # Identify legal issues
            legal_issues = self.reasoning_engine.identify_legal_issues(
                legal_concepts, cognitive_tensor
            )
            
            # Check compliance
            compliance_issues = self.reasoning_engine.check_compliance(
                legal_issues, structural_tensor
            )
            
            # Suggest remedies
            remedies = self.reasoning_engine.suggest_remedies(legal_issues)
            
            # Generate final advice
            legal_advice = self.reasoning_engine.generate_legal_advice(
                legal_issues, compliance_issues, remedies
            )
            
            # Compile results
            results = {
                'legal_analysis': {
                    'legal_issues': legal_issues,
                    'compliance_issues': compliance_issues,
                    'remedies': remedies,
                    'identified_concepts': legal_concepts
                },
                'legal_advice': legal_advice,
                'confidence_score': legal_issues[0]['risk_score'] if legal_issues else 0.0,
                'processing_metadata': {
                    'legal_areas_identified': len(legal_concepts.get('legal_areas', [])),
                    'issues_found': len(legal_issues),
                    'compliance_frameworks': len(compliance_issues)
                }
            }
            
            # Add VNI metadata
            results['vni_metadata'] = {
                'vni_id': self.config.vni_id,
                'vni_type': 'operAction_legal',
                'processing_stages': ['concept_extraction', 'issue_identification', 
                                    'compliance_checking', 'remedy_suggestion'],
                'success': True,
                'domain': 'legal'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Legal operAction processing failed: {str(e)}")
            return self._generate_error_output(str(e))
    
    def _generate_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Generate error output"""
        return {
            'legal_analysis': {},
            'legal_advice': f"Legal analysis unavailable: {error_msg}",
            'vni_metadata': {
                'vni_id': self.config.vni_id,
                'vni_type': 'operAction_legal',
                'success': False,
                'error': error_msg,
                'domain': 'legal'
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return VNI capabilities description"""
        return {
            'vni_type': 'operAction_legal',
            'description': 'Legal analysis and compliance reasoning VNI',
            'capabilities': [
                'Legal issue identification',
                'Regulatory compliance checking',
                'Risk assessment and scoring',
                'Remedy and action suggestion',
                'Multi-jurisdiction considerations'
            ],
            'input_types': ['legal_abstraction_data'],
            'output_types': ['legal_analysis', 'legal_advice', 'compliance_assessment'],
            'domain': 'legal'
        }

# Demonstration and testing
def test_legal_operAction():
    """Test the legal operAction with sample inputs"""
    
    # Initialize legal operAction
    config = LegalOperActionConfig(vni_id="operAction_legal_test_001")
    legal_vni = OperActionLegal(config)
    
    # Create test input data
    test_input = {
        'abstraction_data': {
            'cognitive': {
                'tensor': torch.randn(256),
                'concepts': ['contract', 'liability', 'data', 'privacy', 'compliance', 'gdpr'],
                'intent': 'analysis'
            },
            'structural': {
                'tensor': torch.randn(256),
                'logical_flow': 'complex'
            },
            'signal': {
                'tensor': torch.randn(256)
            }
        },
        'metadata': {
            'source_topics': ['legal'],
            'cross_domain': False
        }
    }
    
    print("=== Legal OperAction Demo Test ===\n")
    
    with torch.no_grad():
        results = legal_vni(test_input)
    
    # Display results
    print("Legal Analysis Results:")
    if results['legal_analysis']['legal_issues']:
        primary_issue = results['legal_analysis']['legal_issues'][0]
        print(f"Primary Issue: {primary_issue['issue']}")
        print(f"Risk Score: {primary_issue['risk_score']:.1%}")
        print(f"Area: {primary_issue['area']}")
    
    print("\nCompliance Considerations:")
    for comp in results['legal_analysis']['compliance_issues'][:2]:
        print(f"- {comp['framework']} (level: {comp['compliance_level']:.1%})")
    
    print(f"\nLegal Advice: {results['legal_advice']}")
    
    return legal_vni

if __name__ == "__main__":
    # Run demonstration
    test_legal_operAction() 