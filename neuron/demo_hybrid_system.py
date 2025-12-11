# demo_hybrid_system.py - Demo all hybrid VNIs
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_system_demo")

def demo_hybrid_system():
    """Demonstrate all hybrid VNIs"""
    
    print("🚀 HYBRID VNI SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize all hybrid VNIs
    from neuron.operAction_unified._medical import (
        MedicalActionVNI, MedicalOperActionConfig,
        LegalActionVNI, LegalOperActionConfig,
        GeneralActionVNI, GeneralOperActionConfig,
        TechnicalActionVNI, TechnicalOperActionConfig
    )
    
    # Create configurations
    medical_config = MedicalOperActionConfig(
        vni_id="medical_hybrid_demo",
        enable_dynamic_adaptation=True
    )
    
    legal_config = LegalOperActionConfig(
        vni_id="legal_hybrid_demo",
        enable_dynamic_adaptation=True
    )
    
    general_config = GeneralOperActionConfig(
        vni_id="general_hybrid_demo",
        enable_dynamic_adaptation=True
    )
    
    technical_config = TechnicalOperActionConfig(
        vni_id="technical_hybrid_demo",
        enable_dynamic_adaptation=True
    )
    
    # Initialize VNIs
    medical_vni = MedicalActionVNI(medical_config)
    legal_vni = LegalActionVNI(legal_config)
    general_vni = GeneralActionVNI(general_config)
    technical_vni = TechnicalActionVNI(technical_config)
    
    print("\n✅ All hybrid VNIs initialized successfully!")
    print(f"   • Medical VNI: {medical_vni.config.vni_id}")
    print(f"   • Legal VNI: {legal_vni.config.vni_id}")
    print(f"   • General VNI: {general_vni.config.vni_id}")
    print(f"   • Technical VNI: {technical_vni.config.vni_id}")
    
    # Test cases
    test_cases = [
        {
            'name': 'Medical Query',
            'text': 'Patient with fever, cough, and chest pain for 3 days',
            'vni': medical_vni,
            'expected_topic': 'medical'
        },
        {
            'name': 'Legal Query',
            'text': 'Contract clause about liability limitation and data privacy compliance',
            'vni': legal_vni,
            'expected_topic': 'legal'
        },
        {
            'name': 'Technical Query',
            'text': 'Python code throwing runtime error when processing large dataset',
            'vni': technical_vni,
            'expected_topic': 'technical'
        },
        {
            'name': 'General Query',
            'text': 'How to analyze market trends and develop business strategy?',
            'vni': general_vni,
            'expected_topic': 'general'
        }
    ]
    
    print("\n🧪 RUNNING TEST CASES")
    print("-" * 50)
    
    for test in test_cases:
        print(f"\n📋 {test['name']}:")
        print(f"   Input: {test['text']}")
        print(f"   Expected Topic: {test['expected_topic']}")
        
        # Create mock base features
        base_features = {
            'primary_topic': test['expected_topic'],
            'complexity': 0.7,
            'abstraction_levels': {
                'semantic': {
                    'tensor': torch.randn(1, 512),
                    'concepts': test['text'].split()[:10],
                    'intent': 'analysis'
                }
            }
        }
        
        # Process with VNI
        with torch.no_grad():
            result = test['vni'](base_features, test['text'])
        
        # Display results
        if result.get('vni_metadata', {}).get('success', False):
            print(f"   ✅ Processing successful")
            print(f"   VNI Type: {result['vni_metadata']['vni_type']}")
            
            # Show confidence
            if 'confidence_score' in result:
                print(f"   Confidence: {result['confidence_score']:.1%}")
            
            # Show dynamic adaptation
            if result.get('dynamic_adaptation_applied', False):
                print(f"   🔄 Dynamic adaptation applied")
            
            # Show key output
            if 'medical_advice' in result:
                print(f"   Medical Advice: {result['medical_advice'][:100]}...")
            elif 'legal_advice' in result:
                print(f"   Legal Advice: {result['legal_advice'][:100]}...")
            elif 'technical_advice' in result:
                print(f"   Technical Advice: {result['technical_advice'][:100]}...")
            elif 'general_advice' in result:
                print(f"   General Advice: {result['general_advice'][:100]}...")
        
        else:
            print(f"   ❌ Processing failed")
    
    print("\n" + "=" * 60)
    print("🎯 HYBRID SYSTEM FEATURES:")
    print("  ✅ Static knowledge bases for each domain")
    print("  ✅ Dynamic adaptation networks")
    print("  ✅ Learning from interactions")
    print("  ✅ State persistence and recovery")
    print("  ✅ Backward compatibility")
    print("  ✅ Multi-domain support (General VNI)")
    
    print("\n📊 VNI CAPABILITIES SUMMARY:")
    for vni_name, vni in [('Medical', medical_vni), ('Legal', legal_vni), 
                         ('General', general_vni), ('Technical', technical_vni)]:
        caps = vni.get_capabilities()
        print(f"  • {vni_name}: {caps['vni_type']}")
        print(f"    - Adaptation strength: {caps['adaptation_strength']:.2f}")
        print(f"    - Learned patterns: {caps['learned_patterns']}")
        print(f"    - Usage count: {caps['usage_count']}")
    
    return {
        'medical': medical_vni,
        'legal': legal_vni,
        'general': general_vni,
        'technical': technical_vni
    }

if __name__ == "__main__":
    # Run the demo
    vnis = demo_hybrid_system()
    
    # Save states for persistence demonstration
    print("\n💾 SAVING VNI STATES...")
    for name, vni in vnis.items():
        vni.save_state(f"{name}_vni_state.json")
        print(f"   Saved {name} VNI state to {name}_vni_state.json")
    
    print("\n✅ Demo complete! Hybrid VNI system is ready for integration.") 
