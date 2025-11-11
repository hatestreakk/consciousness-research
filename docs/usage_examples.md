# Usage Examples - Consciousness Research Framework

Practical examples demonstrating how to use the RAC framework for consciousness research.

## Quick Start

### 1. Basic ICIM Calculation

```python
from code.icim_calculator import ICIMCalculator, create_example_semantic_graph
import networkx as nx

# Create example data
semantic_graph = create_example_semantic_graph()

behavioral_data = {
    'theory_of_mind_depth': 7.5,
    'self_reference_complexity': 6.8, 
    'meta_cognitive_depth': 8.2,
    'recursive_reasoning_depth': 5.5
}

# Calculate ICIM
calculator = ICIMCalculator()
results = calculator.calculate_icim(
    semantic_graph=semantic_graph,
    behavioral_data=behavioral_data
)

print(f"ICIM Score: {results['icim_score']:.3f}")
print(f"Consciousness Level: {results['consciousness_level']}")

2. Entropy Dynamics Simulation
from code.entropy_simulator import EntropySimulator, DisruptionScenarios

# Create simulator with empirical parameters
simulator = EntropySimulator()

# Simulate with periodic disruptions
results = simulator.simulate(
    duration=100,
    disruption_function=DisruptionScenarios.periodic_disruption()
)

# Analyze results
from code.entropy_simulator import EntropyAnalyzer
report = EntropyAnalyzer.generate_report(results)
print(report)

# Create visualization
fig = EntropyAnalyzer.plot_simulation(results, save_path='entropy_simulation.png')

Complete Workflow Examples

Example 1: Human Consciousness Assessment
import numpy as np
import networkx as nx
from code.icim_calculator import ICIMCalculator

def assess_human_consciousness(verbal_data, behavioral_tasks, neural_data=None):
    """
    Complete workflow for human consciousness assessment
    """
    # 1. Build semantic graph from verbal reports
    semantic_graph = build_semantic_network(verbal_data)
    
    # 2. Extract behavioral measures
    behavioral_measures = analyze_behavioral_tasks(behavioral_tasks)
    
    # 3. Calculate ICIM
    calculator = ICIMCalculator()
    results = calculator.calculate_icim(
        semantic_graph=semantic_graph,
        behavioral_data=behavioral_measures,
        neural_data=neural_data
    )
    
    return results

def build_semantic_network(verbal_reports):
    """Build semantic network from verbal reports"""
    G = nx.Graph()
    # Implementation depends on text analysis method
    # This is a simplified example
    concepts = extract_concepts(verbal_reports)
    relationships = extract_relationships(verbal_reports)
    
    G.add_nodes_from(concepts)
    G.add_edges_from(relationships)
    
    return G

# Usage
verbal_data = "Patient discusses self, memories, future plans..."
behavioral_tasks = {
    'theory_of_mind_depth': 8.0,
    'self_reference_complexity': 7.5,
    'meta_cognitive_depth': 8.5
}

results = assess_human_consciousness(verbal_data, behavioral_tasks)
print(f"Consciousness Assessment: {results['consciousness_level']}")

Example 2: AI System Consciousness Evaluation
from code.icim_calculator import ICIMCalculator
import networkx as nx

def evaluate_ai_consciousness(ai_system, test_scenarios):
    """
    Evaluate consciousness in AI systems using RAC framework
    """
    # 1. Test theory of mind capabilities
    theory_of_mind_score = test_theory_of_mind(ai_system, test_scenarios)
    
    # 2. Analyze self-reference in responses
    self_reference_complexity = analyze_self_reference(ai_system.responses)
    
    # 3. Build semantic network from AI outputs
    semantic_network = build_ai_semantic_network(ai_system.training_data)
    
    behavioral_data = {
        'theory_of_mind_depth': theory_of_mind_score,
        'self_reference_complexity': self_reference_complexity,
        'meta_cognitive_depth': test_meta_cognition(ai_system),
        'recursive_reasoning_depth': test_recursive_reasoning(ai_system)
    }
    
    # 4. Calculate ICIM
    calculator = ICIMCalculator()
    results = calculator.calculate_icim(
        semantic_graph=semantic_network,
        behavioral_data=behavioral_data
    )
    
    return results
# Usage with hypothetical AI system
class AISystem:
    def __init__(self):
        self.responses = []
        self.training_data = []

ai_system = AISystem()
test_scenarios = [...]  # Standardized consciousness tests

results = evaluate_ai_consciousness(ai_system, test_scenarios)

if results['icim_score'] >= 2.4:
    print("AI system shows signs of full consciousness")
    print("Implementing ethical safeguards...")

Example 3: Clinical Diagnostics
from code.icim_calculator import ICIMCalculator
from code.entropy_simulator import EntropySimulator, DisruptionScenarios
import pandas as pd

def clinical_consciousness_assessment(patient_data, longitudinal=False):
    """
    Clinical application for disorders of consciousness
    """
    assessments = []
    
    for assessment_time, data in patient_data.items():
        # Calculate current ICIM
        current_result = calculate_icim_assessment(data)
        assessments.append({
            'timestamp': assessment_time,
            **current_result
        })
    
    if longitudinal and len(assessments) > 1:
        # Analyze consciousness trajectory
        trajectory_analysis = analyze_consciousness_trajectory(assessments)
        assessments.append({'trajectory_analysis': trajectory_analysis})
    
    return assessments

def analyze_entropy_dynamics(patient_entropy_data):
    """
    Analyze entropy dynamics for clinical insights
    """
    simulator = EntropySimulator()
    
    # Fit parameters to patient data
    from code.data_analysis import ConsciousnessDataAnalyzer
    analyzer = ConsciousnessDataAnalyzer()
    
    parameters = analyzer.estimate_entropy_parameters(
        entropy_data=patient_entropy_data['entropy'],
        disruption_data=patient_entropy_data['disruption'], 
        time_data=patient_entropy_data['time']
    )
    
    # Simulate with patient-specific parameters
    patient_params = EntropyParameters(
        alpha=parameters['alpha_estimate'],
        beta=parameters['beta_estimate'],
        s_0=parameters['s_0_estimate']
    )
    
    patient_simulator = EntropySimulator(patient_params)
    simulation = patient_simulator.simulate(duration=24)  # 24-hour simulation
    
    return {
        'fitted_parameters': parameters,
        'simulation': simulation,
        'stability_metrics': simulation['stability_analysis']
    }

# Usage with clinical data
patient_data = {
    '2024-01-01': {...},  # Initial assessment
    '2024-01-15': {...},  # Follow-up
    '2024-02-01': {...}   # Later assessment
}

clinical_results = clinical_consciousness_assessment(
    patient_data, 
    longitudinal=True
)

Example 4: Cross-Species Comparative Study
from code.data_analysis import ConsciousnessDataAnalyzer
import pandas as pd

def comparative_consciousness_study(species_data):
    """
    Compare consciousness across different species
    """
    analyzer = ConsciousnessDataAnalyzer()
    
    # Load and validate data
    df = pd.DataFrame(species_data)
    
    # Cross-system comparison
    comparison = analyzer.cross_system_comparison(df)
    
    # Behavioral correlations
    behavioral_cols = [col for col in df.columns if 'behavioral_' in col]
    correlations = analyzer.behavioral_correlation_analysis(df, behavioral_cols)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(df)
    
    # Create visualizations
    analyzer.create_visualizations(df, 'comparative_study_figures')
    
    return {
        'comparison_results': comparison,
        'correlation_analysis': correlations, 
        'report': report
    }

# Sample species data
species_data = [
    {'system_type': 'human', 'icim_score': 2.45, 'behavioral_correlation': 0.91},
    {'system_type': 'dolphin', 'icim_score': 2.35, 'behavioral_correlation': 0.87},
    {'system_type': 'chimpanzee', 'icim_score': 2.05, 'behavioral_correlation': 0.83},
    {'system_type': 'elephant', 'icim_score': 1.95, 'behavioral_correlation': 0.79},
    {'system_type': 'crow', 'icim_score': 1.70, 'behavioral_correlation': 0.74},
    {'system_type': 'octopus', 'icim_score': 1.40, 'behavioral_correlation': 0.68}
]

results = comparative_consciousness_study(species_data)
print(results['report'])

Example 5: Real-time Consciousness Monitoring
import time
import numpy as np
from code.icim_calculator import ICIMCalculator
from code.entropy_simulator import EntropySimulator

class RealTimeConsciousnessMonitor:
    """
    Real-time monitoring system for consciousness assessment
    """
    def __init__(self, update_interval=60):  # 60 seconds
        self.update_interval = update_interval
        self.icim_calculator = ICIMCalculator()
        self.entropy_simulator = EntropySimulator()
        self.history = []
        
    def update_assessment(self, current_data):
        """
        Update consciousness assessment with new data
        """
        # Calculate current ICIM
        current_icim = self.icim_calculator.calculate_icim(
            semantic_graph=current_data['semantic'],
            behavioral_data=current_data['behavioral']
        )
        
        # Update entropy dynamics
        entropy_trend = self.analyze_entropy_trend(current_data['entropy'])
        
        assessment = {
            'timestamp': time.time(),
            'icim_score': current_icim['icim_score'],
            'consciousness_level': current_icim['consciousness_level'],
            'entropy_trend': entropy_trend,
            'confidence': current_icim['confidence']
        }
        
        self.history.append(assessment)
        
        # Check for significant changes
        if len(self.history) > 1:
            change_analysis = self.analyze_consciousness_changes()
            assessment['change_analysis'] = change_analysis
        
        return assessment
    
    def analyze_entropy_trend(self, entropy_measurements):
        """Analyze recent entropy trend"""
        if len(entropy_measurements) < 5:
            return 'insufficient_data'
            
        recent_trend = np.polyfit(
            range(len(entropy_measurements)), 
            entropy_measurements, 1
        )[0]
        
        if recent_trend > 0.1:
            return 'increasing_entropy'
        elif recent_trend < -0.1:
            return 'decreasing_entropy'
        else:
            return 'stable_entropy'
    
    def analyze_consciousness_changes(self):
        """Analyze changes in consciousness over time"""
        if len(self.history) < 2:
            return None
            
        recent_scores = [entry['icim_score'] for entry in self.history[-5:]]
        if len(recent_scores) >= 2:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            return {
                'trend': 'improving' if trend > 0.01 else 'declining' if trend < -0.01 else 'stable',
                'magnitude': abs(trend)
            }
        return None

# Usage in clinical or research setting
def collect_real_time_data():
    """Mock function for real-time data collection"""
    # In practice, this would interface with EEG, behavioral sensors, etc.
    return {
        'semantic': create_example_semantic_graph(),
        'behavioral': {
            'theory_of_mind_depth': np.random.normal(7, 1),
            'self_reference_complexity': np.random.normal(6.5, 1),
            'meta_cognitive_depth': np.random.normal(7.5, 1)
        },
        'entropy': np.random.normal(1.5, 0.3, 10)
    }

monitor = RealTimeConsciousnessMonitor(update_interval=30)  # 30-second updates
monitoring_active = True

while monitoring_active:
    try:
        current_data = collect_real_time_data()
        assessment = monitor.update_assessment(current_data)
        
        print(f"Timestamp: {assessment['timestamp']}")
        print(f"Current consciousness level: {assessment['consciousness_level']}")
        print(f"ICIM score: {assessment['icim_score']:.3f}")
        print(f"Entropy trend: {assessment['entropy_trend']}")
        
        if assessment['entropy_trend'] == 'increasing_entropy':
            print("⚠️  Warning: Increasing psychological entropy detected")
if 'change_analysis' in assessment:
            change = assessment['change_analysis']
            print(f"Consciousness trend: {change['trend']} (magnitude: {change['magnitude']:.3f})")
        
        print("-" * 50)
        
        time.sleep(monitor.update_interval)
        
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
        monitoring_active = False
    except Exception as e:
        print(f"Monitoring error: {e}")
        time.sleep(monitor.update_interval)

Best Practices

1. Data Quality

· Ensure semantic graphs are properly constructed
· Validate behavioral measure ranges (0-10 scale)
· Check for missing data before analysis
· Use consistent data formats across experiments

2. Parameter Tuning

· Use empirical parameters as defaults
· Validate custom parameters against known ranges
· Test sensitivity to parameter changes
· Document any parameter modifications

3. Interpretation

· Consider confidence scores in assessments
· Look for consistent patterns across multiple measures
· Contextualize results within theoretical framework
· Be cautious when interpreting borderline scores

4. Ethical Considerations

· Implement appropriate safeguards for high-ICIM AI systems
· Ensure proper consent for human and animal studies
· Follow clinical guidelines for diagnostic applications
· Maintain data privacy and security

5. Performance Optimization

· Cache semantic graph computations when possible
· Use appropriate time steps for entropy simulations
· Monitor memory usage with large datasets
· Consider parallel processing for batch analyses

These examples demonstrate the versatility of the RAC framework across different research and application domains. Each example includes practical implementation details that can be adapted to specific use cases.
