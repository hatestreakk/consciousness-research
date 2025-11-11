# API Documentation - Consciousness Research Framework

Complete API reference for the Resonance Architecture of Consciousness (RAC) implementation.

## Core Modules

### ICIM Calculator (`icim_calculator.py`)

#### `ICIMCalculator` Class

Main class for computing Integrated Complexity of Internal Models metric.

```python
calculator = ICIMCalculator(weights=None)

Parameters:

· weights: Optional custom weights for ICIM components. Defaults to empirical weights.

Methods:

calculate_icim(semantic_graph, behavioral_data, **kwargs)

Compute complete ICIM score for consciousness assessment.
results = calculator.calculate_icim(
    semantic_graph=networkx_graph,
    behavioral_data=behavior_dict,
    neural_data=neural_array,  # optional
    multi_modal_data=modality_dict  # optional
)

Returns:
{
    'icim_score': 2.45,  # Overall ICIM score
    'consciousness_level': 'Full Consciousness',  # Classification
    'components': {  # Individual component scores
        'structural_complexity': 0.8,
        'recursive_depth': 0.7,
        'information_integration': 0.9,
        'cross_modal_entanglement': 0.6
    },
    'weights': {...},  # Component weights used
    'confidence': 0.85  # Confidence estimate
}

Component Calculation Methods

· calculate_structural_complexity(semantic_graph)
· calculate_recursive_depth(behavioral_data)
· calculate_information_integration(modality_a, modality_b)
· calculate_cross_modal_entanglement(multi_modal_data)

Convenience Functions

calculate_icim(semantic_data, behavioral_data, **kwargs)

Quick function for ICIM calculation without creating class instance.

Entropy Simulator (entropy_simulator.py)

EntropySimulator Class

Simulates psychological entropy dynamics in conscious systems.
simulator = EntropySimulator(parameters=None)

Parameters:

· parameters: EntropyParameters instance. Uses empirical defaults if None.

EntropyParameters Dataclass
@dataclass
class EntropyParameters:
    alpha: float = 0.47    # Entropy production coefficient
    beta: float = 0.32     # Homeostatic regulation coefficient  
    gamma: float = 0.15    # Noise amplitude
    s_max: float = 8.3     # Maximum entropy capacity
    s_0: float = 1.2       # Baseline entropy
    noise_std: float = 0.08  # Noise standard deviation

Methods:

simulate(duration, time_step, initial_entropy, disruption_function, **kwargs)

Run entropy dynamics simulation.
results = simulator.simulate(
    duration=100.0,
    time_step=0.1,
    initial_entropy=1.2,
    disruption_function=disruption_func
)

Returns:
{
    'time': array([0.0, 0.1, 0.2, ...]),      # Time points
    'entropy': array([1.2, 1.18, 1.25, ...]), # Entropy values
    'disruption': array([0.1, 0.12, 0.09, ...]), # Disruption intensities
    'stable': array([True, False, True, ...]), # Stability regions
    'parameters': {...},  # Simulation parameters
    'stability_analysis': {...}  # Stability metrics
}

DisruptionScenarios Class

Predefined disruption scenarios for realistic simulations.

Static Methods:

· constant_disruption(level=0.3)
· periodic_disruption(amplitude=0.4, period=20.0)
· random_spike_disruption(base_level=0.1, spike_prob=0.03)
· increasing_stress_disruption(start_level=0.1, end_level=0.9)
· trauma_response_disruption(trauma_time=30.0, recovery_rate=0.1)

EntropyAnalyzer Class

Analysis and visualization tools for simulation results.

Static Methods:

· plot_simulation(results, save_path=None, show_stability=True)
· generate_report(results)

Data Analysis (data_analysis.py)

ConsciousnessDataAnalyzer Class

Comprehensive statistical analysis for consciousness research data.
analyzer = ConsciousnessDataAnalyzer()

Methods:

load_dataset(filepath)

Load consciousness research dataset from CSV file.

validate_icim_metric(df, ground_truth_col='consciousness_level')

Validate ICIM metric against ground truth assessments.

cross_system_comparison(df)

Compare consciousness metrics across different system types.

estimate_entropy_parameters(entropy_data, disruption_data, time_data)

Estimate entropy dynamics parameters from empirical data.

behavioral_correlation_analysis(df, behavioral_columns)

Analyze correlations between ICIM and behavioral measures.

generate_comprehensive_report(df)

Generate formatted analysis report.

create_visualizations(df, save_dir='figures')

Create comprehensive data visualizations.

Data Structures

Behavioral Data Dictionary
behavioral_data = {
    'theory_of_mind_depth': 7.5,        # 0-10 scale
    'self_reference_complexity': 6.8,   # 0-10 scale  
    'meta_cognitive_depth': 8.2,        # 0-10 scale
    'recursive_reasoning_depth': 5.5    # 0-10 scale
}

Multi-modal Data Dictionary
multi_modal_data = {
    'visual': np.array([...]),      # Visual modality data
    'auditory': np.array([...]),    # Auditory modality data
    'somatosensory': np.array([...]), # Somatosensory data
    # ... additional modalities
}

Error Handling

All methods include comprehensive error handling:

· ValueErrors: Invalid input parameters
· RuntimeErrors: Simulation failures
· Warnings: Non-critical issues that don't halt execution
· Type checking: Input validation for all parameters

Examples

See usage_examples.md for complete usage examples and theory_explanation.md for theoretical background.
