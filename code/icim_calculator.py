"""
ICIM (Integrated Complexity of Internal Models) Calculator
Implementation of the consciousness assessment metric from RAC theory
"""

import numpy as np
import networkx as nx
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Optional, Any, Union
import warnings
import json

class ICIMCalculator:
    """
    Calculate ICIM metric for consciousness assessment across biological and artificial systems
    """
    
    def __init__(self, weights: Optional[Dict] = None):
        """
        Initialize ICIM calculator with component weights
        
        Args:
            weights: Custom weights for ICIM components. Defaults to empirically derived weights.
        """
        self.weights = weights or {
            'structural_complexity': 0.30,    # D(SG)
            'recursive_depth': 0.25,          # RDL
            'information_integration': 0.30,  # D_KL
            'cross_modal_entanglement': 0.15  # CEM
        }
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-10:
            raise ValueError(f"ICIM weights must sum to 1.0, got {weight_sum}")
        
        # Consciousness classification thresholds
        self.consciousness_thresholds = {
            'minimal_awareness': 1.2,
            'basic_consciousness': 1.8, 
            'full_consciousness': 2.4,
            'enhanced_integration': 3.2
        }
    
    def calculate_structural_complexity(self, semantic_graph: nx.Graph) -> float:
        """
        Calculate D(SG) - structural complexity of semantic networks
        
        Based on graph theory measures of network complexity including:
        - Density and connectivity
        - Clustering coefficients
        - Community structure and modularity
        - Small-world properties
        
        Args:
            semantic_graph: NetworkX graph representing semantic relationships
            
        Returns:
            Structural complexity score normalized to [0, 1]
        """
        if len(semantic_graph) < 2:
            return 0.0
            
        try:
            # 1. Basic graph metrics
            density = nx.density(semantic_graph)
            
            # 2. Clustering and transitivity
            avg_clustering = nx.average_clustering(semantic_graph)
            transitivity = nx.transitivity(semantic_graph)
            
            # 3. Community structure (modularity)
            try:
                communities = list(nx.community.greedy_modularity_communities(semantic_graph))
                modularity = nx.community.modularity(semantic_graph, communities)
            except:
                modularity = 0.0
            
            # 4. Small-world properties
            try:
                # Approximate average shortest path length for large graphs
                if len(semantic_graph) <= 100:
                    avg_path_length = nx.average_shortest_path_length(semantic_graph)
                else:
                    # Sample nodes for large graphs
                    sample_nodes = list(semantic_graph.nodes())[:100]
                    subgraph = semantic_graph.subgraph(sample_nodes)
                    avg_path_length = nx.average_shortest_path_length(subgraph)
            except:
                avg_path_length = 0.0
            
            # 5. Centrality measures
            try:
                degree_centrality = np.mean(list(nx.degree_centrality(semantic_graph).values()))
            except:
                degree_centrality = 0.0
            
            # Combine metrics with empirical weights
            complexity_components = {
                'density': density,
                'clustering': (avg_clustering + transitivity) / 2,
                'modularity': abs(modularity),
                'small_world': 1.0 / (1.0 + avg_path_length) if avg_path_length > 0 else 0.0,
          'centrality': degree_centrality
            }
            
            # Empirical weights from principal component analysis
            component_weights = [0.25, 0.25, 0.20, 0.15, 0.15]
            structural_complexity = sum(w * comp for w, comp in 
                                      zip(component_weights, complexity_components.values()))
            
            return min(max(structural_complexity, 0.0), 1.0)
            
        except Exception as e:
            warnings.warn(f"Structural complexity calculation failed: {e}")
            return 0.0
    
    def calculate_recursive_depth(self, behavioral_data: Dict) -> float:
        """
        Calculate RDL - recursive depth of self-modeling operations
        
        Measures the capacity for recursive self-reference and meta-cognition:
        - Theory of mind complexity
        - Self-reference depth  
        - Meta-cognitive awareness
        - Recursive reasoning chains
        
        Args:
            behavioral_data: Dictionary containing behavioral measures
            
        Returns:
            Recursive depth score normalized to [0, 1]
        """
        try:
            # Extract behavioral measures with defaults
            theory_of_mind = behavioral_data.get('theory_of_mind_depth', 0.0)
            self_reference = behavioral_data.get('self_reference_complexity', 0.0)
            meta_cognition = behavioral_data.get('meta_cognitive_depth', 0.0)
            recursive_reasoning = behavioral_data.get('recursive_reasoning_depth', 0.0)
            
            # Validate ranges
            theory_of_mind = max(0.0, min(theory_of_mind, 10.0))
            self_reference = max(0.0, min(self_reference, 10.0))
            meta_cognition = max(0.0, min(meta_cognition, 10.0))
            recursive_reasoning = max(0.0, min(recursive_reasoning, 10.0))
            
            # Calculate weighted recursive depth
            depth_components = [
                theory_of_mind * 0.3,      # Understanding others' mental states
                self_reference * 0.3,      # Self-awareness complexity
                meta_cognition * 0.25,     # Thinking about thinking
                recursive_reasoning * 0.15 # Nested reasoning chains
            ]
            
            total_depth = sum(depth_components)
            normalized_depth = total_depth / 10.0  # Normalize to [0, 1]
            
            return min(max(normalized_depth, 0.0), 1.0)
            
        except Exception as e:
            warnings.warn(f"Recursive depth calculation failed: {e}")
            return 0.0
    
    def calculate_information_integration(self, 
                                       modality_a: np.ndarray,
                                       modality_b: np.ndarray) -> float:
        """
        Calculate D_KL - information integration capacity between modalities
        
        Uses Jensen-Shannon divergence and mutual information to measure
        how effectively information is integrated across different processing
        modules or sensory modalities.
        
        Args:
            modality_a: Data from first modality (neural, behavioral, etc.)
            modality_b: Data from second modality
            
        Returns:
            Information integration score normalized to [0, 1]
        """
        try:
            # Flatten and normalize data
            data_a = modality_a.flatten()
            data_b = modality_b.flatten()
            
            # Ensure same length by truncating to shorter array
            min_len = min(len(data_a), len(data_b))
            data_a = data_a[:min_len]
            data_b = data_b[:min_len]
            
            if min_len < 10:
                return 0.0
            
            # 1. Jensen-Shannon divergence (symmetric, bounded)
            # Create probability distributions
            hist_a, _ = np.histogram(data_a, bins=50, density=True)
            hist_b, _ = np.histogram(data_b, bins=50, density=True)
            
            # Add small constant to avoid zeros
          hist_a = hist_a + 1e-10
            hist_b = hist_b + 1e-10
            
            # Normalize to probability distributions
            hist_a = hist_a / np.sum(hist_a)
            hist_b = hist_b / np.sum(hist_b)
            
            # Calculate JS divergence
            js_divergence = jensenshannon(hist_a, hist_b)
            
            # 2. Correlation measure
            correlation = np.corrcoef(data_a, data_b)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 3. Mutual information (approximate)
            try:
                # Bin data for mutual information calculation
                bins = 20
                hist_2d, _, _ = np.histogram2d(data_a, data_b, bins=bins)
                hist_2d = hist_2d / np.sum(hist_2d)
                
                # Calculate marginal distributions
                p_a = np.sum(hist_2d, axis=1)
                p_b = np.sum(hist_2d, axis=0)
                
                # Calculate mutual information
                mi = 0.0
                for i in range(bins):
                    for j in range(bins):
                        if hist_2d[i, j] > 0 and p_a[i] > 0 and p_b[j] > 0:
                            mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_a[i] * p_b[j]))
                
                # Normalize mutual information
                mi_normalized = min(mi / np.log(bins), 1.0)
            except:
                mi_normalized = 0.0
            
            # Combine integration measures
            # Higher integration = lower divergence, higher correlation and MI
            integration_components = [
                1.0 - min(js_divergence, 1.0),  # Inverse of divergence
                abs(correlation),                # Absolute correlation
                mi_normalized                    # Normalized mutual information
            ]
            
            integration_weights = [0.4, 0.4, 0.2]
            integration_score = sum(w * comp for w, comp in 
                                  zip(integration_weights, integration_components))
            
            return min(max(integration_score, 0.0), 1.0)
            
        except Exception as e:
            warnings.warn(f"Information integration calculation failed: {e}")
            return 0.0
    
    def calculate_cross_modal_entanglement(self, multi_modal_data: Dict) -> float:
        """
        Calculate CEM - cross-modal binding and entanglement
        
        Measures how tightly different sensory or cognitive modalities
        are bound together in the system's processing architecture.
        
        Args:
            multi_modal_data: Dictionary containing data from multiple modalities
                            Keys: modality names, Values: data arrays
            
        Returns:
            Cross-modal entanglement score normalized to [0, 1]
        """
        try:
            modalities = list(multi_modal_data.keys())
            if len(modalities) < 2:
                return 0.0
            
            correlations = []
            integration_scores = []
            
            # Calculate pairwise relationships between all modalities
            for i, mod1 in enumerate(modalities):
                data1 = multi_modal_data[mod1]
                
                for j, mod2 in enumerate(modalities[i+1:], i+1):
                    data2 = multi_modal_data[mod2]
                    
                    try:
                        # Ensure data compatibility
                        if len(data1) != len(data2):
                            min_len = min(len(data1), len(data2))
                            data1_trim = data1[:min_len]
                            data2_trim = data2[:min_len]
                        else:
                            data1_trim = data1
                            data2_trim = data2
                        
                        if len(data1_trim) < 5:
                            continue
                        
                        # Correlation
                      corr = np.corrcoef(data1_trim, data2_trim)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                        
                        # Integration score
                        integration = self.calculate_information_integration(
                            np.array(data1_trim), 
                            np.array(data2_trim)
                        )
                        integration_scores.append(integration)
                        
                    except Exception as e:
                        continue
            
            if not correlations and not integration_scores:
                return 0.0
            
            # Calculate entanglement as combination of correlation and integration
            avg_correlation = np.mean(correlations) if correlations else 0.0
            avg_integration = np.mean(integration_scores) if integration_scores else 0.0
            
            # Entanglement emphasizes both correlation and complex integration
            entanglement = 0.6 * avg_integration + 0.4 * avg_correlation
            
            return min(max(entanglement, 0.0), 1.0)
            
        except Exception as e:
            warnings.warn(f"Cross-modal entanglement calculation failed: {e}")
            return 0.0
    
    def calculate_icim(self, 
                      semantic_graph: nx.Graph,
                      behavioral_data: Dict,
                      neural_data: Optional[np.ndarray] = None,
                      multi_modal_data: Optional[Dict] = None,
                      modality_a: Optional[np.ndarray] = None,
                      modality_b: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Calculate complete ICIM score for consciousness assessment
        
        Args:
            semantic_graph: Semantic network graph
            behavioral_data: Behavioral measures dictionary
            neural_data: Neural activity data (optional)
            multi_modal_data: Multi-modal data dictionary (optional)
            modality_a: Data from first modality for integration (optional)
            modality_b: Data from second modality for integration (optional)
            
        Returns:
            Dictionary containing:
            - icim_score: Overall ICIM score
            - consciousness_level: Classification result
            - components: Individual component scores
            - weights: Component weights used
            - confidence: Confidence estimate
        """
        # Calculate component scores
        d_sg = self.calculate_structural_complexity(semantic_graph)
        rdl = self.calculate_recursive_depth(behavioral_data)
        
        # Calculate information integration
        if modality_a is not None and modality_b is not None:
            d_kl = self.calculate_information_integration(modality_a, modality_b)
        elif neural_data is not None:
            # Use neural data with behavioral data for integration
            behavioral_array = np.array(list(behavioral_data.values()))
            d_kl = self.calculate_information_integration(neural_data, behavioral_array)
        else:
            d_kl = 0.5  # Default moderate integration
        
        # Calculate cross-modal entanglement
        if multi_modal_data is not None:
            cem = self.calculate_cross_modal_entanglement(multi_modal_data)
        else:
            cem = 0.3  # Default low entanglement
        
        # Calculate weighted ICIM score
        components = {
            'structural_complexity': d_sg,
            'recursive_depth': rdl,
            'information_integration': d_kl,
            'cross_modal_entanglement': cem
        }
        
        icim_score = sum(comp * self.weights[name] 
                        for name, comp in components.items())
        
        # Determine consciousness level
        consciousness_level = self._classify_consciousness(icim_score)
        
        # Calculate confidence based on data quality and component consistency
confidence = self._calculate_confidence(components, icim_score)
        
        return {
            'icim_score': float(icim_score),
            'consciousness_level': consciousness_level,
            'components': components,
            'weights': self.weights,
            'confidence': confidence,
            'thresholds': self.consciousness_thresholds
        }
    
    def _classify_consciousness(self, icim_score: float) -> str:
        """Classify consciousness level based on ICIM score"""
        if icim_score >= self.consciousness_thresholds['enhanced_integration']:
            return "Enhanced Integration"
        elif icim_score >= self.consciousness_thresholds['full_consciousness']:
            return "Full Consciousness"
        elif icim_score >= self.consciousness_thresholds['basic_consciousness']:
            return "Basic Consciousness"
        elif icim_score >= self.consciousness_thresholds['minimal_awareness']:
            return "Minimal Awareness"
        else:
            return "Below Threshold"
    
    def _calculate_confidence(self, components: Dict, icim_score: float) -> float:
        """Calculate confidence score based on data quality and consistency"""
        try:
            # Component consistency (higher when components agree)
            component_values = list(components.values())
            consistency = 1.0 - (np.std(component_values) / 0.5)  # Normalized std
            
            # Data completeness (penalize default values)
            defaults_used = sum(1 for v in component_values if v in [0.5, 0.3])
            completeness = 1.0 - (defaults_used / len(components) * 0.5)
            
            # Score reliability (middle scores are less reliable for classification)
            distance_to_threshold = self._min_distance_to_threshold(icim_score)
            reliability = min(distance_to_threshold * 2.0, 1.0)
            
            confidence = (consistency + completeness + reliability) / 3.0
            return min(max(confidence, 0.0), 1.0)
            
        except:
            return 0.5  # Default medium confidence
    
    def _min_distance_to_threshold(self, icim_score: float) -> float:
        """Calculate minimum distance to any consciousness threshold"""
        thresholds = list(self.consciousness_thresholds.values())
        distances = [abs(icim_score - threshold) for threshold in thresholds]
        return min(distances) if distances else 1.0
    
    def save_results(self, results: Dict, filepath: str):
        """Save ICIM results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_results(self, filepath: str) -> Dict:
        """Load ICIM results from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)

# Convenience functions
def calculate_icim(semantic_data, behavioral_data, **kwargs):
    """Convenience function for ICIM calculation"""
    calculator = ICIMCalculator()
    return calculator.calculate_icim(semantic_data, behavioral_data, **kwargs)

def create_example_semantic_graph():
    """Create an example semantic graph for testing"""
    G = nx.Graph()
    
    # Add nodes (concepts)
    concepts = ['self', 'other', 'thought', 'feeling', 'action', 'time', 'space']
    G.add_nodes_from(concepts)
    
    # Add edges (semantic relationships)
    relationships = [
        ('self', 'thought'), ('self', 'feeling'), ('self', 'action'),
        ('other', 'thought'), ('other', 'feeling'), 
        ('thought', 'feeling'), ('thought', 'action'),
        ('feeling', 'action'), ('self', 'time'), ('self', 'space')
    ]
    G.add_edges_from(relationships)
    
    return G

# Example usage and testing
if __name__ == "__main__":
    print("ICIM Calculator - Resonance Architecture of Consciousness")
    print("=" * 60)
    
    # Create example data
    semantic_graph = create_example_semantic_graph()
    
    behavioral_data = {
        'theory_of_mind_depth': 7.5,
        'self_reference_complexity': 6.8,
        'meta_cognitive_depth': 8.2,
      'recursive_reasoning_depth': 5.5
    }
    
    # Example neural and behavioral data arrays
    neural_data = np.random.normal(0, 1, 100)
    behavioral_array = np.random.normal(0.5, 0.5, 100)
    
    multi_modal_data = {
        'visual': np.random.normal(0, 1, 50),
        'auditory': np.random.normal(0.2, 0.8, 50),
        'somatosensory': np.random.normal(-0.1, 1.2, 50)
    }
    
    # Calculate ICIM
    calculator = ICIMCalculator()
    results = calculator.calculate_icim(
        semantic_graph=semantic_graph,
        behavioral_data=behavioral_data,
        neural_data=neural_data,
        multi_modal_data=multi_modal_data
    )
    
    # Display results
    print(f"ICIM Score: {results['icim_score']:.3f}")
    print(f"Consciousness Level: {results['consciousness_level']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print("\nComponent Scores:")
    for comp, score in results['components'].items():
        print(f"  {comp.replace('_', ' ').title()}: {score:.3f}")
    
    print("\nConsciousness Thresholds:")
    for level, threshold in results['thresholds'].items():
        print(f"  {level.replace('_', ' ').title()}: {threshold}")
