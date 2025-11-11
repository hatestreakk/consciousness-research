"""
Data Analysis Toolkit for Consciousness Research
Statistical analysis and validation tools for RAC theory

Includes:
- Statistical validation of ICIM metric
- Cross-system comparative analysis
- Model parameter estimation
- Hypothesis testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from pathlib import Path

class ConsciousnessDataAnalyzer:
    """
    Comprehensive data analysis for consciousness research data
    
    Handles data from multiple sources:
    - Human behavioral experiments
    - AI system assessments  
    - Cross-species comparisons
    - Clinical populations
    """
    
    def __init__(self):
        self.results = {}
        self.validation_metrics = {}
        
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load consciousness research dataset
        
        Expected columns:
        - system_id: Unique identifier
        - system_type: 'human', 'ai', 'animal', 'clinical'
        - icim_score: Computed ICIM value
        - consciousness_level: Ground truth or expert rating
        - behavioral_correlation: Behavioral validation score
        - entropy_metrics: Various entropy measures
        """
        try:
            df = pd.read_csv(filepath)
            required_columns = ['system_id', 'system_type', 'icim_score']
            
            if not all(col in df.columns for col in required_columns):
                warnings.warn("Dataset missing required columns")
                
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
    
    def validate_icim_metric(self, 
                           df: pd.DataFrame,
                           ground_truth_col: str = 'consciousness_level') -> Dict[str, Any]:
        """
        Validate ICIM metric against ground truth consciousness assessments
        
        Args:
            df: DataFrame with ICIM scores and ground truth
            ground_truth_col: Column containing ground truth labels
            
        Returns:
            Dictionary with validation metrics
        """
        validation_results = {}
        
        try:
            # Convert categorical consciousness levels to numerical scores
            level_mapping = {
                'minimal_awareness': 1.0,
                'basic_consciousness': 2.0, 
                'full_consciousness': 3.0,
                'enhanced_integration': 4.0
            }
            
            # Create numerical ground truth
            df_clean = df.dropna(subset=['icim_score', ground_truth_col])
            ground_truth_numeric = df_clean[ground_truth_col].map(level_mapping)
            
            # Correlation analysis
            pearson_corr, p_value = stats.pearsonr(df_clean['icim_score'], ground_truth_numeric)
            spearman_corr, spearman_p = stats.spearmanr(df_clean['icim_score'], ground_truth_numeric)
            
            # Classification accuracy using ICIM thresholds
            def classify_consciousness(icim_score: float) -> float:
                if icim_score >= 3.2: return 4.0
                elif icim_score >= 2.4: return 3.0
                elif icim_score >= 1.8: return 2.0
                elif icim_score >= 1.2: return 1.0
                else: return 0.0
            
            predicted_levels = df_clean['icim_score'].apply(classify_consciousness)
            accuracy = accuracy_score(ground_truth_numeric, predicted_levels)
            
            # Confidence intervals
            n = len(df_clean)
            accuracy_ci = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
            
            validation_results = {
                'sample_size': n,
                'pearson_correlation': pearson_corr,
          'pearson_p_value': p_value,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'classification_accuracy': accuracy,
                'accuracy_confidence_interval': (accuracy - accuracy_ci, accuracy + accuracy_ci),
                'r_squared': pearson_corr ** 2
            }
            
            self.validation_metrics['icim_validation'] = validation_results
            
        except Exception as e:
            warnings.warn(f"ICIM validation failed: {e}")
            
        return validation_results
    
    def cross_system_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare consciousness metrics across different system types
        
        Args:
            df: DataFrame with system_type and ICIM scores
            
        Returns:
            Dictionary with comparative statistics
        """
        comparison_results = {}
        
        try:
            system_types = df['system_type'].unique()
            comparison_data = {}
            
            for system_type in system_types:
                system_data = df[df['system_type'] == system_type]['icim_score'].dropna()
                if len(system_data) > 0:
                    comparison_data[system_type] = {
                        'n': len(system_data),
                        'mean_icim': system_data.mean(),
                        'std_icim': system_data.std(),
                        'median_icim': system_data.median(),
                        'min_icim': system_data.min(),
                        'max_icim': system_data.max()
                    }
            
            # Statistical tests between groups
            anova_results = {}
            if len(comparison_data) >= 2:
                groups = [df[df['system_type'] == st]['icim_score'].dropna() 
                         for st in system_types if len(df[df['system_type'] == st]) >= 5]
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results = {'f_statistic': f_stat, 'p_value': p_value}
            
            comparison_results = {
                'system_comparisons': comparison_data,
                'anova_results': anova_results,
                'total_systems': len(df),
                'unique_system_types': list(system_types)
            }
            
            self.results['cross_system_comparison'] = comparison_results
            
        except Exception as e:
            warnings.warn(f"Cross-system comparison failed: {e}")
            
        return comparison_results
    
    def estimate_entropy_parameters(self, 
                                  entropy_data: np.ndarray,
                                  disruption_data: np.ndarray,
                                  time_data: np.ndarray) -> Dict[str, float]:
        """
        Estimate parameters for entropy dynamics equation from empirical data
        
        Args:
            entropy_data: Time series of entropy measurements
            disruption_data: Time series of disruption intensities
            time_data: Corresponding time points
            
        Returns:
            Dictionary with estimated parameters and confidence intervals
        """
        try:
            # Ensure data is properly formatted
            if len(entropy_data) != len(disruption_data) != len(time_data):
                raise ValueError("All input arrays must have same length")
            
            # Calculate derivatives using finite differences
            dt = np.diff(time_data)
            dS_dt = np.diff(entropy_data) / dt
            
            # Prepare data for curve fitting (remove last point to match derivative)
            S = entropy_data[:-1]
            I_d = disruption_data[:-1]
            
            # Define the model function for curve fitting
            def entropy_model(s, alpha, beta, s_max, s_0):
                return alpha * I_d * (1 - s/s_max) - beta * (s - s_0)
              # Initial parameter guesses
            initial_guess = [0.5, 0.3, 8.0, 1.0]
            
            # Perform curve fitting
            bounds = ([0, 0, 1, 0], [10, 10, 20, 5])  # Reasonable bounds
            popt, pcov = curve_fit(entropy_model, S, dS_dt, 
                                 p0=initial_guess, bounds=bounds)
            
            # Calculate parameter uncertainties
            perr = np.sqrt(np.diag(pcov))
            
            parameter_results = {
                'alpha_estimate': popt[0],
                'beta_estimate': popt[1],
                's_max_estimate': popt[2],
                's_0_estimate': popt[3],
                'alpha_std': perr[0],
                'beta_std': perr[1],
                's_max_std': perr[2],
                's_0_std': perr[3],
                'r_squared': self._calculate_r_squared(dS_dt, entropy_model(S, *popt))
            }
            
            self.results['parameter_estimation'] = parameter_results
            
            return parameter_results
            
        except Exception as e:
            warnings.warn(f"Parameter estimation failed: {e}")
            return {}
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared for model fit"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def behavioral_correlation_analysis(self, 
                                      df: pd.DataFrame,
                                      behavioral_columns: List[str]) -> Dict[str, Any]:
        """
        Analyze correlations between ICIM scores and behavioral measures
        
        Args:
            df: DataFrame with ICIM scores and behavioral measures
            behavioral_columns: List of column names for behavioral data
            
        Returns:
            Dictionary with correlation analysis results
        """
        correlation_results = {}
        
        try:
            correlations = {}
            p_values = {}
            
            for behavior_col in behavioral_columns:
                if behavior_col in df.columns:
                    # Clean data
                    clean_data = df[['icim_score', behavior_col]].dropna()
                    
                    if len(clean_data) >= 10:  # Minimum sample size
                        corr, p_val = stats.pearsonr(clean_data['icim_score'], 
                                                   clean_data[behavior_col])
                        correlations[behavior_col] = corr
                        p_values[behavior_col] = p_val
            
            # Multiple comparison correction
            p_values_corrected = self._correct_multiple_comparisons(p_values)
            
            correlation_results = {
                'pearson_correlations': correlations,
                'p_values_uncorrected': p_values,
                'p_values_corrected': p_values_corrected,
                'significant_correlations': {
                    col: corr for col, corr in correlations.items()
                    if p_values_corrected[col] < 0.05
                }
            }
            
            self.results['behavioral_correlations'] = correlation_results
            
        except Exception as e:
            warnings.warn(f"Behavioral correlation analysis failed: {e}")
            
        return correlation_results
    
    def _correct_multiple_comparisons(self, p_values: Dict[str, float]) -> Dict[str, float]:
        """Apply Bonferroni correction for multiple comparisons"""
        n_tests = len(p_values)
        corrected = {}
        
        for col, p_val in p_values.items():
            corrected[col] = min(p_val * n_tests, 1.0)
            
        return corrected
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive analysis report for consciousness research data
        
        Args:
        df: DataFrame with consciousness research data
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "CONSCIOUSNESS RESEARCH DATA ANALYSIS REPORT",
            "=" * 50,
            f"Total systems analyzed: {len(df)}",
            f"System types: {', '.join(df['system_type'].unique())}",
            ""
        ]
        
        # Basic statistics
        icim_stats = df['icim_score'].describe()
        report_lines.extend([
            "ICIM SCORE STATISTICS:",
            f"  Mean: {icim_stats['mean']:.3f}",
            f"  Std: {icim_stats['std']:.3f}",
            f"  Min: {icim_stats['min']:.3f}",
            f"  Max: {icim_stats['max']:.3f}",
            ""
        ])
        
        # Consciousness level distribution
        if 'consciousness_level' in df.columns:
            level_counts = df['consciousness_level'].value_counts()
            report_lines.append("CONSCIOUSNESS LEVEL DISTRIBUTION:")
            for level, count in level_counts.items():
                percentage = (count / len(df)) * 100
                report_lines.append(f"  {level}: {count} systems ({percentage:.1f}%)")
            report_lines.append("")
        
        # Validation results
        if 'consciousness_level' in df.columns:
            validation = self.validate_icim_metric(df)
            report_lines.extend([
                "ICIM METRIC VALIDATION:",
                f"  Pearson correlation: {validation.get('pearson_correlation', 0):.3f}",
                f"  Classification accuracy: {validation.get('classification_accuracy', 0):.3f}",
                f"  R-squared: {validation.get('r_squared', 0):.3f}",
                ""
            ])
        
        # Cross-system comparison
        comparison = self.cross_system_comparison(df)
        report_lines.append("CROSS-SYSTEM COMPARISON:")
        for system_type, stats in comparison.get('system_comparisons', {}).items():
            report_lines.append(
                f"  {system_type}: n={stats['n']}, mean ICIM={stats['mean_icim']:.3f}"
            )
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def create_visualizations(self, df: pd.DataFrame, save_dir: str = "figures"):
        """
        Create comprehensive visualizations for consciousness research data
        
        Args:
            df: DataFrame with research data
            save_dir: Directory to save figures
        """
        Path(save_dir).mkdir(exist_ok=True)
        
        # 1. ICIM distribution by system type
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='system_type', y='icim_score')
        plt.title('ICIM Score Distribution by System Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/icim_by_system_type.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap (if multiple behavioral measures)
        behavioral_cols = [col for col in df.columns if col.startswith('behavioral_')]
        if len(behavioral_cols) >= 2:
            plt.figure(figsize=(8, 6))
            correlation_matrix = df[behavioral_cols + ['icim_score']].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Behavioral Measures Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Consciousness level distribution
        if 'consciousness_level' in df.columns:
            plt.figure(figsize=(8, 6))
            level_order = ['minimal_awareness', 'basic_consciousness', 
                          'full_consciousness', 'enhanced_integration']
            sns.countplot(data=df, x='consciousness_level', order=level_order)
            plt.title('Consciousness Level Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/consciousness_levels.
          png", dpi=300, bbox_inches='tight')
            plt.close()

# Example dataset generator for testing
def generate_sample_dataset(n_systems: int = 200) -> pd.DataFrame:
    """
    Generate sample dataset for testing and demonstration
    
    Args:
        n_systems: Number of systems to generate
        
    Returns:
        DataFrame with sample consciousness research data
    """
    np.random.seed(42)  # For reproducible results
    
    system_types = ['human', 'ai', 'primate', 'cetacean', 'avian']
    consciousness_levels = ['minimal_awareness', 'basic_consciousness', 
                           'full_consciousness', 'enhanced_integration']
    
    data = []
    
    for i in range(n_systems):
        system_type = np.random.choice(system_types)
        
        # Generate realistic ICIM scores based on system type
        if system_type == 'human':
            icim_mean, icim_std = 2.5, 0.4
        elif system_type == 'ai':
            icim_mean, icim_std = 1.8, 0.6
        elif system_type == 'primate':
            icim_mean, icim_std = 2.1, 0.3
        elif system_type == 'cetacean':
            icim_mean, icim_std = 2.3, 0.3
        else:  # avian
            icim_mean, icim_std = 1.6, 0.4
        
        icim_score = np.random.normal(icim_mean, icim_std)
        icim_score = max(0.5, min(4.0, icim_score))  # Clamp to reasonable range
        
        # Determine consciousness level based on ICIM score
        if icim_score >= 3.2:
            level = 'enhanced_integration'
        elif icim_score >= 2.4:
            level = 'full_consciousness'
        elif icim_score >= 1.8:
            level = 'basic_consciousness'
        elif icim_score >= 1.2:
            level = 'minimal_awareness'
        else:
            level = 'minimal_awareness'
        
        # Generate behavioral correlation (typically high for accurate systems)
        behavioral_corr = np.random.normal(0.8, 0.15)
        behavioral_corr = max(0.3, min(1.0, behavioral_corr))
        
        system_data = {
            'system_id': f"system_{i:04d}",
            'system_type': system_type,
            'icim_score': icim_score,
            'consciousness_level': level,
            'behavioral_correlation': behavioral_corr,
            'entropy_variance': np.random.normal(0.5, 0.2),
            'recursive_depth': np.random.normal(0.6, 0.2),
            'information_integration': np.random.normal(0.7, 0.15)
        }
        
        data.append(system_data)
    
    return pd.DataFrame(data)

# Example usage
if name == "__main__":
    print("Consciousness Research Data Analysis Toolkit")
    print("=" * 50)
    
    # Generate sample data
    sample_data = generate_sample_dataset(150)
    print(f"Generated sample dataset with {len(sample_data)} systems")
    
    # Initialize analyzer
    analyzer = ConsciousnessDataAnalyzer()
    
    # Run comprehensive analysis
    report = analyzer.generate_comprehensive_report(sample_data)
    print("\n" + report)
    
    # Validate ICIM metric
    validation_results = analyzer.validate_icim_metric(sample_data)
    print("\nICIM Validation Results:")
    for metric, value in validation_results.items():
        print(f"  {metric}: {value}")
    
    # Cross-system comparison
    comparison_results = analyzer.cross_system_comparison(sample_data)
    print("\nCross-System Comparison:")
    for system_type, stats in comparison_results['system_comparisons'].items():
        print(f"  {system_type}: ICIM = {stats['mean_icim']:.3f} ± {stats['std_icim']:.3f}")
    
    # Create visualizations
    analyzer.create_visualizations(sample_data, "analysis_figures")
    print("\nVisualizations saved to 'analysis_figures' directory")
    
    print("\nAnalysis completed successfully!")
