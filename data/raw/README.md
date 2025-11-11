# Raw Data Directory

This directory contains raw, unprocessed data from consciousness research experiments.

## Data Organization

### File Naming Convention
- human_behavioral_[date].csv - Human behavioral experiments
- ai_systems_[model]_[date].csv - AI system assessments  
- animal_studies_[species]_[date].csv - Cross-species data
- clinical_populations_[condition]_[date].csv - Clinical studies

### Data Structure

Each dataset should include:

#### Required Columns:
- subject_id: Unique identifier for each system/participant
- system_type: Type of system ('human', 'ai', 'animal', 'clinical')
- timestamp: Date and time of measurement
- icim_score: Computed ICIM metric value

#### Optional Columns:
- consciousness_level: Expert assessment or ground truth
- behavioral_measures: Various behavioral metrics
- neural_correlates: Neural activity data (if available)
- entropy_metrics: Psychological entropy measurements
- demographic_info: Age, gender, species, etc.

### Data Privacy and Ethics

- Human data must be properly anonymized
- Animal studies must follow ethical guidelines  
- AI system data should respect licensing terms
- Clinical data requires proper consent and HIPAA compliance

### Data Validation

Before analysis, ensure:
1. No personally identifiable information
2. Proper data types and ranges
3. Missing values documented
4. Outliers investigated and justified

### Example Dataset

See example_dataset.csv for expected format and structure.
