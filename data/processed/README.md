# Processed Data Directory

This directory contains cleaned, processed, and analysis-ready data.

## Processing Pipeline

### Steps Applied:
1. Data Cleaning
   - Handle missing values
   - Remove duplicates
   - Standardize formats

2. Feature Engineering
   - Compute ICIM scores
   - Calculate entropy metrics
   - Create derived features

3. Quality Control
   - Outlier detection and handling
   - Data validation checks
   - Consistency verification

### File Structure

- cleaned_[original_filename].csv - Cleaned versions of raw data
- feature_engineered_[dataset].csv - Data with computed features
- analysis_ready_[study].csv - Final datasets for analysis
- aggregated_results.csv - Combined results across studies

### Data Standards

All processed data should:
- Be in CSV format with UTF-8 encoding
- Include clear column descriptions
- Have consistent data types
- Include provenance information (source, processing steps)

### Metadata

Each processed dataset should have corresponding metadata in /data/metadata/
