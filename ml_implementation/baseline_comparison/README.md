# Baseline Comparison Studies

This folder contains the essential baseline method implementations and comparison scripts for the research paper.

## Files Structure

- `baseline_methods.py` - Implementation of 3 baseline anomaly detection methods
- `research_comparison.py` - Comprehensive comparison script (XGBoost, Random Forest, 3 baselines)
- `BASELINE_METHODS_DOCUMENTATION.md` - Detailed documentation and analysis
- `research_comparison_results_*.pkl` - Latest comparison results (automatically generated)
- `research_summary_*.csv` - Results summary table (automatically generated)

## Methods Compared

1. **XGBoost** (Primary method) - 96.43% accuracy
2. **Random Forest** (Ensemble comparison) - 96.19% accuracy
3. **Heuristic Voting** (Rule-based ensemble) - ~62% accuracy
4. **Threshold-Based** (Simple rules) - ~60% accuracy
5. **Isolation Forest** (Unsupervised) - ~70% binary accuracy

## Usage

### Run Complete Comparison (Recommended)

```bash
cd baseline_comparison
python research_comparison.py
```

This will run all 5 methods and generate comprehensive results.

## Results

The comparison demonstrates that:

- XGBoost performs best with 96.43% accuracy
- Random Forest achieves 96.19% accuracy (only 0.24% lower)
- Both ensemble ML methods significantly outperform rule-based approaches (30-35% gap)
- Complex anomaly patterns (memory leaks, service crashes) require ML approaches

## Output Files

After running the comparison, you'll get:

- Detailed console output with accuracy metrics for each method
- `research_comparison_results_*.pkl` - Complete results data
- `research_summary_*.csv` - Summary table for analysis
