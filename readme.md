# Semantic Classification Evaluation

This repository contains tools for developing and evaluating semantic classification model created using semantic router.

## Project Structure

- `semantic_classifier.py` - Contains the clasisifer code. 
- `test/`
  - `test_classifier.py` - Contains the main `SemanticClassifierEvaluator` class for evaluating semantic classification models
  - `__init__.py` - Exports the evaluator class
  - `classifier_data.json` - Sample test data for classification evaluation

## Key Components

### SemanticClassifierEvaluator

The main evaluation class that:
- Loads semantic models and test data
- Runs classifications on test cases
- Generates detailed classification reports
- Identifies misclassified examples
- Provides metrics like precision, recall, and F1-score


