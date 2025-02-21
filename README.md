# Hybrid Prompt Injection Detection System (Early Stage)

This project is in its very early stages of development.  It aims to implement a hybrid approach to prompt injection detection, combining the potential strengths of rule-based (regular expressions) and machine learning techniques. Prompt injection is a critical security vulnerability in Large Language Models (LLMs), and this project seeks to explore and develop effective detection strategies.

## Current Status

This project is in its initial exploratory phase.  The core ideas are being prototyped, and the initial focus is on establishing a solid foundation for future development.  While some basic functionality is implemented, significant work remains to achieve the project's long-term goals.

## Overview

The long-term vision for this system is to employ a hybrid approach, combining:

1.  **Rule-Based Detection (Regular Expressions):**  Initial work is focused on building a library of regular expression patterns to identify common indicators of prompt injection attempts.  These patterns will be categorized and potentially weighted based on the severity of the potential attack.

2.  **Machine Learning:**  The plan is to integrate machine learning techniques to detect more complex and nuanced attacks that might evade rule-based detection.  The specific model and features are still under investigation.

## Current Implementation

Currently, the project includes:

*   A basic set of regular expression patterns (see `suspicious_phrases` in the code).  These are under active development and are not yet comprehensive.
*   A preliminary structure for categorizing and weighting regex patterns (see `pattern_categories` and `category_weights` in the code).
*   A basic framework for combining regex matches and scores (see `detect_with_regex` and related functions).

Machine learning integration is in the planning stages, and no ML models have been trained or integrated yet.

## Next Steps / Future Work

The immediate next steps for this project include:

*   Expanding the library of regular expression patterns and improving their accuracy.
*   Developing a more robust system for categorizing and weighting regex patterns.
*   Investigating suitable machine learning models and features for integration.
*   Setting up a training and evaluation pipeline.

Longer-term goals include:

*   Implementing compound pattern detection.
*   Exploring semantic and contextual analysis techniques.
*   Building a comprehensive test dataset.

