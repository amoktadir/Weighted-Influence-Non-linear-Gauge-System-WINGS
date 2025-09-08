# The WINGS method Implementation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red.svg)
![NumPy](https://img.shields.io/badge/numpy-1.24%2B-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-1.5%2B-yellow.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.7%2B-green.svg)
![Graphviz](https://img.shields.io/badge/graphviz-0.20.1-lightgrey.svg)
![python-docx](https://img.shields.io/badge/python--docx-0.8.11-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A Python implementation of the WINGS for multi-criteria decision-making, based on the research paper by Jerzy Michnik (2013). This interactive web application provides an intuitive interface for implementing wings with real-time analysis.

## 📋 Table of Contents
- [Overview WINGS](#Overview-WINGS)

## Overview-WINGS
The WINGS (Weighted Influence Non-linear Gauge System) method is a decision-making tool that helps analyze complex systems with interrelated components. This platform allows you to perform WINGS analysis using either linguistic terms or direct numerical values.

## Step-by-Step Guide
## Configuration (Sidebar)
-Select your input model: Linguistic Terms or Real Data
-Specify the number of components in your system
-For linguistic models, specify the number of experts
-Name each component for easy reference
-Input Data

## Component Strengths: For each component, specify its internal strength/importance
-Influence Matrix: Define how each component influences others
-Use the expandable Linguistic Terms Mapping reference if needed
-Run Analysis

## Click the "Run WINGS Analysis" button to process your inputs
-The system will calculate prominence and relation values
-Interpret Results

-Flowchart: Visual representation of components and their interactions
-Matrices: View the various calculated matrices
-Results: See prominence and relation values for each component
-Classification: Components are classified as Causes or Effects
-Visualization: Graphical representations of the analysis

## Input Models
## Linguistic Terms Model
-Use predefined linguistic terms (e.g., AS, ExS, VSS for strength)
-Multiple experts can provide assessments
-Expert weights can be assigned for weighted averages
-Terms are converted to numerical values for calculation
## Real Data Model
-Input numerical values directly
-Suitable when precise measurements are available
-Single "expert" input mode

## Understanding the Results
-Prominence (r+c): Indicates the overall importance of a component in the system
-Relation (r-c):
-Positive values indicate a component is a Cause (influences others more than it's influenced)
-Negative values indicate a component is an Effect (is influenced more than it influences others)

## Tips for Effective Use
-Start with a small number of components to understand the method
-Use descriptive names for components to make interpretation easier
-For linguistic models, ensure all experts understand the term definitions
-Review the flowchart to verify your inputs match your mental model of the system


## 📚 Citation ##
If you use this implementation in your research, please cite both the original paper and this software:

## Original Paper: 
Michnik, J. (2013). Weighted Influence Non-linear Gauge System (WINGS)–An analysis method for the systems of interrelated components. European Journal of Operational Research, 228(3), 536-544. https://doi.org/10.1016/j.ejor.2013.02.007
## Software Implementation: 
Moktadir, M.A. et.al. (2025). WINGS-Streamlit: A Python implementation of the Weighted Influence Non-linear Gauge Systemd. GitHub repository: [https://github.com/amoktadir/wings-streamlit](https://github.com/amoktadir/Weighted-Influence-Non-linear-Gauge-System-WINGS)
