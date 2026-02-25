# CaD-HSL
This repository contains the source code and demo video for 《CaD-HSL: A Causal-Driven Hypergraph Framework with
GraphRAG for Explainable Technology Trend Prediction》.
## Demo Video
The video shows the four major modules of our system:
### Global Causal Structure
Users can view the global causal graph which shows how individual technologies interact with related fields.By tuning the threshold, users are able to dynamically adjust the edge connections in the graph, thereby effectively controlling the sparsity of the network.
### Local Ego-Network
### Trend Forecasting
### Evaluation Metrics Module
This module provides a quantitative evaluation dashboard to assess the performance of the CaD-HSL model against the baseline (XGB).
- Key Performance Indicators (KPIs): Displays core metrics like MAPE, RMSE, Win Rate, and P-value, with clear improvement indicators.
- Error Comparison: Visualizes the error distribution between the baseline and CaD-HSL, showing where our model performs better.
- Causal Impact Analysis: Analyzes how the number of causal drivers influences model improvement.
- Deep Evaluation Report: Provides a detailed breakdown of all metrics, including stability (error std).
- Performance Leaderboard: Ranks the top 20 technologies by their MAE improvement percentage, highlighting the most impactful use cases.
