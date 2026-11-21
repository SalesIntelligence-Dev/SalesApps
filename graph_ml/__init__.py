"""
GraphReady – Graph-ML-Readiness-Plattform (/graph_ml)
Vollständig isoliert von den übrigen SalesApps-Routen.
Aufbau in Schritten laut GRAPH_ML_PROMPTS.md.
"""
from .routes import graph_bp

__all__ = ["graph_bp"]
