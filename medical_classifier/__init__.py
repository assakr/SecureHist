"""
Medical Image Classifier Module

This module provides a deployable version of the Lung and Colon image classifier.
"""

from .inference import MedicalClassifier

__all__ = ["MedicalClassifier"]
