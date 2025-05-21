"""
Quality Assurance Module for APEGA.
Contains components for evaluating and validating generated MCQs.
"""

from src.quality_assurance.quality_assurance import (
    QualityAssurance, 
    QACriterion,
    FactualAccuracyCriterion,
    ClarityCriterion,
    DistractorPlausibilityCriterion,
    RelevanceCriterion,
    NoCluesCriterion,
    NoBiasCriterion,
    OverallQualityCriterion
)

__all__ = [
    'QualityAssurance',
    'QACriterion',
    'FactualAccuracyCriterion',
    'ClarityCriterion',
    'DistractorPlausibilityCriterion',
    'RelevanceCriterion',
    'NoCluesCriterion',
    'NoBiasCriterion',
    'OverallQualityCriterion'
]