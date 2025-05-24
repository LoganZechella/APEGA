"""
RAG Quality Metrics for APEGA.
Comprehensive evaluation metrics for retrieval quality assessment.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
import re

from src.models.data_models import RetrievedContext, SynthesizedKnowledge
from tests.rag_test_queries import TestQuery

@dataclass
class RetrievalQualityReport:
    query: str
    query_type: str
    primary_domain: str
    retrieved_chunks: int
    relevance_scores: List[float]
    concept_coverage: float
    domain_coverage: float
    source_diversity: float
    avg_chunk_length: float
    synthesis_quality: float
    overall_score: float
    recommendations: List[str]

class RAGQualityMetrics:
    """Comprehensive quality metrics for RAG retrieval evaluation."""
    
    def __init__(self):
        self.concept_keywords = {
            "valuation": ["value", "valuation", "worth", "assessment", "appraisal", "price"],
            "licensing": ["license", "licensing", "agreement", "royalty", "permission"],
            "patent": ["patent", "invention", "claims", "prior art", "novelty"],
            "trademark": ["trademark", "brand", "mark", "logo", "symbol"],
            "copyright": ["copyright", "authorship", "original work", "reproduction"],
            "negotiation": ["negotiate", "bargain", "deal", "terms", "agreement"],
            "commercialization": ["commercialize", "market", "product", "launch"],
            "strategy": ["strategy", "strategic", "plan", "approach", "method"],
            "portfolio": ["portfolio", "collection", "assets", "management"],
            "due diligence": ["due diligence", "audit", "review", "investigation"],
            "market analysis": ["market analysis", "competitive", "landscape", "research"],
            "freedom to operate": ["freedom to operate", "FTO", "clearance"],
            "technology transfer": ["technology transfer", "knowledge transfer", "implementation"],
            "compliance": ["compliance", "monitoring", "reporting", "audit"],
            "dispute": ["dispute", "disagreement", "conflict", "breach"]
        }
    
    def evaluate_retrieval_quality(
        self,
        test_query: TestQuery,
        retrieved_contexts: List[RetrievedContext],
        synthesized_knowledge: SynthesizedKnowledge
    ) -> RetrievalQualityReport:
        """
        Comprehensive evaluation of retrieval quality for a test query.
        
        Args:
            test_query: The test query with expected characteristics
            retrieved_contexts: Contexts retrieved by RAG
            synthesized_knowledge: Synthesized knowledge from DeepAnalyzer
            
        Returns:
            Detailed quality report
        """
        # 1. Basic retrieval metrics
        retrieved_chunks = len(retrieved_contexts)
        relevance_scores = [ctx.rerank_score or ctx.initial_score for ctx in retrieved_contexts]
        
        # 2. Concept coverage analysis
        concept_coverage = self._calculate_concept_coverage(
            retrieved_contexts, 
            test_query.expected_concepts
        )
        
        # 3. Domain coverage analysis
        domain_coverage = self._calculate_domain_coverage(
            retrieved_contexts,
            test_query.primary_domain,
            test_query.secondary_domains
        )
        
        # 4. Source diversity
        source_diversity = self._calculate_source_diversity(retrieved_contexts)
        
        # 5. Content quality metrics
        avg_chunk_length = np.mean([len(ctx.text.split()) for ctx in retrieved_contexts]) if retrieved_contexts else 0
        
        # 6. Synthesis quality
        synthesis_quality = self._evaluate_synthesis_quality(
            synthesized_knowledge,
            test_query.expected_concepts
        )
        
        # 7. Overall score calculation
        overall_score = self._calculate_overall_score(
            concept_coverage, domain_coverage, source_diversity, 
            synthesis_quality, relevance_scores
        )
        
        # 8. Generate recommendations
        recommendations = self._generate_recommendations(
            test_query, concept_coverage, domain_coverage, 
            source_diversity, synthesis_quality
        )
        
        return RetrievalQualityReport(
            query=test_query.query_text,
            query_type=test_query.query_type.value,
            primary_domain=test_query.primary_domain.value,
            retrieved_chunks=retrieved_chunks,
            relevance_scores=relevance_scores,
            concept_coverage=concept_coverage,
            domain_coverage=domain_coverage,
            source_diversity=source_diversity,
            avg_chunk_length=avg_chunk_length,
            synthesis_quality=synthesis_quality,
            overall_score=overall_score,
            recommendations=recommendations
        )
    
    def _calculate_concept_coverage(
        self, 
        contexts: List[RetrievedContext], 
        expected_concepts: List[str]
    ) -> float:
        """Calculate how well retrieved contexts cover expected concepts."""
        if not expected_concepts or not contexts:
            return 0.0
        
        all_text = " ".join([ctx.text.lower() for ctx in contexts])
        
        covered_concepts = 0
        for concept in expected_concepts:
            concept_lower = concept.lower()
            # Check for exact match or related keywords
            if concept_lower in all_text:
                covered_concepts += 1
            else:
                # Check for related keywords
                related_keywords = self.concept_keywords.get(concept_lower, [])
                if any(keyword in all_text for keyword in related_keywords):
                    covered_concepts += 0.5  # Partial credit for related concepts
        
        return min(covered_concepts / len(expected_concepts), 1.0)
    
    def _calculate_domain_coverage(
        self,
        contexts: List[RetrievedContext],
        primary_domain: Any,
        secondary_domains: List[Any]
    ) -> float:
        """Calculate coverage of relevant CLP domains."""
        if not contexts:
            return 0.0
        
        domain_counts = Counter()
        for ctx in contexts:
            domain_id = ctx.metadata.get("clp_domain_id", "unknown")
            domain_counts[domain_id] += 1
        
        # Check primary domain coverage
        primary_coverage = 1.0 if domain_counts.get(primary_domain.value, 0) > 0 else 0.0
        
        # Check secondary domain coverage
        secondary_coverage = 0.0
        if secondary_domains:
            covered_secondary = sum(1 for domain in secondary_domains 
                                  if domain_counts.get(domain.value, 0) > 0)
            secondary_coverage = covered_secondary / len(secondary_domains)
        
        # Weighted combination
        return 0.7 * primary_coverage + 0.3 * secondary_coverage
    
    def _calculate_source_diversity(self, contexts: List[RetrievedContext]) -> float:
        """Calculate diversity of source documents."""
        if not contexts:
            return 0.0
        
        document_ids = [ctx.document_id for ctx in contexts]
        unique_documents = len(set(document_ids))
        total_contexts = len(contexts)
        
        return unique_documents / total_contexts
    
    def _evaluate_synthesis_quality(
        self,
        synthesized_knowledge: SynthesizedKnowledge,
        expected_concepts: List[str]
    ) -> float:
        """Evaluate quality of synthesized knowledge."""
        if not synthesized_knowledge.summary:
            return 0.0
        
        summary_text = synthesized_knowledge.summary.lower()
        
        # Check concept coverage in summary
        concept_score = 0.0
        for concept in expected_concepts:
            if concept.lower() in summary_text:
                concept_score += 1.0
        concept_score = min(concept_score / len(expected_concepts), 1.0) if expected_concepts else 0.0
        
        # Check key concepts quality
        key_concepts_score = 0.0
        if synthesized_knowledge.key_concepts:
            key_concepts_score = min(len(synthesized_knowledge.key_concepts) / 3.0, 1.0)  # Expect at least 3 key concepts
        
        # Check potential exam areas
        exam_areas_score = 0.0
        if synthesized_knowledge.potential_exam_areas:
            exam_areas_score = min(len(synthesized_knowledge.potential_exam_areas) / 2.0, 1.0)  # Expect at least 2 exam areas
        
        return np.mean([concept_score, key_concepts_score, exam_areas_score])
    
    def _calculate_overall_score(
        self,
        concept_coverage: float,
        domain_coverage: float,
        source_diversity: float,
        synthesis_quality: float,
        relevance_scores: List[float]
    ) -> float:
        """Calculate weighted overall quality score."""
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Weighted combination of all metrics
        weights = {
            "concept_coverage": 0.25,
            "domain_coverage": 0.20,
            "source_diversity": 0.15,
            "synthesis_quality": 0.25,
            "avg_relevance": 0.15
        }
        
        overall_score = (
            weights["concept_coverage"] * concept_coverage +
            weights["domain_coverage"] * domain_coverage +
            weights["source_diversity"] * source_diversity +
            weights["synthesis_quality"] * synthesis_quality +
            weights["avg_relevance"] * avg_relevance
        )
        
        return overall_score
    
    def _generate_recommendations(
        self,
        test_query: TestQuery,
        concept_coverage: float,
        domain_coverage: float,
        source_diversity: float,
        synthesis_quality: float
    ) -> List[str]:
        """Generate actionable recommendations for improving retrieval."""
        recommendations = []
        
        if concept_coverage < 0.6:
            recommendations.append(f"Low concept coverage ({concept_coverage:.2f}). Consider expanding search terms or improving embeddings for domain-specific concepts.")
        
        if domain_coverage < 0.7:
            recommendations.append(f"Low domain coverage ({domain_coverage:.2f}). Verify document metadata includes correct domain tags.")
        
        if source_diversity < 0.5:
            recommendations.append(f"Low source diversity ({source_diversity:.2f}). Results may be too concentrated in single documents.")
        
        if synthesis_quality < 0.6:
            recommendations.append(f"Low synthesis quality ({synthesis_quality:.2f}). Consider improving DeepAnalyzer prompts or context length.")
        
        if not recommendations:
            recommendations.append("Good retrieval quality across all metrics.")
        
        return recommendations
    
    def calculate_batch_statistics(self, reports: List[RetrievalQualityReport]) -> Dict[str, Any]:
        """Calculate batch statistics across multiple quality reports."""
        if not reports:
            return {}
        
        # Overall statistics
        overall_scores = [r.overall_score for r in reports]
        concept_coverages = [r.concept_coverage for r in reports]
        domain_coverages = [r.domain_coverage for r in reports]
        source_diversities = [r.source_diversity for r in reports]
        synthesis_qualities = [r.synthesis_quality for r in reports]
        
        # Statistics by query type
        type_stats = {}
        for query_type in set(r.query_type for r in reports):
            type_reports = [r for r in reports if r.query_type == query_type]
            type_scores = [r.overall_score for r in type_reports]
            type_stats[query_type] = {
                "count": len(type_reports),
                "avg_score": np.mean(type_scores),
                "std_score": np.std(type_scores),
                "min_score": np.min(type_scores),
                "max_score": np.max(type_scores)
            }
        
        # Statistics by domain
        domain_stats = {}
        for domain in set(r.primary_domain for r in reports):
            domain_reports = [r for r in reports if r.primary_domain == domain]
            domain_scores = [r.overall_score for r in domain_reports]
            domain_stats[domain] = {
                "count": len(domain_reports),
                "avg_score": np.mean(domain_scores),
                "std_score": np.std(domain_scores),
                "min_score": np.min(domain_scores),
                "max_score": np.max(domain_scores)
            }
        
        return {
            "total_queries": len(reports),
            "overall_statistics": {
                "avg_overall_score": np.mean(overall_scores),
                "std_overall_score": np.std(overall_scores),
                "avg_concept_coverage": np.mean(concept_coverages),
                "avg_domain_coverage": np.mean(domain_coverages),
                "avg_source_diversity": np.mean(source_diversities),
                "avg_synthesis_quality": np.mean(synthesis_qualities)
            },
            "score_distribution": {
                "excellent": sum(1 for s in overall_scores if s >= 0.8),
                "good": sum(1 for s in overall_scores if 0.6 <= s < 0.8),
                "fair": sum(1 for s in overall_scores if 0.4 <= s < 0.6),
                "poor": sum(1 for s in overall_scores if s < 0.4)
            },
            "query_type_statistics": type_stats,
            "domain_statistics": domain_stats
        }
    
    def identify_improvement_areas(self, reports: List[RetrievalQualityReport]) -> List[str]:
        """Identify key improvement areas based on quality reports."""
        if not reports:
            return []
        
        improvements = []
        
        # Check for consistently low metrics
        avg_concept_coverage = np.mean([r.concept_coverage for r in reports])
        avg_domain_coverage = np.mean([r.domain_coverage for r in reports])
        avg_source_diversity = np.mean([r.source_diversity for r in reports])
        avg_synthesis_quality = np.mean([r.synthesis_quality for r in reports])
        
        if avg_concept_coverage < 0.6:
            improvements.append("Improve concept coverage through better embeddings or expanded search terms")
        
        if avg_domain_coverage < 0.7:
            improvements.append("Enhance domain coverage by improving document metadata tagging")
        
        if avg_source_diversity < 0.5:
            improvements.append("Increase source diversity to avoid over-concentration in single documents")
        
        if avg_synthesis_quality < 0.6:
            improvements.append("Improve synthesis quality through better prompts or longer context windows")
        
        # Check for domain-specific issues
        domain_scores = {}
        for report in reports:
            domain = report.primary_domain
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(report.overall_score)
        
        for domain, scores in domain_scores.items():
            avg_score = np.mean(scores)
            if avg_score < 0.6:
                improvements.append(f"Domain {domain} shows consistently low performance (avg: {avg_score:.2f})")
        
        return improvements 