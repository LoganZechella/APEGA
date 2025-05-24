"""
RAG Test Queries for APEGA CLP Knowledge Base.
Comprehensive test suite covering all CLP domains and query types.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    CROSS_DOMAIN = "cross_domain"
    TERMINOLOGY = "terminology"

class CLPDomain(Enum):
    DOMAIN_1 = "Domain_1_Opportunity_Assessment_Development_Valuation"
    DOMAIN_2 = "Domain_2_Intellectual_Property_Protection"
    DOMAIN_3 = "Domain_3_Strategy_Management_Commercialization"
    DOMAIN_4 = "Domain_4_Negotiation_Agreement_Development"
    DOMAIN_5 = "Domain_5_Agreement_Management"

@dataclass
class TestQuery:
    query_text: str
    query_type: QueryType
    primary_domain: CLPDomain
    secondary_domains: List[CLPDomain]
    expected_concepts: List[str]
    min_relevant_chunks: int
    difficulty_level: str  # "basic", "intermediate", "advanced"
    description: str

class CLPTestQueries:
    """Comprehensive test queries for CLP RAG validation."""
    
    @staticmethod
    def get_domain_1_queries() -> List[TestQuery]:
        """Opportunity Assessment, Development, and Valuation queries."""
        return [
            TestQuery(
                query_text="What factors should be considered when valuing intellectual property?",
                query_type=QueryType.ANALYTICAL,
                primary_domain=CLPDomain.DOMAIN_1,
                secondary_domains=[CLPDomain.DOMAIN_2],
                expected_concepts=["market value", "income approach", "cost approach", "market approach", "risk assessment"],
                min_relevant_chunks=3,
                difficulty_level="intermediate",
                description="Core IP valuation methodology question"
            ),
            TestQuery(
                query_text="How do you assess market opportunities for licensing?",
                query_type=QueryType.PROCEDURAL,
                primary_domain=CLPDomain.DOMAIN_1,
                secondary_domains=[CLPDomain.DOMAIN_3],
                expected_concepts=["market analysis", "competitive landscape", "licensing potential", "revenue projections"],
                min_relevant_chunks=4,
                difficulty_level="advanced",
                description="Market opportunity assessment procedure"
            ),
            TestQuery(
                query_text="Define intellectual property portfolio management",
                query_type=QueryType.TERMINOLOGY,
                primary_domain=CLPDomain.DOMAIN_1,
                secondary_domains=[CLPDomain.DOMAIN_2],
                expected_concepts=["portfolio", "management", "IP assets", "strategic alignment"],
                min_relevant_chunks=2,
                difficulty_level="basic",
                description="Basic terminology definition"
            ),
            TestQuery(
                query_text="What are the key steps in conducting IP due diligence?",
                query_type=QueryType.PROCEDURAL,
                primary_domain=CLPDomain.DOMAIN_1,
                secondary_domains=[CLPDomain.DOMAIN_2],
                expected_concepts=["due diligence", "IP audit", "risk assessment", "valuation", "legal review"],
                min_relevant_chunks=4,
                difficulty_level="advanced",
                description="IP due diligence process"
            ),
            TestQuery(
                query_text="How does market size affect IP valuation?",
                query_type=QueryType.ANALYTICAL,
                primary_domain=CLPDomain.DOMAIN_1,
                secondary_domains=[CLPDomain.DOMAIN_3],
                expected_concepts=["market size", "valuation impact", "revenue potential", "market analysis"],
                min_relevant_chunks=3,
                difficulty_level="intermediate",
                description="Market size impact on IP valuation"
            )
        ]
    
    @staticmethod
    def get_domain_2_queries() -> List[TestQuery]:
        """Intellectual Property Protection queries."""
        return [
            TestQuery(
                query_text="What are the key differences between patents, trademarks, and copyrights?",
                query_type=QueryType.FACTUAL,
                primary_domain=CLPDomain.DOMAIN_2,
                secondary_domains=[],
                expected_concepts=["patent", "trademark", "copyright", "protection scope", "duration"],
                min_relevant_chunks=3,
                difficulty_level="basic",
                description="Fundamental IP types comparison"
            ),
            TestQuery(
                query_text="How do you select appropriate international protection mechanisms for IP?",
                query_type=QueryType.ANALYTICAL,
                primary_domain=CLPDomain.DOMAIN_2,
                secondary_domains=[CLPDomain.DOMAIN_1],
                expected_concepts=["international filing", "cost-benefit analysis", "market priorities", "PCT", "Madrid Protocol"],
                min_relevant_chunks=4,
                difficulty_level="advanced",
                description="International IP protection strategy"
            ),
            TestQuery(
                query_text="What is the Patent Cooperation Treaty (PCT)?",
                query_type=QueryType.TERMINOLOGY,
                primary_domain=CLPDomain.DOMAIN_2,
                secondary_domains=[],
                expected_concepts=["PCT", "international patent", "filing system", "unified procedure"],
                min_relevant_chunks=2,
                difficulty_level="basic",
                description="PCT definition and purpose"
            ),
            TestQuery(
                query_text="How do you conduct a freedom-to-operate analysis?",
                query_type=QueryType.PROCEDURAL,
                primary_domain=CLPDomain.DOMAIN_2,
                secondary_domains=[CLPDomain.DOMAIN_1],
                expected_concepts=["freedom to operate", "FTO", "patent landscape", "infringement risk", "prior art"],
                min_relevant_chunks=4,
                difficulty_level="advanced",
                description="Freedom-to-operate analysis procedure"
            ),
            TestQuery(
                query_text="What factors determine patent strength?",
                query_type=QueryType.ANALYTICAL,
                primary_domain=CLPDomain.DOMAIN_2,
                secondary_domains=[CLPDomain.DOMAIN_1],
                expected_concepts=["patent strength", "claims scope", "prior art", "novelty", "non-obviousness"],
                min_relevant_chunks=3,
                difficulty_level="intermediate",
                description="Patent strength assessment factors"
            )
        ]
    
    @staticmethod
    def get_domain_3_queries() -> List[TestQuery]:
        """Strategy Management and Commercialization queries."""
        return [
            TestQuery(
                query_text="What are the key components of an IP commercialization strategy?",
                query_type=QueryType.ANALYTICAL,
                primary_domain=CLPDomain.DOMAIN_3,
                secondary_domains=[CLPDomain.DOMAIN_1, CLPDomain.DOMAIN_4],
                expected_concepts=["commercialization strategy", "licensing", "technology transfer", "market entry", "revenue generation"],
                min_relevant_chunks=4,
                difficulty_level="intermediate",
                description="IP commercialization strategy components"
            ),
            TestQuery(
                query_text="How do you develop a technology transfer plan?",
                query_type=QueryType.PROCEDURAL,
                primary_domain=CLPDomain.DOMAIN_3,
                secondary_domains=[CLPDomain.DOMAIN_4],
                expected_concepts=["technology transfer", "implementation plan", "knowledge transfer", "commercialization roadmap"],
                min_relevant_chunks=3,
                difficulty_level="advanced",
                description="Technology transfer planning process"
            ),
            TestQuery(
                query_text="Define open innovation in the context of IP management",
                query_type=QueryType.TERMINOLOGY,
                primary_domain=CLPDomain.DOMAIN_3,
                secondary_domains=[CLPDomain.DOMAIN_2],
                expected_concepts=["open innovation", "collaborative innovation", "IP sharing", "external partnerships"],
                min_relevant_chunks=2,
                difficulty_level="intermediate",
                description="Open innovation concept definition"
            )
        ]
    
    @staticmethod
    def get_domain_4_queries() -> List[TestQuery]:
        """Negotiation and Agreement Development queries."""
        return [
            TestQuery(
                query_text="What are the essential elements of a licensing agreement?",
                query_type=QueryType.FACTUAL,
                primary_domain=CLPDomain.DOMAIN_4,
                secondary_domains=[CLPDomain.DOMAIN_5],
                expected_concepts=["licensing agreement", "royalties", "territory", "exclusivity", "term duration"],
                min_relevant_chunks=3,
                difficulty_level="basic",
                description="Essential licensing agreement elements"
            ),
            TestQuery(
                query_text="How do you structure royalty payments in licensing deals?",
                query_type=QueryType.ANALYTICAL,
                primary_domain=CLPDomain.DOMAIN_4,
                secondary_domains=[CLPDomain.DOMAIN_1],
                expected_concepts=["royalty structure", "payment terms", "fixed fees", "running royalties", "milestone payments"],
                min_relevant_chunks=4,
                difficulty_level="advanced",
                description="Royalty payment structuring"
            ),
            TestQuery(
                query_text="What negotiation tactics are effective in IP licensing?",
                query_type=QueryType.PROCEDURAL,
                primary_domain=CLPDomain.DOMAIN_4,
                secondary_domains=[CLPDomain.DOMAIN_3],
                expected_concepts=["negotiation tactics", "BATNA", "value proposition", "deal structure", "bargaining power"],
                min_relevant_chunks=3,
                difficulty_level="intermediate",
                description="IP licensing negotiation tactics"
            )
        ]
    
    @staticmethod
    def get_domain_5_queries() -> List[TestQuery]:
        """Agreement Management queries."""
        return [
            TestQuery(
                query_text="How do you monitor compliance with licensing agreements?",
                query_type=QueryType.PROCEDURAL,
                primary_domain=CLPDomain.DOMAIN_5,
                secondary_domains=[CLPDomain.DOMAIN_4],
                expected_concepts=["compliance monitoring", "reporting requirements", "audit rights", "performance metrics"],
                min_relevant_chunks=3,
                difficulty_level="intermediate",
                description="Licensing agreement compliance monitoring"
            ),
            TestQuery(
                query_text="What are the common causes of licensing agreement disputes?",
                query_type=QueryType.ANALYTICAL,
                primary_domain=CLPDomain.DOMAIN_5,
                secondary_domains=[CLPDomain.DOMAIN_4],
                expected_concepts=["licensing disputes", "breach of contract", "royalty disagreements", "scope interpretation"],
                min_relevant_chunks=3,
                difficulty_level="intermediate",
                description="Common licensing dispute causes"
            ),
            TestQuery(
                query_text="Define milestone payments in licensing agreements",
                query_type=QueryType.TERMINOLOGY,
                primary_domain=CLPDomain.DOMAIN_5,
                secondary_domains=[CLPDomain.DOMAIN_4],
                expected_concepts=["milestone payments", "performance-based payments", "development stages", "achievement criteria"],
                min_relevant_chunks=2,
                difficulty_level="basic",
                description="Milestone payments definition"
            )
        ]
    
    @staticmethod
    def get_cross_domain_queries() -> List[TestQuery]:
        """Queries spanning multiple CLP domains."""
        return [
            TestQuery(
                query_text="How does IP valuation influence licensing negotiation strategies?",
                query_type=QueryType.CROSS_DOMAIN,
                primary_domain=CLPDomain.DOMAIN_1,
                secondary_domains=[CLPDomain.DOMAIN_4, CLPDomain.DOMAIN_3],
                expected_concepts=["valuation methods", "negotiation tactics", "licensing terms", "deal structure"],
                min_relevant_chunks=5,
                difficulty_level="advanced",
                description="Integration of valuation and negotiation concepts"
            ),
            TestQuery(
                query_text="What role does IP protection play in commercialization strategy?",
                query_type=QueryType.CROSS_DOMAIN,
                primary_domain=CLPDomain.DOMAIN_2,
                secondary_domains=[CLPDomain.DOMAIN_3, CLPDomain.DOMAIN_1],
                expected_concepts=["protection strategy", "commercialization", "competitive advantage", "market positioning"],
                min_relevant_chunks=4,
                difficulty_level="intermediate",
                description="Protection and commercialization integration"
            ),
            TestQuery(
                query_text="How do agreement management practices affect long-term IP strategy?",
                query_type=QueryType.CROSS_DOMAIN,
                primary_domain=CLPDomain.DOMAIN_5,
                secondary_domains=[CLPDomain.DOMAIN_3, CLPDomain.DOMAIN_1],
                expected_concepts=["agreement management", "strategic planning", "portfolio optimization", "relationship management"],
                min_relevant_chunks=4,
                difficulty_level="advanced",
                description="Agreement management and strategic planning integration"
            ),
            TestQuery(
                query_text="What is the relationship between patent strength and licensing terms?",
                query_type=QueryType.CROSS_DOMAIN,
                primary_domain=CLPDomain.DOMAIN_2,
                secondary_domains=[CLPDomain.DOMAIN_4, CLPDomain.DOMAIN_1],
                expected_concepts=["patent strength", "licensing terms", "royalty rates", "exclusivity", "bargaining power"],
                min_relevant_chunks=4,
                difficulty_level="advanced",
                description="Patent strength impact on licensing terms"
            )
        ]
    
    @staticmethod
    def get_all_test_queries() -> List[TestQuery]:
        """Get all test queries across all domains."""
        queries = []
        queries.extend(CLPTestQueries.get_domain_1_queries())
        queries.extend(CLPTestQueries.get_domain_2_queries())
        queries.extend(CLPTestQueries.get_domain_3_queries())
        queries.extend(CLPTestQueries.get_domain_4_queries())
        queries.extend(CLPTestQueries.get_domain_5_queries())
        queries.extend(CLPTestQueries.get_cross_domain_queries())
        
        return queries
    
    @staticmethod
    def get_queries_by_domain(domain: CLPDomain) -> List[TestQuery]:
        """Get test queries for a specific domain."""
        if domain == CLPDomain.DOMAIN_1:
            return CLPTestQueries.get_domain_1_queries()
        elif domain == CLPDomain.DOMAIN_2:
            return CLPTestQueries.get_domain_2_queries()
        elif domain == CLPDomain.DOMAIN_3:
            return CLPTestQueries.get_domain_3_queries()
        elif domain == CLPDomain.DOMAIN_4:
            return CLPTestQueries.get_domain_4_queries()
        elif domain == CLPDomain.DOMAIN_5:
            return CLPTestQueries.get_domain_5_queries()
        else:
            return []
    
    @staticmethod
    def get_queries_by_type(query_type: QueryType) -> List[TestQuery]:
        """Get test queries of a specific type."""
        all_queries = CLPTestQueries.get_all_test_queries()
        return [q for q in all_queries if q.query_type == query_type]
    
    @staticmethod
    def get_queries_by_difficulty(difficulty: str) -> List[TestQuery]:
        """Get test queries of a specific difficulty level."""
        all_queries = CLPTestQueries.get_all_test_queries()
        return [q for q in all_queries if q.difficulty_level == difficulty] 