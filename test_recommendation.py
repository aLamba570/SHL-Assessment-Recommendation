import unittest
import os
import json
from recommendation import SHLRecommendationEngine

class TestRecommendationEngine(unittest.TestCase):
    """Tests for the SHL Recommendation Engine"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests"""
        # Initialize the recommendation engine
        cls.engine = SHLRecommendationEngine()
        
        # Load test queries
        cls.test_queries = [
            "Java developer with good communication skills",
            "Python data scientist with machine learning expertise",
            "I am hiring for Java developers who can also collaborate effectively with my business teams."
        ]
    
    def test_engine_initialization(self):
        """Test that engine initializes properly with assessments and model"""
        self.assertIsNotNone(self.engine.assessments)
        self.assertGreater(len(self.engine.assessments), 0)
        self.assertIsNotNone(self.engine.model)
        self.assertIsNotNone(self.engine.assessment_embeddings)
    
    def test_skill_extraction(self):
        """Test skill extraction functionality"""
        skills = self.engine.extract_skills_from_query("Looking for Java and Python developers with AWS experience")
        self.assertIn("java", skills)
        self.assertIn("python", skills)
        self.assertIn("aws", skills)
    
    def test_job_role_extraction(self):
        """Test job role extraction functionality"""
        roles = self.engine.extract_job_roles_from_query("Need a software engineer who knows Java")
        self.assertTrue(any("software engineer" in role for role in roles))
        
    def test_test_type_extraction(self):
        """Test test type extraction functionality"""
        test_types = self.engine.extract_test_types_from_query("Technical assessment for Python developers")
        self.assertIn("technical", test_types)
        self.assertIn("skills", test_types)
    
    def test_duration_constraint_parsing(self):
        """Test duration constraint parsing"""
        max_duration = self.engine.parse_duration_constraint("Test should be less than 30 minutes")
        self.assertEqual(max_duration, 30)
        
        max_duration = self.engine.parse_duration_constraint("Test should be under 1 hour")
        self.assertEqual(max_duration, 60)
    
    def test_recommendation_basic(self):
        """Test basic recommendation functionality"""
        results = self.engine.recommend("Java developer", max_results=3)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 3)
        
        # Check that structure is as expected
        self.assertIn("name", results[0])
        self.assertIn("similarity_score", results[0])
    
    def test_filtering(self):
        """Test filtering functionality"""
        filters = {"remote_testing": True, "test_types": ["Technical"]}
        results = self.engine.recommend("Java developer", max_results=5, filters=filters)
        
        # All results should have remote_testing=True
        for result in results:
            self.assertTrue(result.get("remote_testing", False))
            # At least one test type should be Technical
            self.assertTrue(any(t == "Technical" for t in result.get("test_type", [])))
    
    def test_explanation(self):
        """Test explanation functionality"""
        results = self.engine.recommend("Java developer", max_results=1)
        explanation = self.engine.explain_recommendation(results[0], "Java developer")
        self.assertIn("factors", explanation)
        self.assertIn("score_breakdown", explanation)


if __name__ == "__main__":
    unittest.main()