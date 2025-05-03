import json
import numpy as np
from recommendation import SHLRecommendationEngine
import argparse
import re
from typing import List, Dict, Any

class RecommendationEvaluator:
    
    def __init__(self, engine=None, assessments_file="data/assessments.json", queries_file="data/test_queries.json"):
        self.engine = engine if engine else SHLRecommendationEngine(assessments_file=assessments_file)
        self.test_queries = self._load_test_queries(queries_file)
        self.metrics = {}
    
    def _load_test_queries(self, queries_file=None) -> List[Dict[str, Any]]:
        if queries_file and os.path.exists(queries_file):
            try:
                with open(queries_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return [
            {
                "query": "Java developers with collaboration skills, 40 min assessment",
                "expected_types": ["Technical", "Skills"],
                "expected_keywords": ["java", "programming", "technical"],
                "relevant_assessments": ["Technical Skills Assessment", "Java Coding Assessment"],
                "remote_testing": True
            },
            {
                "query": "Python data scientist with ML expertise",
                "expected_types": ["Technical", "Skills"],
                "expected_keywords": ["python", "data", "machine learning"],
                "relevant_assessments": ["Data Science Assessment", "Python Coding Test"],
                "remote_testing": True
            }
        ]
    
    def evaluate_type_relevance(self, recommendations: List[Dict], expected_types: List[str]) -> float:
        if not recommendations or not expected_types:
            return 0.0
        
        matches = []
        for rec in recommendations:
            if 'test_type' in rec and rec['test_type']:
                rec_types = [t.lower() for t in rec['test_type']]
                expected_lower = [t.lower() for t in expected_types]
                
                match_score = sum(1 for t in expected_lower if any(t in rt for rt in rec_types)) / len(expected_lower)
                matches.append(match_score)
            else:
                matches.append(0.0)
        
        return sum(matches) / len(recommendations) if matches else 0.0
    
    def evaluate_keyword_relevance(self, recommendations: List[Dict], expected_keywords: List[str]) -> float:
        if not recommendations or not expected_keywords:
            return 0.0
        
        matches = []
        for rec in recommendations:
            rec_text = f"{rec['name']} {rec.get('description', '')}".lower()
            match_score = sum(1 for kw in expected_keywords if kw.lower() in rec_text) / len(expected_keywords)
            matches.append(match_score)
        
        return sum(matches) / len(recommendations) if matches else 0.0
    
    def calculate_recall_at_k(self, recommendations: List[Dict], relevant_assessments: List[str], k: int) -> float:
        if not relevant_assessments:
            return 0.0
        
        top_k_recs = recommendations[:k] if len(recommendations) >= k else recommendations
        
        relevant_found = 0
        for rec in top_k_recs:
            rec_name = rec['name'].lower()
            if any(self._is_assessment_match(rec_name, rel_name.lower()) for rel_name in relevant_assessments):
                relevant_found += 1
        
        return relevant_found / len(relevant_assessments)
    
    def _is_assessment_match(self, rec_name, relevant_name):
        if rec_name == relevant_name:
            return True
        if relevant_name in rec_name or rec_name in relevant_name:
            return True
        
        rec_words = set(rec_name.split())
        rel_words = set(relevant_name.split())
        common_words = rec_words.intersection(rel_words)
        if common_words and len(common_words) / min(len(rec_words), len(rel_words)) >= 0.5:
            return True
            
        return False
    
    def run_evaluation(self, num_recommendations=5):
        overall_results = {
            "type_relevance": [],
            "keyword_relevance": [],
            "overall_score": [],
            "recall_at_k": []
        }
        
        k_values = [3, 5]
        recall_at_k = {k: [] for k in k_values}
        
        for idx, test_case in enumerate(self.test_queries):
            query = test_case["query"]
            print(f"\nEvaluating query {idx+1}/{len(self.test_queries)}: '{query}'")
            
            recommendations = self.engine.recommend(query, max_results=max(k_values))
            
            if not recommendations:
                print("  No recommendations found")
                continue
            
            type_score = self.evaluate_type_relevance(
                recommendations[:num_recommendations], test_case["expected_types"]
            )
            
            keyword_score = self.evaluate_keyword_relevance(
                recommendations[:num_recommendations], test_case["expected_keywords"]
            )
            
            if "relevant_assessments" in test_case:
                for k in k_values:
                    r_at_k = self.calculate_recall_at_k(
                        recommendations, test_case["relevant_assessments"], k
                    )
                    recall_at_k[k].append(r_at_k)
                    print(f"  Recall@{k}: {r_at_k:.2f}")
            
            overall_score = (0.5 * type_score + 0.5 * keyword_score)
            
            overall_results["type_relevance"].append(type_score)
            overall_results["keyword_relevance"].append(keyword_score)
            overall_results["overall_score"].append(overall_score)
            
            print(f"  Type relevance: {type_score:.2f}")
            print(f"  Keyword relevance: {keyword_score:.2f}")
            print(f"  Overall score: {overall_score:.2f}")
            print(f"  Top recommendation: {recommendations[0]['name'] if recommendations else 'None'}")
        
        mean_recall_at_k = {k: np.mean(recall_at_k[k]) for k in k_values if recall_at_k[k]}
        
        self.metrics = {
            "avg_type_relevance": np.mean(overall_results["type_relevance"]),
            "avg_keyword_relevance": np.mean(overall_results["keyword_relevance"]),
            "avg_overall_score": np.mean(overall_results["overall_score"]),
            "mean_recall_at_k": mean_recall_at_k,
        }
        
        print("\nOverall Evaluation Results:")
        print(f"Average Type Relevance: {self.metrics['avg_type_relevance']:.2f}")
        print(f"Average Keyword Relevance: {self.metrics['avg_keyword_relevance']:.2f}")
        print(f"Average Overall Score: {self.metrics['avg_overall_score']:.2f}")
        
        for k in k_values:
            if k in mean_recall_at_k:
                print(f"Mean Recall@{k}: {mean_recall_at_k[k]:.2f}")
        
        return self.metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL recommendation engine")
    parser.add_argument("--num-recs", type=int, default=5, help="Number of recommendations to evaluate")
    parser.add_argument("--queries", type=str, help="Path to test queries JSON file")
    args = parser.parse_args()
    
    evaluator = RecommendationEvaluator(queries_file=args.queries)
    evaluator.run_evaluation(num_recommendations=args.num_recs)

if __name__ == "__main__":
    import os
    main()