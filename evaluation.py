import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from recommendation import SHLRecommendationEngine
import argparse
import re
from typing import List, Dict, Any
from sklearn.metrics import ndcg_score, precision_score, recall_score

class RecommendationEvaluator:
    """
    A class to evaluate the performance of the SHL recommendation engine
    """
    
    def __init__(self, engine=None, assessments_file="data/assessments.json"):
        """Initialize with an engine or create a new one"""
        self.engine = engine if engine else SHLRecommendationEngine(assessments_file=assessments_file)
        self.test_queries = self._load_test_queries()
        self.metrics = {}
        self.all_assessments = self._load_assessments(assessments_file)
    
    def _load_assessments(self, assessments_file):
        """Load all assessments from file"""
        try:
            with open(assessments_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading assessments file: {e}")
            return []
    
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load or create test queries with expected assessments"""
        # These are mock test cases - in a real-world scenario, you would collect 
        # user queries and expert judgments on what assessments are most relevant
        return [
            {
                "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
                "expected_types": ["Technical", "Skills"],
                "expected_keywords": ["java", "programming", "coding", "technical"],
                "relevant_assessments": ["Technical Skills Assessment", "IT Programming Skills", "Software Developer Aptitude Test", "Java Coding Assessment"],
                "remote_testing": True
            },
            {
                "query": "Python data scientist with machine learning expertise",
                "expected_types": ["Technical", "Skills", "Cognitive"],
                "expected_keywords": ["python", "data", "machine learning", "analytical"],
                "relevant_assessments": ["Data Science Assessment", "Python Coding Test", "Analytics Professional Assessment", "Machine Learning Skills Assessment"],
                "remote_testing": True
            },
            {
                "query": "Sales manager with leadership skills, need personality assessment",
                "expected_types": ["Personality", "Behavior"],
                "expected_keywords": ["personality", "leadership", "management"],
                "relevant_assessments": ["Leadership and Management Assessment", "OPQ Sales Manager Assessment", "Management Assessment", "Leadership Personality Assessment"],
                "remote_testing": None
            },
            {
                "query": "Frontend developer with React and Angular experience",
                "expected_types": ["Technical", "Skills"],
                "expected_keywords": ["frontend", "react", "angular", "javascript"],
                "relevant_assessments": ["Frontend Development Assessment", "Web Development Skills Test", "JavaScript Programming Test", "UI Developer Assessment"],
                "remote_testing": True
            },
            {
                "query": "Project manager with agile experience, test should be under 30 minutes",
                "expected_types": ["Professional", "Behavior"],
                "expected_keywords": ["project", "management", "agile"],
                "relevant_assessments": ["Project Management Assessment", "Agile Methodologies Test", "SCRUM Master Assessment", "Leadership Skills Test"],
                "max_duration": 30,
                "remote_testing": None 
            },
            {
                "query": "Customer service representative assessment, must be remote testing",
                "expected_types": ["Behavior", "Skills"],
                "expected_keywords": ["customer", "service", "communication"],
                "relevant_assessments": ["Customer Service Assessment", "Call Center Assessment", "Customer Support Test", "Telephone Skills Assessment"],
                "remote_testing": True
            },
            {
                "query": "DevOps engineer with AWS and Kubernetes experience",
                "expected_types": ["Technical", "Skills"],
                "expected_keywords": ["devops", "aws", "kubernetes", "technical"],
                "relevant_assessments": ["DevOps Skills Assessment", "Cloud Infrastructure Test", "AWS Technical Assessment", "Infrastructure Skills Test"],
                "remote_testing": None
            }
        ]
    
    def evaluate_type_relevance(self, recommendations: List[Dict], expected_types: List[str]) -> float:
        """
        Evaluate how well the recommendations match the expected test types
        
        Returns a score from 0 to 1 where 1 means perfect match
        """
        if not recommendations or not expected_types:
            return 0.0
        
        matches = []
        for rec in recommendations:
            # Check if any of the recommendation's test types match the expected types
            if 'test_type' in rec and rec['test_type']:
                # Check for any overlap between actual and expected types
                rec_types = [t.lower() for t in rec['test_type']]
                expected_lower = [t.lower() for t in expected_types]
                
                # Calculate match as proportion of expected types that are found
                match_score = sum(1 for t in expected_lower if any(t in rt for rt in rec_types)) / len(expected_lower)
                matches.append(match_score)
            else:
                matches.append(0.0)
        
        # Return average match score across top recommendations
        return sum(matches) / len(recommendations) if matches else 0.0
    
    def evaluate_keyword_relevance(self, recommendations: List[Dict], expected_keywords: List[str]) -> float:
        """
        Evaluate how well the recommendations match the expected keywords
        
        Returns a score from 0 to 1 where 1 means perfect match
        """
        if not recommendations or not expected_keywords:
            return 0.0
        
        matches = []
        for rec in recommendations:
            # Check if the recommendation contains the expected keywords
            rec_text = f"{rec['name']} {rec.get('description', '')}".lower()
            
            # Calculate match as proportion of expected keywords that are found
            match_score = sum(1 for kw in expected_keywords if kw.lower() in rec_text) / len(expected_keywords)
            matches.append(match_score)
        
        # Return average match score across top recommendations
        return sum(matches) / len(recommendations) if matches else 0.0
    
    def evaluate_constraint_satisfaction(self, recommendations: List[Dict], constraints: Dict) -> float:
        """
        Evaluate how well the recommendations satisfy the specified constraints
        
        Returns a score from 0 to 1 where 1 means all constraints satisfied
        """
        if not recommendations:
            return 0.0
        
        satisfaction_scores = []
        
        for rec in recommendations:
            constraint_checks = []
            
            # Check remote testing constraint
            if 'remote_testing' in constraints and constraints['remote_testing'] is not None:
                remote_match = rec.get('remote_testing', False) == constraints['remote_testing']
                constraint_checks.append(remote_match)
            
            # Check max duration constraint
            if 'max_duration' in constraints:
                duration_str = rec.get('duration', '').lower()
                
                try:
                    # Try to parse the duration
                    duration_value = None
                    
                    # Pattern for "X minutes"
                    minutes_match = re.search(r'(\d+)\s*(?:minute|minutes|mins?)', duration_str)
                    if minutes_match:
                        duration_value = int(minutes_match.group(1))
                    
                    # Pattern for "X hours" or "X hour Y minutes"
                    hours_match = re.search(r'(\d+)\s*(?:hour|hours?)', duration_str)
                    if hours_match:
                        hours = int(hours_match.group(1))
                        duration_value = hours * 60
                        
                        # Check for additional minutes
                        additional_mins = re.search(r'(\d+)\s*(?:minute|minutes|mins?)', duration_str)
                        if additional_mins:
                            duration_value += int(additional_mins.group(1))
                    
                    # If we could parse the duration, check if it meets the constraint
                    if duration_value is not None:
                        duration_match = duration_value <= constraints['max_duration']
                        constraint_checks.append(duration_match)
                except:
                    # If there was an error parsing the duration, we can't check this constraint
                    pass
            
            # Calculate satisfaction score for this recommendation
            if constraint_checks:
                satisfaction_scores.append(sum(constraint_checks) / len(constraint_checks))
            else:
                satisfaction_scores.append(1.0)  # No constraints to check
            
        return sum(satisfaction_scores) / len(recommendations)
    
    def calculate_recall_at_k(self, recommendations: List[Dict], relevant_assessments: List[str], k: int) -> float:
        """
        Calculate Recall@K for a single query
        
        Recall@K = Number of relevant assessments in top K / Total relevant assessments
        """
        if not relevant_assessments:
            return 0.0
        
        # Get the top K recommendations
        top_k_recs = recommendations[:k] if len(recommendations) >= k else recommendations
        
        # Check which recommendations match the relevant assessments
        relevant_found = 0
        for rec in top_k_recs:
            rec_name = rec['name'].lower()
            # Check if this recommendation matches any of the relevant assessments
            if any(self._is_assessment_match(rec_name, rel_name.lower()) for rel_name in relevant_assessments):
                relevant_found += 1
        
        # Calculate recall
        return relevant_found / len(relevant_assessments)
    
    def _is_assessment_match(self, rec_name, relevant_name):
        """
        Determine if a recommendation name matches a relevant assessment name
        Uses partial matching to handle naming variations
        """
        # Exact match
        if rec_name == relevant_name:
            return True
            
        # Check if the relevant name is contained in the recommendation name
        if relevant_name in rec_name or rec_name in relevant_name:
            return True
            
        # Check for word overlap (at least 50% of words match)
        rec_words = set(rec_name.split())
        rel_words = set(relevant_name.split())
        common_words = rec_words.intersection(rel_words)
        if common_words and len(common_words) / min(len(rec_words), len(rel_words)) >= 0.5:
            return True
            
        return False
    
    def calculate_ap_at_k(self, recommendations: List[Dict], relevant_assessments: List[str], k: int) -> float:
        """
        Calculate Average Precision@K for a single query
        
        AP@K = (1/min(K, R)) * sum(P(k) * rel(k)) for k = 1 to K
        where:
        - R = total relevant assessments for the query
        - P(k) = precision at position k
        - rel(k) = 1 if the result at position k is relevant, 0 otherwise
        """
        if not relevant_assessments or not recommendations:
            return 0.0
        
        # Get the top K recommendations
        top_k_recs = recommendations[:k] if len(recommendations) >= k else recommendations
        
        relevant_at_positions = []
        
        # Determine which positions contain relevant assessments
        for i, rec in enumerate(top_k_recs):
            rec_name = rec['name'].lower()
            is_relevant = any(self._is_assessment_match(rec_name, rel_name.lower()) for rel_name in relevant_assessments)
            relevant_at_positions.append(1 if is_relevant else 0)
        
        # Calculate AP@K
        precision_sum = 0
        relevant_count = 0
        
        for i, is_relevant in enumerate(relevant_at_positions):
            if is_relevant:
                # Position is 1-indexed in the formula
                position = i + 1
                relevant_count += 1
                precision_at_position = relevant_count / position
                precision_sum += precision_at_position
        
        # Denominator is min(K, R) where R is the total number of relevant assessments
        denominator = min(k, len(relevant_assessments))
        
        return precision_sum / denominator if denominator > 0 else 0.0
    
    def run_evaluation(self, num_recommendations=5):
        """Run evaluation on all test queries and compute metrics"""
        overall_results = {
            "type_relevance": [],
            "keyword_relevance": [],
            "constraint_satisfaction": [],
            "overall_score": [],
            "recall_at_k": [],
            "ap_at_k": []
        }
        
        query_results = {}
        k_values = [3, 5, 10]  # Common values for Recall@K and MAP@K
        
        # Track metrics for each K value
        recall_at_k = {k: [] for k in k_values}
        ap_at_k = {k: [] for k in k_values}
        
        for idx, test_case in enumerate(self.test_queries):
            query = test_case["query"]
            print(f"\nEvaluating query {idx+1}/{len(self.test_queries)}: '{query}'")
            
            # Get recommendations
            recommendations = self.engine.recommend(query, max_results=max(k_values))
            
            # Skip if no recommendations found
            if not recommendations:
                print("  No recommendations found")
                continue
            
            # Evaluate type relevance
            type_score = self.evaluate_type_relevance(
                recommendations[:num_recommendations], test_case["expected_types"]
            )
            
            # Evaluate keyword relevance
            keyword_score = self.evaluate_keyword_relevance(
                recommendations[:num_recommendations], test_case["expected_keywords"]
            )
            
            # Evaluate constraint satisfaction
            constraints = {}
            if "remote_testing" in test_case:
                constraints["remote_testing"] = test_case["remote_testing"]
            if "max_duration" in test_case:
                constraints["max_duration"] = test_case["max_duration"]
                
            constraint_score = self.evaluate_constraint_satisfaction(
                recommendations[:num_recommendations], constraints
            )
            
            # Calculate Recall@K and AP@K for each K value
            if "relevant_assessments" in test_case:
                for k in k_values:
                    r_at_k = self.calculate_recall_at_k(
                        recommendations, test_case["relevant_assessments"], k
                    )
                    recall_at_k[k].append(r_at_k)
                    
                    a_at_k = self.calculate_ap_at_k(
                        recommendations, test_case["relevant_assessments"], k
                    )
                    ap_at_k[k].append(a_at_k)
                    
                    # Print K-specific metrics
                    print(f"  Recall@{k}: {r_at_k:.2f}")
                    print(f"  AP@{k}: {a_at_k:.2f}")
            
            # Calculate overall score (weighted average)
            overall_score = (0.4 * type_score + 0.4 * keyword_score + 0.2 * constraint_score)
            
            # Store results
            overall_results["type_relevance"].append(type_score)
            overall_results["keyword_relevance"].append(keyword_score)
            overall_results["constraint_satisfaction"].append(constraint_score)
            overall_results["overall_score"].append(overall_score)
            
            # Store individual query results
            query_results[query] = {
                "type_relevance": type_score,
                "keyword_relevance": keyword_score,
                "constraint_satisfaction": constraint_score,
                "overall_score": overall_score,
                "num_recommendations": len(recommendations[:num_recommendations]),
                "top_recommendation": recommendations[0]["name"] if recommendations else "None"
            }
            
            # Add K-specific metrics to query results
            for k in k_values:
                if "relevant_assessments" in test_case:
                    query_results[query][f"recall_at_{k}"] = recall_at_k[k][-1]
                    query_results[query][f"ap_at_{k}"] = ap_at_k[k][-1]
            
            # Print results for this query
            print(f"  Type relevance: {type_score:.2f}")
            print(f"  Keyword relevance: {keyword_score:.2f}")
            print(f"  Constraint satisfaction: {constraint_score:.2f}")
            print(f"  Overall score: {overall_score:.2f}")
            print(f"  Top recommendation: {recommendations[0]['name'] if recommendations else 'None'}")
        
        # Calculate MAP@K for each K value
        mean_ap_at_k = {k: np.mean(ap_at_k[k]) for k in k_values if ap_at_k[k]}
        mean_recall_at_k = {k: np.mean(recall_at_k[k]) for k in k_values if recall_at_k[k]}
        
        # Compute average metrics
        self.metrics = {
            "avg_type_relevance": np.mean(overall_results["type_relevance"]),
            "avg_keyword_relevance": np.mean(overall_results["keyword_relevance"]),
            "avg_constraint_satisfaction": np.mean(overall_results["constraint_satisfaction"]),
            "avg_overall_score": np.mean(overall_results["overall_score"]),
            "mean_recall_at_k": mean_recall_at_k,
            "mean_ap_at_k": mean_ap_at_k,
            "query_results": query_results
        }
        
        # Print overall results
        print("\nOverall Evaluation Results:")
        print(f"Average Type Relevance: {self.metrics['avg_type_relevance']:.2f}")
        print(f"Average Keyword Relevance: {self.metrics['avg_keyword_relevance']:.2f}")
        print(f"Average Constraint Satisfaction: {self.metrics['avg_constraint_satisfaction']:.2f}")
        print(f"Average Overall Score: {self.metrics['avg_overall_score']:.2f}")
        
        # Print Mean Recall@K and MAP@K values
        for k in k_values:
            if k in mean_recall_at_k:
                print(f"Mean Recall@{k}: {mean_recall_at_k[k]:.2f}")
            if k in mean_ap_at_k:
                print(f"MAP@{k}: {mean_ap_at_k[k]:.2f}")
        
        return self.metrics
    
    def weight_optimization(self):
        """
        Optimize the recommendation engine weights to improve performance
        
        This uses a simple grid search to find the best weights
        """
        print("\nOptimizing recommendation weights...")
        
        # Grid of weights to try
        content_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
        skill_weights = [0.1, 0.2, 0.3, 0.4]
        
        best_weights = (0.5, 0.25, 0.25)  # default weights
        best_score = 0.0
        
        results = []
        
        for cw in content_weights:
            for sw in skill_weights:
                tw = 1 - cw - sw
                # Skip invalid weights
                if tw <= 0:
                    continue
                
                # Set weights
                self.engine.adjust_weights(content_weight=cw, skill_weight=sw, type_weight=tw)
                
                # Run evaluation
                self.run_evaluation(num_recommendations=3)
                score = self.metrics["avg_overall_score"]
                
                results.append({
                    "content_weight": cw,
                    "skill_weight": sw,
                    "type_weight": tw,
                    "score": score
                })
                
                print(f"  Weights (content={cw:.1f}, skill={sw:.1f}, type={tw:.1f}): score = {score:.3f}")
                
                # Update best weights if this configuration is better
                if score > best_score:
                    best_score = score
                    best_weights = (cw, sw, tw)
        
        # Set the best weights
        cw, sw, tw = best_weights
        self.engine.adjust_weights(content_weight=cw, skill_weight=sw, type_weight=tw)
        
        print(f"\nBest weights found: content={cw:.1f}, skill={sw:.1f}, type={tw:.1f}")
        print(f"Best score: {best_score:.3f}")
        
        # Plot the results
        try:
            df = pd.DataFrame(results)
            pivot_table = df.pivot_table(index='content_weight', columns='skill_weight', values='score')
            
            plt.figure(figsize=(10, 8))
            plt.title('Recommendation Performance by Weight Configuration')
            
            # Create heatmap
            heatmap = plt.pcolor(pivot_table)
            plt.colorbar(heatmap)
            
            # Add weight values to cells
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    plt.text(j + 0.5, i + 0.5, f'{pivot_table.iloc[i, j]:.3f}',
                             horizontalalignment='center',
                             verticalalignment='center')
            
            plt.xticks(np.arange(0.5, len(pivot_table.columns)), pivot_table.columns)
            plt.yticks(np.arange(0.5, len(pivot_table.index)), pivot_table.index)
            plt.xlabel('Skill Weight')
            plt.ylabel('Content Weight')
            
            plt.savefig('weight_optimization.png')
            print("\nSaved weight optimization plot to 'weight_optimization.png'")
        except Exception as e:
            print(f"Error generating optimization plot: {e}")
        
        return best_weights
    
    def plot_evaluation_results(self):
        """Plot evaluation results"""
        if not self.metrics or not self.metrics.get("query_results"):
            print("No evaluation results to plot")
            return
        
        try:
            # Extract results for each query
            queries = list(self.metrics["query_results"].keys())
            short_queries = [q[:20] + "..." if len(q) > 20 else q for q in queries]
            
            # Create two subplot figures - one for traditional metrics, one for ranking metrics
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # Plot traditional metrics
            type_scores = [self.metrics["query_results"][q]["type_relevance"] for q in queries]
            keyword_scores = [self.metrics["query_results"][q]["keyword_relevance"] for q in queries]
            constraint_scores = [self.metrics["query_results"][q]["constraint_satisfaction"] for q in queries]
            overall_scores = [self.metrics["query_results"][q]["overall_score"] for q in queries]
            
            x = np.arange(len(short_queries))
            width = 0.2
            
            ax1.bar(x - width*1.5, type_scores, width, label='Type Relevance')
            ax1.bar(x - width/2, keyword_scores, width, label='Keyword Relevance')
            ax1.bar(x + width/2, constraint_scores, width, label='Constraint Satisfaction')
            ax1.bar(x + width*1.5, overall_scores, width, label='Overall Score')
            
            ax1.set_xlabel('Test Queries')
            ax1.set_ylabel('Score (0-1)')
            ax1.set_title('Traditional Evaluation Metrics')
            ax1.set_xticks(x)
            ax1.set_xticklabels(short_queries, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot ranking metrics
            # Check if we have Recall@K and AP@K values
            k_values = sorted([int(k.split('_')[-1]) for k in self.metrics["query_results"][queries[0]] if k.startswith("recall_at_")])
            
            if k_values:
                recall_data = []
                ap_data = []
                
                for k in k_values:
                    recall_data.append([self.metrics["query_results"][q].get(f"recall_at_{k}", 0) for q in queries])
                    ap_data.append([self.metrics["query_results"][q].get(f"ap_at_{k}", 0) for q in queries])
                
                # Plot Recall@K
                for i, k in enumerate(k_values):
                    ax2.plot(x, recall_data[i], marker='o', linestyle='-', label=f'Recall@{k}')
                
                # Plot AP@K
                for i, k in enumerate(k_values):
                    ax2.plot(x, ap_data[i], marker='s', linestyle='--', label=f'AP@{k}')
                
                ax2.set_xlabel('Test Queries')
                ax2.set_ylabel('Score (0-1)')
                ax2.set_title('Ranking Evaluation Metrics')
                ax2.set_xticks(x)
                ax2.set_xticklabels(short_queries, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('evaluation_metrics.png')
            print("\nSaved comprehensive evaluation metrics plot to 'evaluation_metrics.png'")
            
            # Create a separate plot for Mean Recall@K and MAP@K comparison
            if "mean_recall_at_k" in self.metrics and "mean_ap_at_k" in self.metrics:
                plt.figure(figsize=(10, 6))
                
                k_values = sorted(list(self.metrics["mean_recall_at_k"].keys()))
                mean_recall = [self.metrics["mean_recall_at_k"][k] for k in k_values]
                mean_ap = [self.metrics["mean_ap_at_k"][k] for k in k_values]
                
                plt.bar(np.array(k_values) - 0.2, mean_recall, 0.4, label='Mean Recall@K')
                plt.bar(np.array(k_values) + 0.2, mean_ap, 0.4, label='MAP@K')
                
                plt.xlabel('K Value')
                plt.ylabel('Score (0-1)')
                plt.title('Mean Recall@K and MAP@K Comparison')
                plt.xticks(k_values)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig('ranking_metrics_comparison.png')
                print("Saved ranking metrics comparison plot to 'ranking_metrics_comparison.png'")
            
        except Exception as e:
            print(f"Error generating evaluation plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL recommendation engine")
    parser.add_argument("--optimize", action="store_true", help="Optimize recommendation weights")
    parser.add_argument("--num-recs", type=int, default=5, help="Number of recommendations to evaluate")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    parser.add_argument("--k-values", type=str, default="3,5,10", help="Comma-separated K values for Recall@K and MAP@K")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator()
    
    # Run weight optimization if requested
    if args.optimize:
        evaluator.weight_optimization()
    
    # Run evaluation
    evaluator.run_evaluation(num_recommendations=args.num_recs)
    
    # Generate plots if requested
    if args.plot:
        evaluator.plot_evaluation_results()

if __name__ == "__main__":
    main()