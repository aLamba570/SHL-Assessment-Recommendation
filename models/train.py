import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to path to import recommendation and evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommendation import SHLRecommendationEngine
from evaluation import RecommendationEvaluator
from models.model import SHLRecommendationModel

def train_and_evaluate(
    assessments_file="data/assessments.json", 
    embedding_model="all-mpnet-base-v2",
    model_name="shl_recommendation_model",
    save_model=True
):
    """
    Train and evaluate a recommendation model
    
    Args:
        assessments_file: Path to the assessments data
        embedding_model: Name of the embedding model to use
        model_name: Name for the saved model
        save_model: Whether to save the model after training
    
    Returns:
        Tuple of (model, metrics, model_path)
    """
    print(f"Training model '{model_name}' with embedding model: {embedding_model}")
    print(f"Using assessments data from: {assessments_file}")
    
    # Create the model
    model = SHLRecommendationModel(
        model_name=model_name,
        assessments_file=assessments_file,
        embedding_model=embedding_model
    )
    
    # Evaluate the model
    print("Evaluating model performance...")
    evaluator = RecommendationEvaluator(engine=model.engine)
    metrics = evaluator.run_evaluation(num_recommendations=10)
    
    # Update the model with metrics
    model.update_performance_metrics(metrics)
    
    # Print summary of metrics
    print("\nModel Performance Metrics:")
    print(f"  Average Type Relevance: {metrics['avg_type_relevance']:.2f}")
    print(f"  Average Keyword Relevance: {metrics['avg_keyword_relevance']:.2f}")
    print(f"  Average Constraint Satisfaction: {metrics['avg_constraint_satisfaction']:.2f}")
    print(f"  Average Overall Score: {metrics['avg_overall_score']:.2f}")
    print(f"  Mean Recall@3: {metrics['mean_recall_at_k'][3]:.2f}")
    print(f"  MAP@3: {metrics['mean_ap_at_k'][3]:.2f}")
    
    # Save the model if requested
    model_path = None
    if save_model:
        model_path = model.save()
        print(f"\nModel saved to: {model_path}")
    
    return model, metrics, model_path

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an SHL recommendation model")
    parser.add_argument("--assessments", type=str, default="data/assessments.json", 
                        help="Path to assessments data")
    parser.add_argument("--embedding-model", type=str, default="all-mpnet-base-v2", 
                        help="Name of the embedding model to use")
    parser.add_argument("--model-name", type=str, default=f"shl_recommendation_model", 
                        help="Name for the saved model")
    parser.add_argument("--no-save", action="store_true", 
                        help="Skip saving the model")
    parser.add_argument("--optimize", action="store_true", 
                        help="Optimize model weights before saving")
    
    args = parser.parse_args()
    
    # If optimize flag is set, run weight optimization
    if args.optimize:
        print("Optimizing model weights...")
        evaluator = RecommendationEvaluator(assessments_file=args.assessments)
        best_weights = evaluator.weight_optimization()
        print(f"Found optimal weights: {best_weights}")
        
        # We still need to create a model with those weights
        # This is done inside train_and_evaluate
    
    # Train and evaluate the model
    model, metrics, model_path = train_and_evaluate(
        assessments_file=args.assessments,
        embedding_model=args.embedding_model,
        model_name=args.model_name,
        save_model=not args.no_save
    )
    
    # Generate plots if model was evaluated
    if metrics:
        try:
            print("Generating evaluation plots...")
            evaluator = RecommendationEvaluator(engine=model.engine)
            evaluator.metrics = metrics
            evaluator.plot_evaluation_results()
            
            # If model was saved, copy plots to model directory
            if model_path:
                import shutil
                if os.path.exists("evaluation_metrics.png"):
                    shutil.copy("evaluation_metrics.png", os.path.join(model_path, "evaluation_metrics.png"))
                if os.path.exists("ranking_metrics_comparison.png"):
                    shutil.copy("ranking_metrics_comparison.png", os.path.join(model_path, "ranking_metrics_comparison.png"))
                
                print(f"Evaluation plots copied to model directory: {model_path}")
        except Exception as e:
            print(f"Error generating evaluation plots: {e}")

if __name__ == "__main__":
    main()