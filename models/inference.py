import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import SHLRecommendationModel

def get_available_models(models_dir="models"):
    """Get a list of available saved models"""
    if not os.path.exists(models_dir):
        return []
        
    model_dirs = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and "shl_recommendation_model" in item:
            # Check if it has necessary files to be considered a valid model
            if os.path.exists(os.path.join(item_path, "metadata.json")):
                model_dirs.append(item)
    
    return model_dirs

def load_model(model_path):
    """Load a model from a saved path"""
    try:
        model = SHLRecommendationModel.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def get_recommendations(model, query, url=None, max_results=5, filters=None):
    """Get recommendations using a loaded model"""
    try:
        recommendations = model.recommend(
            query=query,
            url=url,
            max_results=max_results,
            filters=filters
        )
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

def print_recommendations(recommendations):
    """Print recommendations in a formatted way"""
    if not recommendations:
        print("No recommendations found.")
        return
        
    print(f"\nFound {len(recommendations)} recommendations:\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Test Types: {', '.join(rec.get('test_type', ['Unknown']))}")
        print(f"   Duration: {rec.get('duration', 'Unknown')}")
        print(f"   Remote Testing: {'Yes' if rec.get('remote_testing', False) else 'No'}")
        print(f"   Adaptive IRT: {'Yes' if rec.get('adaptive_irt', False) else 'No'}")
        print(f"   Similarity Score: {rec.get('similarity_score', 0.0):.4f}")
        print()

def format_recommendations_for_api(recommendations):
    return [
        {
            "name": rec.get("name", ""),
            "test_type": rec.get("test_type", []),
            "duration": rec.get("duration", ""),
            "remote_testing": rec.get("remote_testing", False),
            "adaptive_irt": rec.get("adaptive_irt", False),
            "similarity_score": float(rec.get("similarity_score", 0.0)),
            "url": rec.get("url", "")
        }
        for rec in recommendations
    ]

def main():
    parser = argparse.ArgumentParser(description="Make predictions using a saved SHL recommendation model")
    parser.add_argument("--query", type=str, required=True,
                        help="The query to get recommendations for")
    parser.add_argument("--url", type=str, default=None,
                        help="Optional URL to extract additional content from")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a specific model to use, if not provided the latest model will be used")
    parser.add_argument("--max-results", type=int, default=5,
                        help="Maximum number of recommendations to return")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--remote-testing", type=str, default=None, choices=["yes", "no"],
                        help="Filter for remote testing support")
    parser.add_argument("--adaptive", type=str, default=None, choices=["yes", "no"],
                        help="Filter for adaptive IRT support")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to a JSON file")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        models = get_available_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No saved models found.")
        return
    
    # Determine which model to use
    model_path = args.model_path
    if not model_path:
        # Find the latest model
        models = get_available_models()
        if not models:
            print("No saved models found. Please train a model first or specify a model path.")
            return
        # Sort by creation time (assuming the timestamp is in the name)
        models.sort(reverse=True)
        model_path = os.path.join("models", models[0])
        print(f"Using latest model: {models[0]}")
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    if not model:
        return
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model: {model_info['name']}")
    print(f"Version: {model_info['version']}")
    print(f"Created: {model_info['created_at']}")
    print(f"Embedding Model: {model_info['embedding_model']}")
    print(f"Number of Assessments: {model_info['num_assessments']}")
    
    # Prepare filters
    filters = {}
    if args.remote_testing:
        filters["remote_testing"] = args.remote_testing.lower() == "yes"
    if args.adaptive:
        filters["adaptive_irt"] = args.adaptive.lower() == "yes"
    
    # Get recommendations
    print(f"\nGetting recommendations for query: '{args.query}'")
    if filters:
        print(f"Applying filters: {filters}")
    
    recommendations = get_recommendations(
        model=model,
        query=args.query,
        url=args.url,
        max_results=args.max_results,
        filters=filters if filters else None
    )
    
    # Print recommendations
    print_recommendations(recommendations)
    
    # Save to JSON if requested
    if args.output_json:
        formatted_recs = format_recommendations_for_api(recommendations)
        with open(args.output_json, 'w') as f:
            json.dump(formatted_recs, f, indent=2)
        print(f"Recommendations saved to {args.output_json}")

if __name__ == "__main__":
    main()