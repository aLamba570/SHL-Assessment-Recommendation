import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add parent directory to path to import recommendation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommendation import SHLRecommendationEngine

class SHLRecommendationModel:
    """
    A wrapper class for the SHLRecommendationEngine that provides model management
    capabilities such as saving, loading, versioning, and metadata.
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self, 
        model_name: str = "shl_recommendation_model",
        assessments_file: str = "data/assessments.json", 
        embedding_model: str = "all-mpnet-base-v2"
    ):
        """Initialize the model with a recommendation engine"""
        self.model_name = model_name
        self.engine = SHLRecommendationEngine(
            assessments_file=assessments_file,
            model_name=embedding_model
        )
        self.metadata = {
            "version": self.VERSION,
            "created_at": datetime.now().isoformat(),
            "embedding_model": embedding_model,
            "assessments_file": assessments_file,
            "performance_metrics": {}
        }
    
    def recommend(self, query: str, **kwargs) -> List[Dict]:
        """Wrapper for the recommend method of the engine"""
        return self.engine.recommend(query, **kwargs)
    
    def explain_recommendation(self, assessment: Dict, query: str) -> Dict:
        """Wrapper for the explain_recommendation method of the engine"""
        return self.engine.explain_recommendation(assessment, query)
    
    def update_performance_metrics(self, metrics: Dict) -> None:
        """Update the model's metadata with performance metrics"""
        self.metadata["performance_metrics"] = metrics
        self.metadata["last_evaluated"] = datetime.now().isoformat()
    
    def save(self, directory: str = "models") -> str:
        """
        Save the model to disk
        
        Returns:
            Path to the saved model directory
        """
        # Create unique model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(directory, f"{self.model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save embedding model reference
        with open(os.path.join(model_dir, "embedding_model.txt"), 'w') as f:
            f.write(self.engine.model.__class__.__name__)
            f.write("\n")
            f.write(self.metadata["embedding_model"])
        
        # Save assessment embeddings
        np.save(
            os.path.join(model_dir, "assessment_embeddings.npy"), 
            self.engine.assessment_embeddings
        )
        
        # Save assessment data
        with open(os.path.join(model_dir, "assessments.json"), 'w') as f:
            json.dump(self.engine.assessments, f, indent=2)
        
        # Save engine object (without the embedding model to save space)
        engine_to_save = self.engine
        embedding_model = engine_to_save.model
        engine_to_save.model = None
        
        with open(os.path.join(model_dir, "engine.pkl"), 'wb') as f:
            pickle.dump(engine_to_save, f)
        
        # Restore the model reference
        engine_to_save.model = embedding_model
        
        # Create a simple "model card" markdown file
        with open(os.path.join(model_dir, "README.md"), 'w') as f:
            f.write(f"# {self.model_name}\n\n")
            f.write(f"Version: {self.metadata['version']}\n")
            f.write(f"Created: {self.metadata['created_at']}\n")
            f.write(f"Embedding Model: {self.metadata['embedding_model']}\n\n")
            
            f.write("## Performance Metrics\n\n")
            if self.metadata['performance_metrics']:
                for metric, value in self.metadata['performance_metrics'].items():
                    if isinstance(value, dict):
                        f.write(f"### {metric}:\n")
                        for sub_metric, sub_value in value.items():
                            f.write(f"- {sub_metric}: {sub_value}\n")
                    else:
                        f.write(f"- {metric}: {value}\n")
            else:
                f.write("No performance metrics available yet.\n")
        
        return model_dir
    
    @classmethod
    def load(cls, model_path: str) -> 'SHLRecommendationModel':
        """
        Load a saved model from disk
        
        Args:
            model_path: Path to the saved model directory
            
        Returns:
            Loaded SHLRecommendationModel
        """
        from sentence_transformers import SentenceTransformer
        
        # Load metadata
        with open(os.path.join(model_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Create an empty model instance
        model = cls(model_name=os.path.basename(model_path))
        model.metadata = metadata
        
        # Load engine object
        with open(os.path.join(model_path, "engine.pkl"), 'rb') as f:
            model.engine = pickle.load(f)
        
        # Load embedding model
        with open(os.path.join(model_path, "embedding_model.txt"), 'r') as f:
            embedding_model_name = f.readlines()[1].strip()
        
        model.engine.model = SentenceTransformer(embedding_model_name)
        
        # Load assessment embeddings
        model.engine.assessment_embeddings = np.load(
            os.path.join(model_path, "assessment_embeddings.npy")
        )
        
        # Load assessments
        with open(os.path.join(model_path, "assessments.json"), 'r') as f:
            model.engine.assessments = json.load(f)
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "name": self.model_name,
            "version": self.metadata["version"],
            "created_at": self.metadata["created_at"],
            "embedding_model": self.metadata["embedding_model"],
            "num_assessments": len(self.engine.assessments),
            "embedding_dim": self.engine.embedding_dim,
            "performance_metrics": self.metadata["performance_metrics"],
        }


if __name__ == "__main__":
    # Example usage of the model
    model = SHLRecommendationModel()
    
    # Make recommendations
    results = model.recommend(
        "Java developer with good communication skills",
        max_results=3
    )
    
    # Save the model
    model_path = model.save()
    print(f"Model saved to: {model_path}")
    
    # Load the model
    loaded_model = SHLRecommendationModel.load(model_path)
    print(f"Model loaded successfully: {loaded_model.get_model_info()['name']}")