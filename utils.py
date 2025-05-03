from urllib.parse import urlparse
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def format_recommendations_for_display(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format recommendations for display by ensuring all required fields are present"""
    formatted_recommendations = []
    
    for rec in recommendations:
        formatted_rec = rec.copy()
        
        if "name" not in formatted_rec:
            formatted_rec["name"] = "Unknown Assessment"
            
        if "test_type" not in formatted_rec or not formatted_rec["test_type"]:
            formatted_rec["test_type"] = ["Unknown"]
        
        if "duration" not in formatted_rec or not formatted_rec["duration"]:
            formatted_rec["duration"] = "Unknown"
            
        if "remote_testing" not in formatted_rec:
            formatted_rec["remote_testing"] = False
            
        if "adaptive_irt" not in formatted_rec:
            formatted_rec["adaptive_irt"] = False
            
        if "url" not in formatted_rec or not formatted_rec["url"]:
            formatted_rec["url"] = "#"
            
        formatted_rec["remote_testing_display"] = "Yes" if formatted_rec["remote_testing"] else "No"
        formatted_rec["adaptive_irt_display"] = "Yes" if formatted_rec["adaptive_irt"] else "No"
        
        if isinstance(formatted_rec["test_type"], list):
            formatted_rec["test_type_display"] = ", ".join(formatted_rec["test_type"])
        else:
            formatted_rec["test_type_display"] = str(formatted_rec["test_type"])
            
        formatted_recommendations.append(formatted_rec)
        
    return formatted_recommendations
