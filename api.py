from flask import Flask, request, jsonify
import os
import json
from recommendation import SHLRecommendationEngine
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Google API if available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI initialized successfully")
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables. GenAI features will be disabled.")

app = Flask(__name__)

# Initialize the recommendation engine
engine = SHLRecommendationEngine()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "assessments_loaded": len(engine.assessments),
        "model_loaded": engine.model is not None
    })

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get assessment recommendations based on query or job description"""
    data = request.json
    
    if not data or ('query' not in data and 'url' not in data):
        return jsonify({
            "error": "Missing required parameters. Please provide 'query' or 'url'."
        }), 400
    
    # Extract parameters from request
    query = data.get('query', '')
    url = data.get('url', None)
    max_results = data.get('max_results', 10)
    
    # Extract filters
    filters = {}
    if 'remote_testing' in data:
        filters['remote_testing'] = data['remote_testing']
    if 'adaptive_irt' in data:
        filters['adaptive_irt'] = data['adaptive_irt']
    if 'test_types' in data and isinstance(data['test_types'], list):
        filters['test_types'] = data['test_types']
    
    # Log the request
    logger.info(f"Recommendation request: query='{query}', url={url}, filters={filters}")
    
    try:
        # Get recommendations
        recommendations = engine.recommend(query, url, max_results, filters)
        
        # Get explanations for top recommendations
        for rec in recommendations[:3]:  # Only explain top 3
            rec['explanation'] = engine.explain_recommendation(rec, query)
        
        # Extract metadata about the query
        query_metadata = {
            "skills_detected": list(engine.extract_skills_from_query(query)),
            "job_roles_detected": list(engine.extract_job_roles_from_query(query)),
            "test_types_detected": list(engine.extract_test_types_from_query(query)),
            "max_duration": engine.parse_duration_constraint(query),
            "testing_preferences": engine.parse_testing_preferences(query)
        }
        
        return jsonify({
            "recommendations": recommendations,
            "query_metadata": query_metadata,
            "total_results": len(recommendations)
        })
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500

@app.route('/api/assessments', methods=['GET'])
def get_assessments():
    """Get all available assessments"""
    try:
        # Return all assessments
        return jsonify(engine.assessments)
    except Exception as e:
        logger.error(f"Error getting assessments: {str(e)}")
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500

@app.route('/api/assessment_types', methods=['GET'])
def get_assessment_types():
    """Get all available assessment types"""
    try:
        # Extract unique test types
        test_types = set()
        for assessment in engine.assessments:
            if 'test_type' in assessment and assessment['test_type']:
                for test_type in assessment['test_type']:
                    test_types.add(test_type)
        
        return jsonify(sorted(list(test_types)))
    except Exception as e:
        logger.error(f"Error getting assessment types: {str(e)}")
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500


@app.route('/api/extract_skills', methods=['POST'])
def extract_skills():
    """Extract technical skills from text"""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({
            "error": "Missing required parameter 'text'."
        }), 400
    
    text = data['text']
    
    try:
        skills = engine.extract_skills_from_query(text)
        
        return jsonify({
            "skills": list(skills)
        })
    except Exception as e:
        logger.error(f"Error extracting skills: {str(e)}")
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)