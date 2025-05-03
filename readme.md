# SHL Assessment Recommendation System

## Overview
The SHL Assessment Recommendation System is an intelligent tool that helps hiring managers find the most relevant SHL assessments for their hiring needs. Given a natural language query or a job description (text or URL), the system recommends appropriate assessments from SHL's product catalog.

## Features
- Process natural language queries to understand hiring needs
- Extract and analyze content from job description URLs
- Filter assessments based on multiple constraints (duration, test type, etc.)
- Provide explanations for recommendations

## API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
Check if the API is running properly and the model is loaded.

- **URL**: `/api/health`
- **Method**: `GET`
- **Response Example**:
  ```json
  {
    "status": "healthy",
    "assessments_loaded": 513,
    "model_loaded": true
  }
  ```

#### 2. Get Recommendations
Get assessment recommendations based on a query or job description.

- **URL**: `/api/recommendations`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "Java developer with good communication skills",
    "max_results": 5,
    "remote_testing": true
  }
  ```
  OR
  ```json
  {
    "url": "https://example.com/job-posting",
    "max_results": 3,
    "test_types": ["Technical", "Skills"]
  }
  ```
- **Parameters**:
  - `query` (string): Natural language query describing the role
  - `url` (string, optional): URL to a job description
  - `max_results` (integer, optional): Maximum number of results to return
  - `remote_testing` (boolean, optional): Filter for remote testing support
  - `adaptive_irt` (boolean, optional): Filter for adaptive testing support
  - `test_types` (array, optional): Filter for specific test types

- **Response Example**:
  ```json
  {
    "recommendations": [
      {
        "name": "Java Coding Assessment",
        "test_type": ["Technical", "Skills"],
        "duration": "35 minutes",
        "remote_testing": true,
        "similarity_score": 0.85,
        "explanation": {
          "factors": ["Matches skills: java, programming"],
          "score_breakdown": {"skill_match": 0.7}
        }
      }
    ],
    "query_metadata": {
      "skills_detected": ["java", "communication"],
      "job_roles_detected": ["developer"],
      "test_types_detected": ["technical", "skills"],
      "max_duration": null,
      "testing_preferences": {}
    },
    "total_results": 1
  }
  ```

#### 3. Get All Assessments
Get a list of all available assessments.

- **URL**: `/api/assessments`
- **Method**: `GET`
- **Response**: JSON array of all assessment objects

#### 4. Get Assessment Types
Get all available assessment types.

- **URL**: `/api/assessment_types`
- **Method**: `GET`
- **Response Example**:
  ```json
  ["Technical", "Cognitive", "Personality", "Behavioral", "Skills"]
  ```

#### 5. Adjust Weights
Adjust the weights used in the recommendation algorithm.

- **URL**: `/api/adjust_weights`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "content_weight": 0.6,
    "skill_weight": 0.2,
    "type_weight": 0.2
  }
  ```
- **Parameters**:
  - `content_weight` (float): Weight for content similarity
  - `skill_weight` (float): Weight for skill matching
  - `type_weight` (float): Weight for test type matching

- **Response Example**:
  ```json
  {
    "success": true,
    "current_weights": {
      "content_weight": 0.6,
      "skill_weight": 0.2,
      "type_weight": 0.2
    }
  }
  ```

#### 6. Extract Skills
Extract technical skills from provided text.

- **URL**: `/api/extract_skills`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Looking for a senior developer with experience in Java, Python, and AWS."
  }
  ```
- **Response Example**:
  ```json
  {
    "skills": ["java", "python", "aws"]
  }
  ```

## Testing with Postman

A Postman collection file (`shl_api_tests.postman_collection.json`) is included in the repository for easy testing of all API endpoints:

1. Start the API server with `python api.py`
2. Import the collection into Postman
3. Run individual requests or use the collection runner to test all endpoints

## Setup and Usage

### Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation
```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API
```bash
python api.py
```
This will start the API server on port 5000.

### Environment Variables
- `GOOGLE_API_KEY`: (Optional) API key for Google Generative AI features
- `PORT`: (Optional) Port to run the API server on (default: 5000)

## Core Components

1. **Recommendation Engine**: Processes queries and finds relevant assessments
2. **Embedding Model**: Creates semantic representations of text
3. **API Server**: Handles HTTP requests and responses 

## Performance and Evaluation
The system is evaluated using metrics like Mean Recall@K and Mean Average Precision (MAP@K). See evaluation metrics in `evaluation_metrics.png` and `ranking_metrics_comparison.png`.

## Future Improvements
- Support for more languages
- Enhanced explanation capabilities
- Integration with application tracking systems
- Custom assessment creation recommendations
