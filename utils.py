import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os
import json
from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import numpy as np
from sentence_transformers import util
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to load spaCy for better NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    logger.info("spaCy model not available. Using basic NLP processing.")

def extract_url_content(url: str) -> str:
    """
    Extract content from a URL
    
    Args:
        url: The URL to extract content from
        
    Returns:
        The extracted content as a string
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            # Get text
            text = soup.get_text()
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        else:
            logger.warning(f"Failed to get content from {url}, status code: {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting content from URL {url}: {e}")
        return ""

def extract_links_from_html(html: str) -> List[str]:
    """
    Extract all links from HTML content
    
    Args:
        html: The HTML content as a string
        
    Returns:
        A list of URLs found in the HTML content
    """
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Make sure to resolve relative URLs
        full_url = urljoin(url, href)
        links.append(full_url)
    return links

def is_valid_url(url: str) -> bool:
    """Check if a URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def format_duration(duration_str: str) -> str:
    """Format duration string for display"""
    if not duration_str or duration_str.lower() == "unknown":
        return "Not specified"
    return duration_str

def format_test_type(test_type_codes: str, test_types: List[str]) -> str:
    """Format test type for display"""
    if test_types:
        return ", ".join(test_types)
    if test_type_codes:
        return test_type_codes
    return "Not specified"

def create_data_directory() -> None:
    """Create data directory if it doesn't exist"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def load_json_file(file_path: str, default_value=None) -> Any:
    """Load JSON from file with error handling"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            print(f"File not found: {file_path}")
            return default_value
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return default_value

def save_json_file(data: Any, file_path: str) -> bool:
    """Save JSON to file with error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False

def extract_text_from_url(url: str, timeout: int = 10) -> str:
    """Extract text content from a URL"""
    if not is_valid_url(url):
        return ""
        
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to get content from {url}, status code: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error extracting content from URL {url}: {e}")
        return ""

def rate_limit(min_interval: float = 0.5) -> None:
    """Rate limiting function to prevent too many requests"""
    if not hasattr(rate_limit, "last_call"):
        rate_limit.last_call = 0
    
    elapsed = time.time() - rate_limit.last_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    
    rate_limit.last_call = time.time()

def format_recommendations_for_display(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format recommendations for display by ensuring all required fields are present"""
    formatted_recommendations = []
    
    for rec in recommendations:
        # Create a copy of the recommendation
        formatted_rec = rec.copy()
        
        # Ensure all required fields are present
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
            
        # Format boolean values as Yes/No
        formatted_rec["remote_testing_display"] = "Yes" if formatted_rec["remote_testing"] else "No"
        formatted_rec["adaptive_irt_display"] = "Yes" if formatted_rec["adaptive_irt"] else "No"
        
        # Format test types as comma-separated string
        if isinstance(formatted_rec["test_type"], list):
            formatted_rec["test_type_display"] = ", ".join(formatted_rec["test_type"])
        else:
            formatted_rec["test_type_display"] = str(formatted_rec["test_type"])
            
        # Add to formatted recommendations
        formatted_recommendations.append(formatted_rec)
        
    return formatted_recommendations

def get_assessment_name_from_url(url: str) -> str:
    """Extract assessment name from URL"""
    if not url:
        return "Unknown"
        
    try:
        # Extract the last part of the URL path
        path = urlparse(url).path
        last_part = os.path.basename(path.rstrip('/'))
        
        # Clean up and format
        name = last_part.replace('-', ' ').replace('view', '').strip()
        name = re.sub(r'\s+', ' ', name)  # Remove extra spaces
        
        if name:
            return name.title()
        return "Unknown"
    except:
        return "Unknown"

def format_duration_string(duration_mins: int) -> str:
    """Convert duration in minutes to a human-readable string"""
    if duration_mins < 60:
        return f"{duration_mins} minutes"
    else:
        hours = duration_mins // 60
        mins = duration_mins % 60
        if mins == 0:
            return f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            return f"{hours} hour{'s' if hours > 1 else ''} {mins} minute{'s' if mins > 1 else ''}"

def parse_duration_string(duration_str: str) -> Optional[int]:
    """Parse a duration string and return the duration in minutes"""
    if not duration_str:
        return None
        
    duration_str = duration_str.lower()
    
    # Pattern for "X minutes"
    minutes_match = re.search(r'(\d+)\s*(?:minute|minutes|mins?)', duration_str)
    if minutes_match:
        return int(minutes_match.group(1))
    
    # Pattern for "X hours" or "X hour Y minutes"
    hours_match = re.search(r'(\d+)\s*(?:hour|hours?)', duration_str)
    if hours_match:
        hours = int(hours_match.group(1))
        duration_mins = hours * 60
        
        # Check for additional minutes
        additional_mins = re.search(r'(\d+)\s*(?:minute|minutes|mins?)', duration_str)
        if additional_mins:
            duration_mins += int(additional_mins.group(1))
            
        return duration_mins
        
    return None

def evaluate_recommendation_accuracy(recommendations, ground_truth):
    """
    Evaluate the accuracy of recommendations against ground truth
    
    Args:
        recommendations: List of recommended assessment objects
        ground_truth: List of ground truth assessment objects
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not recommendations or not ground_truth:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "matched_count": 0,
            "total_relevant": len(ground_truth) if ground_truth else 0,
            "total_recommended": len(recommendations) if recommendations else 0
        }
    
    # Extract names for easier matching
    rec_names = [r.get("name", "").lower() for r in recommendations]
    gt_names = [g.get("name", "").lower() for g in ground_truth]
    
    # Count matches
    matches = sum(1 for name in rec_names if any(
        gt_name.lower() in name.lower() or name.lower() in gt_name.lower() 
        for gt_name in gt_names
    ))
    
    # Calculate metrics
    precision = matches / len(recommendations) if recommendations else 0
    recall = matches / len(ground_truth) if ground_truth else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "matched_count": matches,
        "total_relevant": len(ground_truth),
        "total_recommended": len(recommendations)
    }

def compute_mean_average_precision(recommendations_list, ground_truth_list, k=3):
    """
    Compute Mean Average Precision @ K across multiple queries
    
    Args:
        recommendations_list: List of recommendation lists for multiple queries
        ground_truth_list: List of ground truth lists for multiple queries
        k: K value for MAP@K
        
    Returns:
        MAP@K score
    """
    if not recommendations_list or not ground_truth_list:
        return 0.0
    
    if len(recommendations_list) != len(ground_truth_list):
        logger.warning("Number of recommendation lists doesn't match number of ground truth lists")
        return 0.0
    
    average_precisions = []
    
    for recommendations, ground_truth in zip(recommendations_list, ground_truth_list):
        if not recommendations or not ground_truth:
            average_precisions.append(0.0)
            continue
            
        # Cut recommendations to k
        recs_at_k = recommendations[:k] if len(recommendations) > k else recommendations
        
        # Extract names
        rec_names = [r.get("name", "").lower() for r in recs_at_k]
        gt_names = [g.get("name", "").lower() for g in ground_truth]
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for i, rec_name in enumerate(rec_names):
            is_relevant = any(
                gt_name.lower() in rec_name.lower() or rec_name.lower() in gt_name.lower() 
                for gt_name in gt_names
            )
            
            if is_relevant:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))  # precision at position i+1
        
        # Calculate average precision
        if precisions:
            ap = sum(precisions) / len(ground_truth)  # normalize by total number of relevant items
            average_precisions.append(ap)
        else:
            average_precisions.append(0.0)
    
    # Calculate MAP
    map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    return map_score

def compute_mean_recall_at_k(recommendations_list, ground_truth_list, k=3):
    """
    Compute Mean Recall @ K across multiple queries
    
    Args:
        recommendations_list: List of recommendation lists for multiple queries
        ground_truth_list: List of ground truth lists for multiple queries
        k: K value for Recall@K
        
    Returns:
        Mean Recall@K score
    """
    if not recommendations_list or not ground_truth_list:
        return 0.0
    
    if len(recommendations_list) != len(ground_truth_list):
        logger.warning("Number of recommendation lists doesn't match number of ground truth lists")
        return 0.0
    
    recalls = []
    
    for recommendations, ground_truth in zip(recommendations_list, ground_truth_list):
        if not recommendations or not ground_truth:
            recalls.append(0.0)
            continue
            
        # Cut recommendations to k
        recs_at_k = recommendations[:k] if len(recommendations) > k else recommendations
        
        # Extract names
        rec_names = [r.get("name", "").lower() for r in recs_at_k]
        gt_names = [g.get("name", "").lower() for g in ground_truth]
        
        # Count relevant items in top-K
        relevant_count = sum(1 for rec_name in rec_names if any(
            gt_name.lower() in rec_name.lower() or rec_name.lower() in gt_name.lower() 
            for gt_name in gt_names
        ))
        
        # Calculate recall
        recall = relevant_count / len(ground_truth) if ground_truth else 0.0
        recalls.append(recall)
    
    # Calculate mean recall
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    return mean_recall

def extract_skills_from_query(query: str) -> Set[str]:
    """Extract potential skills from the query with improved NLP"""
    skills = set()
    
    # Common technical skills to look for
    tech_skills = {
        'python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'c#', 'ruby', 'php', 'go',
        'rust', 'scala', 'kotlin', 'swift', 'objective-c', 'r', 'matlab', 'sql', 'nosql', 'mongodb',
        'postgresql', 'mysql', 'oracle', 'cassandra', 'redis', 'react', 'angular', 'vue', 'node',
        'express', 'django', 'flask', 'spring', 'asp.net', 'html', 'css', 'sass', 'less',
        'bootstrap', 'tailwind', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform',
        'jenkins', 'git', 'ci/cd', 'devops', 'machine learning', 'ml', 'ai', 'data science',
        'nlp', 'computer vision', 'cv', 'deep learning', 'neural networks', 'tensorflow', 'pytorch',
        'keras', 'scikit-learn', 'pandas', 'numpy', 'hadoop', 'spark', 'jira', 'scrum', 'agile',
        'kanban', 'uml', 'requirements', 'systems analysis', 'cloud computing',
        'microservices', 'restful api', 'graphql', 'product management', 'embedded systems',
        'iot', 'blockchain', 'cybersecurity', 'networking', 'linux', 'unix', 'windows', 'macos'
    }
    
    # Common soft skills
    soft_skills = {
        'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
        'creativity', 'time management', 'adaptability', 'project management', 'conflict resolution',
        'emotional intelligence', 'negotiation', 'presentation', 'writing', 'decision making',
        'mentoring', 'coaching', 'analytical', 'research', 'attention to detail', 'interpersonal',
        'customer service', 'sales', 'marketing', 'multitasking', 'delegation', 'strategic thinking',
        'innovation', 'planning', 'organization', 'self-motivation', 'results-oriented'
    }
    
    # Create a combined skill set
    all_skills = tech_skills.union(soft_skills)
    
    # Lowercase the query for case-insensitive matching
    query_lower = query.lower()
    
    # If spaCy is available, use it for better entity and phrase extraction
    if SPACY_AVAILABLE:
        doc = nlp(query_lower)
        
        # Extract noun phrases as potential skills
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if chunk_text in all_skills:
                skills.add(chunk_text)
                
        # Extract named entities that might be technologies
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                skills.add(ent.text.lower())
    
    # Always do direct matching for common skills
    for skill in all_skills:
        if skill in query_lower:
            # Handle compound skills (containing spaces)
            if ' ' in skill:
                # Make sure it's a proper match by checking word boundaries
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, query_lower):
                    skills.add(skill)
            else:
                # For single words, simple boundary check
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, query_lower):
                    skills.add(skill)
    
    # Extract programming languages and technologies with special patterns
    # Look for specific patterns like "X developer", "X engineer", "X programming"
    tech_patterns = [
        r'\b(\w+)(?:\s+|-)?developer\b',
        r'\b(\w+)(?:\s+|-)?engineer\b',
        r'\b(\w+)(?:\s+|-)?programmer\b',
        r'\b(\w+)(?:\s+|-)?programming\b',
        r'\bexperience (?:in|with) (\w+)\b',
        r'\bknowledge of (\w+)\b'
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if match in tech_skills:
                skills.add(match)
    
    return skills

def extract_job_roles_from_query(query: str) -> Set[str]:
    """Extract job roles from the query with improved detection"""
    roles = set()
    
    # Common job roles
    job_roles = {
        'software engineer', 'developer', 'frontend developer', 'front-end developer', 
        'backend developer', 'back-end developer', 'full stack developer', 'fullstack developer',
        'data scientist', 'data analyst', 'data engineer', 'machine learning engineer',
        'devops engineer', 'site reliability engineer', 'sre', 'cloud engineer', 'security engineer',
        'qa engineer', 'quality assurance engineer', 'test engineer', 'ux designer', 'ui designer',
        'product manager', 'project manager', 'program manager', 'scrum master', 'product owner',
        'business analyst', 'systems analyst', 'network engineer', 'database administrator', 'dba',
        'system administrator', 'sysadmin', 'technical support', 'help desk', 'it specialist',
        'technical writer', 'content developer', 'solutions architect', 'enterprise architect',
        'sales engineer', 'customer success manager', 'account manager', 'marketing specialist',
        'director', 'vp', 'cto', 'ceo', 'cio', 'manager', 'team lead', 'tech lead'
    }
    
    # Lowercase the query for case-insensitive matching
    query_lower = query.lower()
    
    # If spaCy is available, use it for better entity and phrase extraction
    if SPACY_AVAILABLE:
        doc = nlp(query_lower)
        
        # Extract noun phrases as potential job roles
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if chunk_text in job_roles:
                roles.add(chunk_text)
    
    # Direct matching for common roles
    for role in job_roles:
        if ' ' in role:
            # Make sure it's a proper match by checking word boundaries
            pattern = r'\b' + re.escape(role) + r'\b'
            if re.search(pattern, query_lower):
                roles.add(role)
        else:
            # For single words, simple boundary check
            pattern = r'\b' + re.escape(role) + r'\b'
            if re.search(pattern, query_lower):
                roles.add(role)
    
    return roles

def extract_test_types_from_query(query: str) -> Set[str]:
    """Extract test types from the query with better pattern matching"""
    test_types = set()
    
    # Common test types
    type_keywords = {
        'technical': {'coding', 'programming', 'development', 'technical', 'code'},
        'cognitive': {'cognitive', 'aptitude', 'reasoning', 'intelligence', 'iq'},
        'personality': {'personality', 'trait', 'character', 'temperament', 'type', 'assessment'},
        'skills': {'skills', 'capability', 'ability', 'competency', 'skill'},
        'behavior': {'behavior', 'behaviour', 'situational', 'judgment', 'judgement'},
        'language': {'language', 'verbal', 'communication', 'linguistic'}
    }
    
    # Lowercase the query for case-insensitive matching
    query_lower = query.lower()
    
    # Look for test type indicators
    patterns = [
        r'\b(\w+)(?:\s+|-)?test\b',
        r'\b(\w+)(?:\s+|-)?assessment\b',
        r'\b(\w+)(?:\s+|-)?evaluation\b',
        r'\bneed (\w+)(?:\s+|-)?(?:test|assessment|evaluation)\b',
        r'\brequire (\w+)(?:\s+|-)?(?:test|assessment|evaluation)\b',
        r'\bwant (\w+)(?:\s+|-)?(?:test|assessment|evaluation)\b'
    ]
    
    # Search for explicit test type mentions
    for test_type, keywords in type_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                if ' ' in keyword:
                    # Make sure it's a proper match by checking word boundaries
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, query_lower):
                        test_types.add(test_type)
                else:
                    # For single words, simple boundary check
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, query_lower):
                        test_types.add(test_type)
    
    # Extract test types from patterns
    for pattern in patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            # Check if the match is in any of the keyword sets
            for test_type, keywords in type_keywords.items():
                if match in keywords:
                    test_types.add(test_type)
    
    # Special case for technical assessment based on job roles
    if not test_types and any(tech_term in query_lower for tech_term in ['software', 'developer', 'programming', 'java', 'python', 'frontend', 'backend', 'fullstack', 'data scientist']):
        test_types.add('technical')
    
    return test_types

def parse_duration_constraint(query: str) -> Optional[int]:
    """Parse maximum duration constraints from the query"""
    duration_patterns = [
        r'under (\d+) minutes',
        r'less than (\d+) minutes',
        r'no more than (\d+) minutes',
        r'within (\d+) minutes',
        r'maximum (\d+) minutes',
        r'max (\d+) minutes',
        r'not longer than (\d+) minutes',
        r'(\d+) minutes or less',
        r'(\d+) min or less',
        r'shorter than (\d+) minutes',
        r'(\d+) min max',
        # Hour patterns
        r'under (\d+) hour',
        r'less than (\d+) hour',
        r'no more than (\d+) hour',
    ]
    
    query_lower = query.lower()
    
    for pattern in duration_patterns:
        match = re.search(pattern, query_lower)
        if match:
            duration = int(match.group(1))
            
            # Convert hours to minutes if needed
            if 'hour' in pattern:
                duration *= 60
                
            return duration
    
    return None

def parse_testing_preferences(query: str) -> Dict[str, Optional[bool]]:
    """Parse testing preferences/constraints from the query"""
    preferences = {
        "remote_testing": None,
        "adaptive_irt": None
    }
    
    query_lower = query.lower()
    
    # Remote testing patterns
    remote_positive = ['remote testing', 'online testing', 'remote assessment', 'online assessment', 
                      'test remotely', 'assess remotely', 'from home', 'virtual assessment']
    remote_negative = ['in person', 'on site', 'on-site', 'in-person', 'in office', 'in-office']
    
    for phrase in remote_positive:
        if phrase in query_lower:
            preferences["remote_testing"] = True
            break
            
    for phrase in remote_negative:
        if phrase in query_lower:
            preferences["remote_testing"] = False
            break
    
    # Adaptive testing patterns
    adaptive_positive = ['adaptive test', 'adaptive assessment', 'irt', 'item response theory', 'personalized test']
    adaptive_negative = ['standard test', 'fixed questions', 'fixed length', 'non-adaptive']
    
    for phrase in adaptive_positive:
        if phrase in query_lower:
            preferences["adaptive_irt"] = True
            break
            
    for phrase in adaptive_negative:
        if phrase in query_lower:
            preferences["adaptive_irt"] = False
            break
    
    return preferences

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between embeddings a and b
    
    Args:
        a: First embedding matrix
        b: Second embedding matrix
        
    Returns:
        Matrix of cosine similarities
    """
    if len(a) == 0 or len(b) == 0:
        return np.array([])
        
    try:
        return util.cos_sim(a, b).numpy()
    except:
        # Fallback to manual calculation
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        
        # Avoid division by zero
        a_norm[a_norm == 0] = 1e-10
        b_norm[b_norm == 0] = 1e-10
        
        a_normalized = a / a_norm
        b_normalized = b / b_norm
        
        return np.dot(a_normalized, b_normalized.T)
