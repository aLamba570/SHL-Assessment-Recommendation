import streamlit as st
import json
from recommendation import SHLRecommendationEngine
from utils import is_valid_url, format_recommendations_for_display
import os
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evaluation import RecommendationEvaluator

st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    page_icon="📋",
    layout="wide"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file. Some features may not work.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_resource
def load_recommendation_engine():
    try:
        from models.inference import get_available_models
        from models.model import SHLRecommendationModel
        
        models = get_available_models()
        if models:
            models.sort(reverse=True)
            model_path = os.path.join("models", models[0])
            st.success(f"Using saved model: {models[0]}")
            model = SHLRecommendationModel.load(model_path)
            return model.engine
        
        st.info("No saved model found. Creating a new recommendation engine.")
        engine = SHLRecommendationEngine()
        return engine
    except Exception as e:
        st.error(f"Error initializing recommendation engine: {e}")
        return None

@st.cache_resource
def initialize_rag_system():
    try:
        with open("data/assessments.json", "r") as f:
            assessments = json.load(f)
        
        documents = []
        for assessment in assessments:
            content = (
                f"Assessment: {assessment['name']}\n"
                f"Description: {assessment.get('description', 'No description available')}\n"
                f"Test Types: {', '.join(assessment.get('test_type', ['Unknown']))}\n"
                f"Duration: {assessment.get('duration', 'Unknown')}\n"
                f"Remote Testing: {'Yes' if assessment.get('remote_testing', False) else 'No'}\n"
                f"Adaptive/IRT Testing: {'Yes' if assessment.get('adaptive_irt', False) else 'No'}\n"
                f"URL: {assessment.get('url', '#')}\n"
            )
            documents.append({"content": content, "metadata": {"name": assessment['name'], "url": assessment.get('url', '#')}})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([doc["content"] for doc in documents], metadatas=[doc["metadata"] for doc in documents])
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        return retriever
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

@st.cache_data
def load_evaluation_metrics():
    try:
        engine = load_recommendation_engine()
        if engine:
            evaluator = RecommendationEvaluator(engine=engine)
            metrics = evaluator.run_evaluation(num_recommendations=10)
            return metrics
        return None
    except Exception as e:
        st.error(f"Error loading evaluation metrics: {str(e)}")
        return None

@st.cache_resource
def initialize_llm():
    try:
        if not GOOGLE_API_KEY:
            return None
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def generate_enhanced_recommendations(query, basic_recommendations, retriever, llm):
    if not llm or not retriever:
        return basic_recommendations, None
        
    try:
        template = """
        You are a specialized AI assistant for SHL assessments and testing solutions. Your task is to recommend the best SHL assessments based on the user's query or job description.
        
        USER QUERY: {query}
        
        CONTEXT FROM SHL ASSESSMENTS DATABASE:
        {context}
        
        BASIC RECOMMENDATIONS FROM VECTOR SIMILARITY:
        {basic_recommendations}
        
        Based on the user query and context, provide:
        1. An analysis of what the user is looking for in terms of assessment needs
        2. A ranked list of the top 5 most relevant assessments from the basic recommendations with a brief explanation for each
        3. Any specific constraints the user mentioned (like time limitations, remote testing requirements, etc.) and how your recommendations address them
        
        FORMAT YOUR RESPONSE AS:
        <analysis>
        Brief analysis of the user's assessment needs
        </analysis>
        
        <recommendations>
        1. Assessment Name 1 - Brief explanation of why this is relevant
        2. Assessment Name 2 - Brief explanation of why this is relevant
        ...etc.
        </recommendations>
        
        <constraints>
        Any constraints mentioned by the user and how recommendations address them
        </constraints>
        """
        
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        basic_recs_text = "\n".join([f"- {rec['name']} ({', '.join(rec.get('test_type', ['Unknown']))}, Duration: {rec.get('duration', 'Unknown')}, Remote: {'Yes' if rec.get('remote_testing', False) else 'No'})" for rec in basic_recommendations])
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"query": RunnablePassthrough(), "context": lambda x: context, "basic_recommendations": lambda x: basic_recs_text}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke(query)
        
        analysis = ""
        recommendations_text = ""
        constraints = ""
        
        if "<analysis>" in response and "</analysis>" in response:
            analysis = response.split("<analysis>")[1].split("</analysis>")[0].strip()
        
        if "<recommendations>" in response and "</recommendations>" in response:
            recommendations_text = response.split("<recommendations>")[1].split("</recommendations>")[0].strip()
        
        if "<constraints>" in response and "</constraints>" in response:
            constraints = response.split("<constraints>")[1].split("</constraints>")[0].strip()
        
        insights = {
            "analysis": analysis,
            "recommendations_explanation": recommendations_text,
            "constraints": constraints,
            "full_response": response
        }
        
        return basic_recommendations, insights
    except Exception as e:
        st.error(f"Error generating enhanced recommendations: {str(e)}")
        return basic_recommendations, None

def display_recommendations(recommendations, insights=None):
    if not recommendations:
        st.warning("No relevant assessments found. Please try a different query.")
        return
    
    if insights:
        with st.container():
            st.markdown("### Analysis")
            st.markdown(insights["analysis"])
            
            st.markdown("### Key Constraints and Considerations")
            st.markdown(insights["constraints"])
        
        st.markdown("---")
    
    st.success(f"Found {len(recommendations)} relevant assessments")
    
    df = pd.DataFrame(recommendations)
    
    cols_to_show = ['name', 'remote_testing', 'adaptive_irt', 'duration', 'test_type']
    renamed_cols = {
        'name': 'Assessment Name', 
        'remote_testing': 'Remote Testing',
        'adaptive_irt': 'Adaptive/IRT',
        'duration': 'Duration',
        'test_type': 'Test Type'
    }
    
    for col in cols_to_show:
        if col not in df.columns:
            df[col] = "N/A"
    
    df_display = df[cols_to_show].rename(columns=renamed_cols)
    
    df_display['Assessment Name'] = df.apply(
        lambda x: f"<a href='{x.get('url', '#')}' target='_blank'>{x['name']}</a>", 
        axis=1
    )
    
    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    if insights and insights.get("recommendations_explanation"):
        with st.expander("Why these assessments were recommended", expanded=True):
            st.markdown(insights["recommendations_explanation"])

def display_system_evaluation():
    st.markdown("## System Performance Metrics")
    
    metrics = load_evaluation_metrics()
    if not metrics:
        st.warning("Evaluation metrics are not available.")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Relevance Metrics")
        
        metrics_data = {
            "Metric": ["Type Relevance", "Keyword Relevance", "Constraint Satisfaction", "Overall Score"],
            "Value": [
                f"{metrics['avg_type_relevance']:.2f}",
                f"{metrics['avg_keyword_relevance']:.2f}",
                f"{metrics['avg_constraint_satisfaction']:.2f}",
                f"{metrics['avg_overall_score']:.2f}"
            ],
            "Interpretation": [
                "Low" if metrics['avg_type_relevance'] < 0.3 else ("Moderate" if metrics['avg_type_relevance'] < 0.7 else "High"),
                "Low" if metrics['avg_keyword_relevance'] < 0.3 else ("Moderate" if metrics['avg_keyword_relevance'] < 0.7 else "High"),
                "Low" if metrics['avg_constraint_satisfaction'] < 0.3 else ("Moderate" if metrics['avg_constraint_satisfaction'] < 0.7 else "High"),
                "Low" if metrics['avg_overall_score'] < 0.3 else ("Moderate" if metrics['avg_overall_score'] < 0.7 else "High")
            ]
        }
        
        st.table(pd.DataFrame(metrics_data))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics_data["Metric"], [float(x) for x in metrics_data["Value"]])
        
        colors = ['#ff9999' if x == "Low" else '#ffcc99' if x == "Moderate" else '#99ff99' for x in metrics_data["Interpretation"]]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Recommendation System Relevance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Ranking Metrics")
        
        k_values = sorted(list(metrics["mean_recall_at_k"].keys()))
        mean_recall = [metrics["mean_recall_at_k"][k] for k in k_values]
        mean_ap = [metrics["mean_ap_at_k"][k] for k in k_values]
        
        ranking_data = {
            "K Value": k_values,
            "Mean Recall@K": [f"{val:.2f}" for val in mean_recall],
            "MAP@K": [f"{val:.2f}" for val in mean_ap],
            "Interpretation": [
                "Low" if (mean_recall[i] < 0.3 and mean_ap[i] < 0.3) 
                else ("Moderate" if (mean_recall[i] < 0.7 or mean_ap[i] < 0.7) else "High")
                for i in range(len(k_values))
            ]
        }
        
        st.table(pd.DataFrame(ranking_data))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(k_values))
        width = 0.35
        
        recall_bars = ax.bar(x - width/2, mean_recall, width, label='Mean Recall@K')
        map_bars = ax.bar(x + width/2, mean_ap, width, label='MAP@K')
        
        recall_colors = ['#ff9999' if val < 0.3 else '#ffcc99' if val < 0.7 else '#99ff99' for val in mean_recall]
        map_colors = ['#ff9999' if val < 0.3 else '#ffcc99' if val < 0.7 else '#99ff99' for val in mean_ap]
        
        for bar, color in zip(recall_bars, recall_colors):
            bar.set_color(color)
            
        for bar, color in zip(map_bars, map_colors):
            bar.set_color(color)
        
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('K Value')
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Recommendation System Ranking Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.legend()
        
        plt.tight_layout()
        
        st.pyplot(fig)
    
    st.markdown("### Interpretation of Evaluation Results")
    st.info("""
    **Current System Performance Summary:**
    
    The recommendation system is currently performing at a suboptimal level, with most metrics showing low to moderate performance:
    
    - **Type Relevance (0.74)**: The system struggles to match query intent with appropriate assessment types
    - **Keyword Relevance (0.21)**: The system needs improvement in detecting and matching relevant keywords
    - **Constraint Satisfaction (0.91)**: The system is moderately effective at respecting constraints like remote testing and duration
    - **Mean Recall@K**: The system retrieves only a small portion of relevant assessments at different K values
    - **MAP@K**: The system needs improvement in both retrieving relevant assessments and ranking them appropriately
    
    
    The RAG and GenAI components help mitigate some of these limitations by providing additional context and explanations.
    """)

def main():
    st.title("SHL Assessment Recommendation System with RAG and GenAI")
    st.markdown("""
    This system uses Retrieval-Augmented Generation and Generative AI to recommend SHL assessments 
    that best match your job requirements or assessment needs.
    
    Enter your query or job description below, or provide a URL to a job description.
    """)
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system uses Retrieval-Augmented Generation (RAG) and Generative AI to recommend SHL assessments that best match your needs.
        
        **Features:**
        - Natural language query processing with GenAI
        - Analyze job description text with RAG
        - Extract content from job description URLs
        - Filter by assessment duration and requirements
        - Receive explanations for recommendation decisions
        - Advanced semantic matching algorithm
        """)
        
        st.markdown("---")
        
        st.header("Sample Queries")
        st.markdown("""
        - I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.
        - Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.
        - I am hiring for an analyst and want applications to screen using Cognitive and personality tests, what options are available within 45 mins.
        """)
        
        st.markdown("---")
        
        st.header("Navigation")
        page = st.radio("Go to", ["Recommendation Engine", "System Evaluation"])
    
    engine = load_recommendation_engine()
    retriever = initialize_rag_system()
    llm = initialize_llm()
    
    if engine is None:
        st.error("Failed to initialize recommendation engine. Please check logs for more details.")
        return
    
    page = st.sidebar.radio("Select Page", ["Recommendation Engine", "System Evaluation"])
    
    if page == "Recommendation Engine":
        with st.form("query_form"):
            query = st.text_area(
                "Enter your query or job description", 
                height=150,
                placeholder="E.g., 'Looking for technical assessment for Java developers that can be completed in 30 minutes'"
            )
            
            url = st.text_input(
                "Or enter a job description URL (optional)",
                placeholder="https://example.com/job-description"
            )
            
            with st.expander("Advanced Options"):
                use_rag = st.checkbox("Use RAG and GenAI for enhanced recommendations", value=True)
                max_results = st.slider("Maximum number of recommendations", min_value=1, max_value=20, value=10)
                
                col1, col2 = st.columns(2)
                with col1:
                    remote_testing = st.radio("Remote Testing", [None, True, False], format_func=lambda x: "Any" if x is None else ("Yes" if x else "No"))
                with col2:
                    adaptive_testing = st.radio("Adaptive/IRT Testing", [None, True, False], format_func=lambda x: "Any" if x is None else ("Yes" if x else "No"))
            
            submit = st.form_submit_button("Get Recommendations")
        
        if submit:
            if not query and not url:
                st.warning("Please provide either a query or a URL.")
                return
                
            if url and not is_valid_url(url):
                st.warning("Please enter a valid URL.")
                return
                
            with st.spinner("Processing your request..."):
                try:
                    filters = {}
                    if remote_testing is not None:
                        filters["remote_testing"] = remote_testing
                    if adaptive_testing is not None:
                        filters["adaptive_irt"] = adaptive_testing
                    
                    basic_recommendations = engine.recommend(query, url=url, max_results=max_results, filters=filters)
                    
                    recommendations = basic_recommendations
                    insights = None
                    
                    if use_rag and retriever and llm:
                        st.info("Enhancing recommendations with Generative AI...")
                        recommendations, insights = generate_enhanced_recommendations(query, basic_recommendations, retriever, llm)
                    
                    formatted_recs = format_recommendations_for_display(recommendations)
                    
                    display_recommendations(formatted_recs, insights)
                    
                except Exception as e:
                    st.error(f"Error getting recommendations: {str(e)}")
                    
        st.info("""
        **Note**: Our recommendation engine is continuously learning and improving. 
        For detailed information about the system's performance metrics and limitations, 
        please visit the "System Evaluation" page in the sidebar.
        """)
                
    elif page == "System Evaluation":
        display_system_evaluation()
    
    st.markdown("---")
    st.markdown("© 2025 SHL Assessment Recommendation System")

if __name__ == "__main__":
    main()