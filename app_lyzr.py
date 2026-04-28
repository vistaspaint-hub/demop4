"""
Flask Web App for HR Chatbot with Lyzr Integration
Simple GUI to ask questions and get AI-powered responses
"""

from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv('.env')

app = Flask(__name__)

# Global variables
pinecone_client = None
index = None
embedding_model = None
lyzr_api_key = None
lyzr_agent_id = None


def initialize():
    """Initialize Pinecone and Lyzr credentials"""
    global pinecone_client, index, embedding_model, lyzr_api_key, lyzr_agent_id
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    index = pinecone_client.Index("hr-employee-data")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load Lyzr credentials
    lyzr_api_key = os.getenv("LYZR_API_KEY")
    lyzr_agent_id = os.getenv("LYZR_AGENT_ID")
    
    print("✅ Initialized successfully!")


def retrieve_employees(query, top_k=5):
    """Retrieve relevant employees from Pinecone"""
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results['matches']


def format_context(matches):
    """Format employee data for Lyzr"""
    if not matches:
        return "No relevant employee data found."
    
    context = "Here are the relevant employee records:\n\n"
    
    for i, match in enumerate(matches, 1):
        metadata = match['metadata']
        score = match['score']
        
        context += f"--- Employee {i} (Relevance Score: {score:.3f}) ---\n"
        context += f"Employee ID: {metadata.get('employee_id', 'N/A')}\n"
        context += f"Age: {metadata.get('age', 'N/A')} years\n"
        context += f"Gender: {metadata.get('gender', 'N/A')}\n"
        context += f"Department: {metadata.get('department', 'N/A')}\n"
        context += f"Role: {metadata.get('role', 'N/A')}\n"
        context += f"Experience: {metadata.get('experience_years', 'N/A')} years\n"
        context += f"Gross Salary: ${metadata.get('gross_salary', 0):,.2f}\n"
        context += f"Net Salary: ${metadata.get('net_salary', 0):,.2f}\n"
        context += f"Performance Score: {metadata.get('performance_score', 'N/A')}/5.0\n"
        context += f"Job Satisfaction: {metadata.get('job_satisfaction', 'N/A')}/5.0\n"
        context += f"Promotion Eligible: {metadata.get('promotion_eligible', 'N/A')}\n"
        context += "\n"
    
    return context


def call_lyzr(question, context):
    """Call Lyzr API"""
    url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
    
    user_prompt = f"""Based on the following employee data, please answer this question:

QUESTION: {question}

EMPLOYEE DATA:
{context}

Please provide a comprehensive, clear answer based on the data above."""
    
    headers = {
        "x-api-key": lyzr_api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "user_id": "hr_chatbot_user",
        "agent_id": lyzr_agent_id,
        "message": user_prompt,
        "session_id": "hr_session"
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    result = response.json()
    
    return result.get('response', result.get('message', str(result)))


@app.route('/')
def home():
    """Home page with chat interface"""
    return render_template('hr_chatbot.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Please enter a question'
            }), 400
        
        # Retrieve employees
        matches = retrieve_employees(question, top_k=5)
        
        # Format context
        context = format_context(matches)
        
        # Call Lyzr
        answer = call_lyzr(question, context)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'num_employees_found': len(matches)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🤖 HR CHATBOT WEB APP (Lyzr Integration)")
    print("="*80)
    print("\nInitializing...")
    
    try:
        initialize()
        print("\n" + "="*80)
        print("🌐 Starting Flask server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("="*80 + "\n")
        
        app.run(debug=True, host='127.0.0.1', port=5000)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
