"""
HR RAG Chatbot Script
This script uses the module1_embedd module to access Pinecone embeddings
and implements a RAG chatbot with OpenAI for answering HR queries.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# Import the embedding module
from module1_embedd import PineconeEmbeddingManager

# Load environment variables
load_dotenv()


class HRChatbot:
    """
    HR Chatbot using RAG (Retrieval-Augmented Generation)
    Uses embeddings from Pinecone and OpenAI for response generation
    """
    
    def __init__(self, index_name="hr-employee-data"):
        """
        Initialize the HR Chatbot
        
        Args:
            index_name (str): Name of the Pinecone index to use
        """
        print("Initializing HR Chatbot...")
        
        # Load OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize the embedding manager from module1
        self.embedding_manager = PineconeEmbeddingManager(index_name=index_name)
        
        # Get the Pinecone index
        self.index = self.embedding_manager.get_index()
        
        # Get the embedding model
        self.embedding_model = self.embedding_manager.get_model()
        
        print("✅ HR Chatbot initialized successfully!")
        print(f"Connected to Pinecone index: {index_name}")
        print(f"Index stats: {self.index.describe_index_stats()}\n")
    
    def retrieve_relevant_employees(self, query, top_k=5):
        """
        Retrieve the most relevant employees from Pinecone based on the query
        
        Args:
            query (str): User's natural language query
            top_k (int): Number of top results to retrieve
            
        Returns:
            list: List of matching employee records with metadata
        """
        print(f"🔍 Searching for relevant employees...")
        
        # Convert query to embedding using the model from module1
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search Pinecone for similar vectors
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"✅ Found {len(results['matches'])} relevant employees\n")
        return results['matches']
    
    def format_context(self, matches):
        """
        Format retrieved employee data as context for the LLM
        
        Args:
            matches (list): List of matching employee records from Pinecone
            
        Returns:
            str: Formatted context string
        """
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
            context += f"Bonus: ${metadata.get('bonus', 0):,.2f}\n"
            context += f"Work Hours per Week: {metadata.get('work_hours_per_week', 'N/A')} hours\n"
            context += f"Leave Taken: {metadata.get('leave_taken', 'N/A')} days\n"
            context += f"Performance Score: {metadata.get('performance_score', 'N/A')}/5.0\n"
            context += f"Productivity Score: {metadata.get('productivity_score', 'N/A')}/100\n"
            context += f"Feedback Score: {metadata.get('feedback_score', 'N/A')}/10\n"
            context += f"Suggestions Score: {metadata.get('suggestions_score', 'N/A')}/10\n"
            context += f"Job Satisfaction: {metadata.get('job_satisfaction', 'N/A')}/5.0\n"
            context += f"Promotion Eligible: {metadata.get('promotion_eligible', 'N/A')}\n"
            context += f"Remote Work Frequency: {metadata.get('remote_work_frequency', 'N/A')}\n"
            context += f"Training Hours: {metadata.get('training_hours', 'N/A')} hours\n"
            context += "\n"
        
        return context
    
    def generate_response(self, query, context, model="gpt-4o-mini"):
        """
        Generate a natural language response using OpenAI GPT
        
        Args:
            query (str): User's original query
            context (str): Retrieved employee data as context
            model (str): OpenAI model to use
            
        Returns:
            str: Generated response
        """
        print("🤖 Generating response with OpenAI...\n")
        
        system_prompt = """You are an intelligent HR assistant AI that helps answer questions about employee data.
        You have access to comprehensive employee records including demographics, roles, compensation, 
        performance metrics, and work-related information.
        
        Your responsibilities:
        1. Analyze the provided employee data carefully
        2. Answer the user's question accurately and completely based on the data
        3. Provide specific details, numbers, and names when relevant
        4. Be professional, helpful, and concise
        5. If the data doesn't contain enough information to fully answer the question, clearly state what's missing
        6. When comparing employees, highlight key differences
        7. Format your responses in a clear, easy-to-read manner
        """
        
        user_prompt = f"""Based on the following employee data, please answer this question:

QUESTION: {query}

EMPLOYEE DATA:
{context}

Please provide a comprehensive, clear answer based on the data above."""
        
        # Call OpenAI API
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, query, top_k=5, verbose=False, show_context=False):
        """
        Main chat function implementing the full RAG pipeline
        
        Args:
            query (str): User's question
            top_k (int): Number of employees to retrieve
            verbose (bool): Whether to show detailed steps
            show_context (bool): Whether to display the retrieved context
            
        Returns:
            dict: Response containing answer and metadata
        """
        print("\n" + "="*80)
        print(f"💬 USER QUERY: {query}")
        print("="*80 + "\n")
        
        # Step 1: RETRIEVAL - Get relevant employees from Pinecone
        if verbose:
            print("📊 STEP 1: RETRIEVAL")
        matches = self.retrieve_relevant_employees(query, top_k)
        
        # Step 2: AUGMENTATION - Format context
        if verbose:
            print("📝 STEP 2: AUGMENTATION")
        context = self.format_context(matches)
        
        if show_context:
            print("\n" + "-"*80)
            print("RETRIEVED CONTEXT:")
            print("-"*80)
            print(context)
            print("-"*80 + "\n")
        
        # Step 3: GENERATION - Generate response using OpenAI
        if verbose:
            print("🎯 STEP 3: GENERATION")
        answer = self.generate_response(query, context)
        
        print("="*80)
        print("✨ CHATBOT RESPONSE:")
        print("="*80)
        print(answer)
        print("="*80 + "\n")
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_employees": matches,
            "num_employees_found": len(matches),
            "context": context
        }
    
    def interactive_mode(self):
        """
        Start an interactive chat session
        """
        print("\n" + "="*80)
        print("🤖 HR CHATBOT - INTERACTIVE MODE")
        print("="*80)
        print("Ask questions about employee data. Type 'quit' or 'exit' to end the session.")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using HR Chatbot! Goodbye!\n")
                    break
                
                if not user_input:
                    print("Please enter a question.\n")
                    continue
                
                # Process the query
                self.chat(user_input, top_k=5)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def main():
    """
    Main function to demonstrate the chatbot
    """
    # Initialize the chatbot
    chatbot = HRChatbot(index_name="hr-employee-data")
    
    # Example queries
    example_queries = [
        "Show me employees with high performance scores in the Engineering department",
        "Which employees are eligible for promotion?",
        "Who are the top 3 earners in the company?",
        "Find employees with low job satisfaction scores",
        "Show me remote workers with more than 10 years of experience"
    ]
    
    print("\n" + "="*80)
    print("🎯 RUNNING EXAMPLE QUERIES")
    print("="*80 + "\n")
    
    # Run example queries
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}/{len(example_queries)}")
        print(f"{'='*80}")
        chatbot.chat(query, top_k=3, verbose=False)
        
        if i < len(example_queries):
            input("\nPress Enter to continue to next example...\n")
    
    # Ask if user wants to try interactive mode
    print("\n" + "="*80)
    response = input("Would you like to try interactive mode? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        chatbot.interactive_mode()
    else:
        print("\n👋 Thank you for using HR Chatbot!\n")


if __name__ == "__main__":
    main()
