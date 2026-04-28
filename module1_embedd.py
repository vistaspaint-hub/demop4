"""
Module 1: Pinecone Embedding Generator
This module handles loading HR data, creating embeddings, and uploading to Pinecone.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


class PineconeEmbeddingManager:
    """
    Manages the creation and upload of embeddings to Pinecone vector database
    """
    
    def __init__(self, index_name="hr-employee-data", model_name='all-MiniLM-L6-v2'):
        """
        Initialize the Pinecone Embedding Manager
        
        Args:
            index_name (str): Name of the Pinecone index
            model_name (str): Name of the sentence transformer model
        """
        self.index_name = index_name
        self.model_name = model_name
        self.model = None
        self.pc = None
        self.index = None
        
        # Load Pinecone API key
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
    
    def load_data(self, csv_path):
        """
        Load HR dataset from CSV file
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} employee records")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def create_employee_text(self, row):
        """
        Create a comprehensive text description for each employee record
        
        Args:
            row: DataFrame row containing employee data
            
        Returns:
            str: Formatted text representation
        """
        text = f"""
        Employee ID: {row['employee_id']}
        Age: {row['age']} years old
        Gender: {row['gender']}
        Department: {row['department']}
        Role: {row['role']}
        Experience: {row['experience_years']} years
        
        Compensation:
        - Gross Salary: ${row['gross_salary']:,.2f}
        - Tax Amount: ${row['tax_amount']:,.2f}
        - Deductions: ${row['deductions']:,.2f}
        - Bonus: ${row['bonus']:,.2f}
        - Net Salary: ${row['net_salary']:,.2f}
        
        Work Details:
        - Work Hours per Week: {row['work_hours_per_week']} hours
        - Leave Taken: {row['leave_taken']} days
        - Remote Work Frequency: {row['remote_work_frequency']}
        - Training Hours: {row['training_hours']} hours
        
        Performance Metrics:
        - Performance Score: {row['performance_score']}/5.0
        - Productivity Score: {row['productivity_score']}/100
        - Feedback Score: {row['feedback_score']}/10
        - Suggestions Score: {row['suggestions_score']}/10
        - Job Satisfaction: {row['job_satisfaction']}/5.0
        - Promotion Eligible: {row['promotion_eligible']}
        """
        return text.strip()
    
    def prepare_text_data(self, df):
        """
        Prepare text representations for all employees
        
        Args:
            df (pd.DataFrame): Employee dataframe
            
        Returns:
            pd.DataFrame: Dataframe with text_representation column added
        """
        print("\nCreating text representations for all employees...")
        df['text_representation'] = df.apply(self.create_employee_text, axis=1)
        print(f"✅ Text representations created for {len(df)} employees")
        return df
    
    def load_embedding_model(self):
        """
        Load the sentence transformer model for creating embeddings
        """
        print(f"\nLoading sentence transformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for text data
        
        Args:
            texts (list): List of text strings
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if self.model is None:
            self.load_embedding_model()
        
        print("\nGenerating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def initialize_pinecone(self):
        """
        Initialize connection to Pinecone and create/connect to index
        """
        print("\nInitializing Pinecone...")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Check if index exists, if not create it
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for 'all-MiniLM-L6-v2'
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"✅ Index '{self.index_name}' created successfully!")
        else:
            print(f"✅ Index '{self.index_name}' already exists!")
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        print(f"✅ Connected to index. Stats: {self.index.describe_index_stats()}")
    
    def prepare_vectors(self, df, embeddings):
        """
        Prepare vectors with metadata for Pinecone upload
        
        Args:
            df (pd.DataFrame): Employee dataframe
            embeddings (np.ndarray): Generated embeddings
            
        Returns:
            list: List of vector dictionaries
        """
        print("\nPreparing vectors for upload...")
        vectors_to_upsert = []
        
        for idx, row in df.iterrows():
            vector_id = str(row['employee_id'])
            embedding = embeddings[idx].tolist()
            
            # Create metadata with all employee information
            metadata = {
                'employee_id': str(row['employee_id']),
                'age': int(row['age']),
                'gender': str(row['gender']),
                'department': str(row['department']),
                'role': str(row['role']),
                'experience_years': int(row['experience_years']),
                'gross_salary': float(row['gross_salary']),
                'net_salary': float(row['net_salary']),
                'bonus': float(row['bonus']),
                'work_hours_per_week': int(row['work_hours_per_week']),
                'leave_taken': int(row['leave_taken']),
                'performance_score': float(row['performance_score']),
                'productivity_score': float(row['productivity_score']),
                'feedback_score': float(row['feedback_score']),
                'suggestions_score': float(row['suggestions_score']),
                'job_satisfaction': float(row['job_satisfaction']),
                'promotion_eligible': str(row['promotion_eligible']),
                'remote_work_frequency': str(row['remote_work_frequency']),
                'training_hours': int(row['training_hours']),
                'text': row['text_representation']
            }
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        print(f"✅ Prepared {len(vectors_to_upsert)} vectors for upload")
        return vectors_to_upsert
    
    def upload_to_pinecone(self, vectors, batch_size=100):
        """
        Upload vectors to Pinecone in batches
        
        Args:
            vectors (list): List of vector dictionaries
            batch_size (int): Batch size for upload
        """
        if self.index is None:
            self.initialize_pinecone()
        
        total_vectors = len(vectors)
        print(f"\nUploading {total_vectors} vectors to Pinecone in batches of {batch_size}...")
        
        for i in tqdm(range(0, total_vectors, batch_size)):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print("\n✅ Upload completed successfully!")
        print(f"Final index stats: {self.index.describe_index_stats()}")
    
    def process_and_upload(self, csv_path, batch_size=100):
        """
        Complete pipeline: Load data, create embeddings, upload to Pinecone
        
        Args:
            csv_path (str): Path to CSV file
            batch_size (int): Batch size for upload
        """
        print("="*80)
        print("PINECONE EMBEDDING PIPELINE")
        print("="*80)
        
        # Load data
        df = self.load_data(csv_path)
        
        # Prepare text representations
        df = self.prepare_text_data(df)
        
        # Load model
        self.load_embedding_model()
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df['text_representation'].tolist())
        
        # Initialize Pinecone
        self.initialize_pinecone()
        
        # Prepare vectors
        vectors = self.prepare_vectors(df, embeddings)
        
        # Upload to Pinecone
        self.upload_to_pinecone(vectors, batch_size)
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return df, embeddings
    
    def get_index(self):
        """
        Get the Pinecone index object
        
        Returns:
            Pinecone Index object
        """
        if self.index is None:
            self.initialize_pinecone()
        return self.index
    
    def get_model(self):
        """
        Get the sentence transformer model
        
        Returns:
            SentenceTransformer model
        """
        if self.model is None:
            self.load_embedding_model()
        return self.model


def main():
    """
    Example usage of the PineconeEmbeddingManager
    """
    # Initialize the embedding manager
    manager = PineconeEmbeddingManager(
        index_name="hr-employee-data",
        model_name='all-MiniLM-L6-v2'
    )
    
    # Process and upload the HR dataset
    csv_path = 'hr_productivity_payroll_dataset_600(in) (1).csv'
    manager.process_and_upload(csv_path)


if __name__ == "__main__":
    main()
