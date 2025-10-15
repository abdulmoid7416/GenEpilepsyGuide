"""
Configuration file for the Epilepsy Treatment Planner
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# Model Configuration
GROQ_MODEL = "qwen/qwen3-32b"
BIOMEDICAL_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

# Pinecone Configuration
PINECONE_INDEX_NAME = "epilepsy-guidelines"

# API Endpoints
OMIM_API_URL = "https://api.omim.org/api/entry/search"
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Application Configuration
MAX_CLINICAL_TRIALS = 5
MAX_TREATMENT_RECOMMENDATIONS = 3
MAX_SYNDROMES = 3
