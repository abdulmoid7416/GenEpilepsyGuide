"""
Treatment Recommender Agent - Generates treatment recommendations using Pinecone vector store
"""
# agents/treatment_recommender_agent.py

import os
from typing import Dict, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re

class TreatmentRecommenderAgent:
    def __init__(self, groq_api_key: str, pinecone_api_key: str):
        load_dotenv()
        
        # Environment variables
        self.GROQ_API_KEY = groq_api_key
        self.PINECONE_API_KEY = pinecone_api_key

        # Constants from vector DB creation
        self.INDEX_NAME = "epilepsy-guidelines"
        self.NAMESPACE = "guidelines"
        self.EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

        # Set up embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)

        # Set up Pinecone vector store
        self.vectorstore = PineconeVectorStore(
            index_name=self.INDEX_NAME,
            embedding=self.embeddings,
            namespace=self.NAMESPACE,
            pinecone_api_key=self.PINECONE_API_KEY
        )

        # Set up direct Pinecone connection for retrieval
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index = self.pc.Index(self.INDEX_NAME)

        # Set up LLM (using Qwen from Groq)
        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            groq_api_key=self.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=1000
        )

        # Prompt template for generating treatment pathway recommendation
        self.recommendation_prompt = ChatPromptTemplate.from_template("""
            You are an expert in epilepsy treatment guidelines. Based on the following retrieved context from official guidelines, 
            recommend a treatment pathway for the epilepsy syndromes: {syndromes}.

            Retrieved context (each chunk followed by its source):
            {context}

            Provide a clear, step-by-step treatment pathway for each syndrome listed, citing relevant guidelines where possible. 
            For each key piece of information, cite the source in the exact format: [document name(year), section name, page number] 
            immediately after the statement. Use the sources provided in the context.

            Keep the response concise and focused on evidence-based recommendations.
            """)

        # Chain for generating recommendation
        self.recommendation_chain = (
            self.recommendation_prompt
            | self.llm
            | StrOutputParser()
        )

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state to generate treatment recommendations"""
        # Get syndromes from ClinVar agent and patient input
        clinvar_syndromes = state.get("clinvar_syndromes", [])
        patient_input = state.get("input", "")
        
        if not clinvar_syndromes:
            return {"treatments": "No syndromes identified to recommend treatments for."}

        try:
            # Process each syndrome individually for more targeted recommendations
            all_treatments = []
            
            for syndrome in clinvar_syndromes:
                print(f"\n🔍 Processing syndrome: {syndrome}")
                
                # Query vector database for chunks containing treatment info for this syndrome
                query_text = f"How would you treat patient: {patient_input} possibly diagnosed by {syndrome}"
                
                # Generate embedding for the query
                query_embedding = self.embeddings.embed_query(query_text)
                
                # Query Pinecone directly
                results = self.index.query(
                    vector=query_embedding,
                    top_k=5,
                    namespace=self.NAMESPACE,
                    include_metadata=True
                )
                
                if not results.matches:
                    print(f"  No treatment information found in vector database")
                    # Add a section indicating no treatment information found
                    syndrome_section = f"## Treatment for {syndrome}\n\nNo treatment information found in vector database.\n"
                    all_treatments.append(syndrome_section)
                    continue
                
                print(f"  Found {len(results.matches)} chunks with treatment information")
                
                # Format context with sources from the retrieved chunks
                context_parts = []
                for i, match in enumerate(results.matches):
                    # Extract text content from metadata
                    if 'text' in match.metadata:
                        content = match.metadata['text']
                    else:
                        # Fallback to _node_content if text not directly available
                        import json
                        node_content = json.loads(match.metadata.get('_node_content', '{}'))
                        content = node_content.get('text', 'No content available')
                    
                    source_info = self._format_source(match.metadata)
                    context_parts.append(f"{content}\nSource: {source_info}")
                
                context = "\n\n".join(context_parts)
                
                # Generate treatment recommendations for this syndrome using the chunks as context
                syndrome_treatments = self.recommendation_chain.invoke({
                    "syndromes": syndrome,
                    "context": context
                })
                
                # Remove thinking tags if present
                if "<think>" in syndrome_treatments and "</think>" in syndrome_treatments:
                    syndrome_treatments = re.sub(r'<think>.*?</think>', '', syndrome_treatments, flags=re.DOTALL)
                    syndrome_treatments = syndrome_treatments.strip()
                
                # Add syndrome-specific section
                syndrome_section = f"## Treatment for {syndrome}\n\n{syndrome_treatments}\n"
                all_treatments.append(syndrome_section)
            
            # Combine all syndrome-specific treatments
            if all_treatments:
                combined_treatments = "\n".join(all_treatments)
                return {"treatments": combined_treatments}
            else:
                return {"treatments": "No treatment guidelines found for the identified syndromes."}
            
        except Exception as e:
            print(f"Error in treatment recommender: {e}")
            return {"treatments": f"Error generating treatment recommendations: {str(e)}"}
    
    def _format_source(self, metadata: Dict) -> str:
        """Format source information from metadata using document_name and page_number"""
        # Get document name and page number from metadata
        doc_name = metadata.get('document_name', 'Unknown Document')
        page_number = metadata.get('page_number', 'Unknown')
        
        # Return formatted source as: document_name, page_number
        return f"{doc_name}, page {page_number}"