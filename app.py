import streamlit as st
import os
import json
from typing import Dict, List, TypedDict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

# Import agents
from agents import (
    InputParserAgent,
    ClinVarAgent,
    TreatmentRecommenderAgent
)

# Load environment variables
load_dotenv()

# Initialize environment variables with fallbacks
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Still needed for embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# State definition for LangGraph
class AgentState(TypedDict):
    input: str
    parsed_data: Dict[str, Any]
    clinvar_results: List[Dict[str, Any]]
    clinvar_syndromes: List[str]
    treatments: str

class EpilepsyTreatmentPlanner:
    """Main orchestrator class for the epilepsy treatment planner workflow"""
    
    def __init__(self):
        """Initialize the treatment planner with all agents"""
        # Initialize all agents
        self.input_parser_agent = InputParserAgent(GROQ_API_KEY)
        self.clinvar_agent = ClinVarAgent()  # ClinVar doesn't require API key for basic queries
        self.treatment_agent = TreatmentRecommenderAgent(GROQ_API_KEY, PINECONE_API_KEY)
    
    
    def create_workflow(self) -> StateGraph:
        """Create and return the LangGraph workflow with all nodes and edges"""
        workflow = StateGraph(AgentState)
        
        # Add nodes using agent methods
        workflow.add_node("input_parser", self.input_parser_agent.process)
        workflow.add_node("clinvar_agent", self.clinvar_agent.process)
        workflow.add_node("treatment_recommender", self.treatment_agent.process)
        
        # Add edges
        workflow.set_entry_point("input_parser")
        workflow.add_edge("input_parser", "clinvar_agent")
        workflow.add_edge("clinvar_agent", "treatment_recommender")
        workflow.add_edge("treatment_recommender", END)
        
        return workflow.compile()

def main():
    st.set_page_config(
        page_title="Epilepsy Treatment Planner",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 GenEpilepsyGuide: Intelligent Epilepsy Treatment Guidance")
    st.markdown("Provide a gene and variant to retrieve ClinVar data, then select a syndrome for treatment recommendations.")
    
    # Initialize the treatment planner
    if "planner" not in st.session_state:
        st.session_state.planner = EpilepsyTreatmentPlanner()
        st.session_state.workflow = st.session_state.planner.create_workflow()
    
    # Input section for gene and variant
    st.subheader("ClinVar Query")
    col_g, col_v = st.columns(2)
    with col_g:
        gene = st.text_input("Gene symbol", placeholder="e.g., TSC2, SCN1A")
    with col_v:
        variant = st.text_input("Variant", placeholder="e.g., p.Arg905Gln or c.3733C>T")

    if st.button("Search ClinVar", type="primary"):
        if not gene.strip() or not variant.strip():
            st.error("Please enter both gene and variant.")
            return

        with st.spinner("Querying ClinVar..."):
            try:
                # Build minimal state expected by ClinVarAgent
                state = {
                    "parsed_data": {
                        "gene": gene.strip(),
                        "variant": variant.strip(),
                        "phenotypes": []
                    }
                }

                clinvar_out = st.session_state.planner.clinvar_agent.process(state)

                # Save to session
                st.session_state.clinvar_results = clinvar_out.get("clinvar_results", [])
                st.session_state.clinvar_syndromes = clinvar_out.get("clinvar_syndromes", [])
                st.session_state.clinvar_raw = clinvar_out.get("clinvar_raw", {})
                st.session_state.clinvar_doctor_reports = clinvar_out.get("clinvar_doctor_reports", [])
                st.session_state.gene = gene.strip()
                st.session_state.variant = variant.strip()

                st.success("ClinVar data retrieved.")
            except Exception as e:
                st.error(f"Error querying ClinVar: {e}")
                st.stop()

    # Show ClinVar results if available
    if "clinvar_raw" in st.session_state:
        st.subheader("ClinVar Summary (Doctor-Friendly)")
        
        doctor_reports = st.session_state.get("clinvar_doctor_reports", [])
        
        if doctor_reports:
            # If multiple entries, use tabs
            if len(doctor_reports) > 1:
                tab_labels = [f"Entry {i+1}: {report['title'][:50]}..." if len(report['title']) > 50 else f"Entry {i+1}: {report['title']}" 
                             for i, report in enumerate(doctor_reports)]
                tabs = st.tabs(tab_labels)
                
                for i, tab in enumerate(tabs):
                    with tab:
                        report_text = doctor_reports[i]["report"]
                        st.text_area(
                            f"Variant ID: {doctor_reports[i]['variant_id']}", 
                            value=report_text, 
                            height=400,
                            key=f"report_{i}",
                            disabled=True
                        )
            else:
                # Single entry - just display it
                report_text = doctor_reports[0]["report"]
                st.text_area(
                    f"Variant ID: {doctor_reports[0]['variant_id']}", 
                    value=report_text, 
                    height=400,
                    disabled=True
                )
        else:
            st.warning("No Entry Found")

        # Optional: raw JSON view
        with st.expander("View Raw ClinVar JSON"):
            raw_text = json.dumps(st.session_state.clinvar_raw, indent=2)
            st.text_area("ClinVar JSON", value=raw_text, height=250, disabled=True)

        # Query another entry button
        if st.button("Query another ClinVar Entry", type="secondary"):
            # Clear session state to reset the page
            for key in list(st.session_state.keys()):
                if key.startswith('clinvar_') or key in ['gene', 'variant']:
                    del st.session_state[key]
            st.rerun()

        # Syndromes selection
        st.subheader("Select Syndrome")
        syndromes = st.session_state.get("clinvar_syndromes", [])
        if syndromes:
            selected_syndrome = st.selectbox("Syndrome", options=sorted(syndromes))
            
            # Guidelines database information
            st.info("""
            **Guidelines Database Information:**
            
            At the moment, the Guidelines database consists of 2 documents:
            - NICE Guidelines - Epilepsies in children, young people and adults (2025)
            - ILAE Treatment Guidelines: Evidence-based Analysis of Antiepileptic Drug Efficacy and Effectiveness as Initial Monotherapy for Epileptic Seizures and Syndromes (2006)
            
            We would recommend treatments based on the data from these 2 documents only.
            """)
            
            if st.button("Recommend Treatment"):
                with st.spinner("Generating treatment recommendations..."):
                    try:
                        # Build state for TreatmentRecommender to process only selected syndrome
                        input_summary = f"Gene: {st.session_state.get('gene','')}, Variant: {st.session_state.get('variant','')}"
                        t_state = {
                            "input": input_summary,
                            "clinvar_syndromes": [selected_syndrome]
                        }
                        t_out = st.session_state.planner.treatment_agent.process(t_state)

                        st.subheader("💊 Treatment Recommendations")
                        st.markdown(t_out.get("treatments", "No recommendations available."))
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
        else:
            st.info("No syndromes identified from ClinVar for the given gene/variant.")

if __name__ == "__main__":
    main()
