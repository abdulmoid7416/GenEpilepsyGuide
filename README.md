# Epilepsy Treatment Planner

A Streamlit application for personalized treatment recommendations in genetic epilepsy using LangGraph, ClinVar API, and evidence-based guidelines.

## Features

- **Input Parser**: Uses Groq LLM (Qwen3-32b) to extract structured data from patient descriptions
- **ClinVar Integration**: Queries NCBI ClinVar API for variant information and associated epilepsy syndromes
- **Doctor-Friendly Reports**: LLM-formatted clinical summaries from ClinVar data
- **Treatment Recommender**: Uses Pinecone vector store with NICE and ILAE treatment guidelines
- **LangGraph Workflow**: Sequential agentic flow with three specialized agents
- **Interactive UI**: Two-step process with syndrome selection and tabbed variant displays

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

**Note:** ClinVar access via NCBI E-utilities does not require an API key for basic queries.

### 3. Pinecone Setup

Ensure you have a Pinecone index named `epilepsy-guidelines` with the following treatment guidelines:
- NICE Guidelines - Epilepsies in children, young people and adults (2025)
- ILAE Treatment Guidelines: Evidence-based Analysis of Antiepileptic Drug Efficacy and Effectiveness as Initial Monotherapy for Epileptic Seizures and Syndromes (2006)

The index should use the `NeuML/pubmedbert-base-embeddings` model for embeddings.

### 4. Run the Application

```bash
streamlit run app.py
```

## Usage

### Step 1: Query ClinVar

1. Enter a **gene symbol** (e.g., `SCN1A`, `TSC2`)
2. Enter a **variant** notation (e.g., `p.Arg905Gln` or `c.3733C>T`)
3. Click "Search ClinVar"

The app will:
- Query ClinVar for matching variants
- Generate doctor-friendly clinical reports for each variant found
- Extract epilepsy-related syndromes using LLM analysis
- Display results in tabbed interface (if multiple variants found)

### Step 2: Select Syndrome & Get Treatment

1. Review the doctor-friendly ClinVar summary
2. Select an epilepsy syndrome from the dropdown
3. Click "Recommend Treatment"

The app will:
- Query the Pinecone vector database for relevant treatment guidelines
- Generate evidence-based treatment recommendations with citations
- Display step-by-step treatment pathways

## Example Input

**Gene:** `SCN1A`  
**Variant:** `c.3733C>T`

Or enter a full patient description to be parsed:
```
3-year-old female with SCN1A c.3733C>T variant presenting with febrile seizures, myoclonic jerks, and developmental delay.
```

## Architecture

The application uses a modular architecture with three dedicated agent files and LangGraph orchestration:

### Agent Modules (`agents/` directory)
- `input_parser.py` - Input parsing agent (Groq LLM)
- `clinvar_agent.py` - ClinVar API integration and syndrome extraction
- `treatment_recommender.py` - Treatment recommendation agent (Pinecone + Groq)

### Workflow Orchestration
The main `app.py` contains the orchestrator code using LangGraph with a linear workflow:

```
input_parser → clinvar_agent → treatment_recommender → END
```

### State Management
```python
class AgentState(TypedDict):
    input: str
    parsed_data: Dict[str, Any]
    clinvar_results: List[Dict[str, Any]]
    clinvar_syndromes: List[str]
    treatments: str
```

## API Integrations

- **Groq**: Qwen3-32b model for text parsing and report formatting
- **ClinVar (NCBI E-utilities)**: Variant information and clinical significance
- **Pinecone**: Vector similarity search for treatment guidelines
- **HuggingFace**: PubMedBERT embeddings for biomedical text

## Treatment Guidelines Database

The Pinecone vector store currently contains:
- **NICE Guidelines** - Epilepsies in children, young people and adults (2025)
- **ILAE Treatment Guidelines** - Evidence-based Analysis of Antiepileptic Drug Efficacy and Effectiveness as Initial Monotherapy for Epileptic Seizures and Syndromes (2006)

Treatment recommendations are based solely on these two authoritative sources.

## Key Features

### ClinVar Integration
- Queries NCBI ClinVar API using gene and variant information
- Retrieves comprehensive variant data including:
  - Clinical significance and review status
  - Associated epilepsy syndromes
  - HPO phenotypes
  - Molecular consequences
  - Cross-references (dbSNP, ClinGen, OMIM, Orphanet)

### Doctor-Friendly Reports
- LLM-generated clinical summaries structured for epileptologists
- Sections include:
  - Variant summary with chromosomal location
  - Clinical significance and pathogenicity
  - Epilepsy syndromes with database references
  - Clinical phenotypes (HPO terms)
  - Other associated conditions
  - Molecular details

### Syndrome Extraction
- Automated identification of epilepsy-related syndromes from ClinVar data
- LLM-based filtering for epilepsy-specific conditions
- Excludes generic terms and non-epilepsy phenotypes

### Evidence-Based Treatment
- Vector similarity search against treatment guidelines
- Context-aware recommendations based on syndrome
- Citations to specific guideline sections and page numbers
- Multiple syndrome support with separate treatment pathways

## Error Handling

- Clear error messages when API keys are missing or invalid
- Graceful handling of ClinVar API errors
- User-friendly error display in Streamlit interface
- Informative warnings when no results are found

## Programmatic Usage

Use the demo script to test the workflow programmatically:

```bash
python demo.py
```

## Project Structure

```
Epilepsy Guide/
├── app.py                          # Main Streamlit app (orchestrator)
├── demo.py                         # Programmatic demo script
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── agents/                         # Agent modules
│   ├── __init__.py                # Package initialization
│   ├── input_parser.py            # Input parsing agent
│   ├── clinvar_agent.py           # ClinVar API agent
│   └── treatment_recommender.py   # Treatment recommendation agent
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── modular_architecture.txt        # Architecture diagram
└── workflow_diagram.txt            # Workflow visualization
```

## Technical Details

### LLM Configuration
- **Model**: Qwen3-32b via Groq API
- **Temperature**: 0 for parsing, 0.1 for ClinVar formatting, 0.3 for treatment recommendations
- **Max tokens**: 2000 for reports, 1000 for treatments

### Embedding Configuration
- **Model**: NeuML/pubmedbert-base-embeddings
- **Vector Store**: Pinecone with namespace "guidelines"
- **Top-k retrieval**: 5 chunks per syndrome query

### ClinVar Query
- **API**: NCBI E-utilities (esearch + esummary)
- **Database**: clinvar
- **Max results**: 20 variants per query
- **Format**: JSON with full metadata

## Future Enhancements

Potential additions to consider:
- Clinical trial matching integration
- OMIM phenotype series lookup
- Additional treatment guideline sources
- Multi-gene panel analysis
- Family history visualization
- Treatment outcome tracking

## License

This is a research/educational tool. Always consult with qualified medical professionals for clinical decision-making.
