# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
# Option A: Use the setup script
chmod +x setup.sh
./setup.sh

# Option B: Manual installation
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit the `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
OMIM_API_KEY=your_omim_api_key_here
```

### 3. Run the Application
```bash
streamlit run app.py
```

## 📋 What You Get

- **Input Parser**: Extracts gene, variant, demographics, and phenotypes
- **OMIM Integration**: Finds associated genetic syndromes
- **Clinical Trials**: Searches for relevant recruiting trials
- **Treatment Recommendations**: Evidence-based treatment suggestions

## 🧪 Test the Installation
```bash
python test_installation.py
```

## 🎯 Example Usage

Enter a patient description like:
```
25-year-old female with TSC2 variant p.Arg905Gln presenting with focal seizures, skin lesions, and developmental delay.
```

The app will:
1. Parse the input into structured data
2. Query OMIM for associated syndromes
3. Find relevant clinical trials
4. Generate treatment recommendations

## 🔧 Troubleshooting

- **Missing API keys**: App will fail with clear error messages
- **Pinecone not available**: App will fail - Pinecone is required for treatment recommendations
- **Installation issues**: Run `python setup.py` for guided setup
- **API errors**: Check your API keys and internet connection

## 📁 Project Structure

```
Epilepsy Treatment Planner/
├── app.py                 # Main Streamlit application (orchestrator only)
├── agents/                # Agent modules directory
│   ├── __init__.py       # Package initialization
│   ├── input_parser.py   # Input parsing agent
│   ├── omim_agent.py     # OMIM API agent
│   ├── clinical_trial_matcher.py  # Clinical trials agent
│   └── treatment_recommender.py   # Treatment recommendation agent
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── setup.py              # Python setup script
├── setup.sh              # Shell setup script
├── test_installation.py  # Installation test
├── demo.py               # Programmatic demo
├── README.md             # Detailed documentation
├── QUICKSTART.md         # This file
└── workflow_diagram.txt  # Workflow visualization
```

## 🎨 Features

- **LangGraph Workflow**: State-based agentic processing
- **Conditional Routing**: Smart branching based on data availability
- **Error Handling**: Graceful fallbacks and user-friendly messages
- **Modern UI**: Clean Streamlit interface with organized sections
- **API Integration**: Multiple external services for comprehensive analysis
