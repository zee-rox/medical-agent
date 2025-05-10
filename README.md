# Medical Agent

An AI-powered diagnostic assistant that processes medical queries and chest X-ray images to provide comprehensive medical assessments.

## 🏥 Project Overview

Medical Agent is an advanced medical diagnostic tool that leverages various AI models and techniques to analyze medical queries and chest X-ray images. The system employs a sophisticated agent-based architecture to process different modalities of medical information and produce evidence-based diagnostic reports.

Key features:
- Natural language processing of clinical queries
- Chest X-ray image analysis
- Integration of multiple data sources for diagnosis
- Self-reflection and refinement of diagnostic outputs
- Structured diagnostic reporting

## 📋 Project Structure

```
medical-agent/
├── data/                      # Data storage and indices
│   ├── agent_memory.db        # Agent memory database
│   ├── agent_memory.json      # Agent memory in JSON format
│   ├── chunks.pkl             # Data chunks for retrieval
│   ├── faiss_index_*.idx      # FAISS indices for vector similarity search
│   ├── merged_df_diagnosis.pkl# Merged diagnosis dataframe
│   ├── diagnosis_results/     # Output directory for diagnosis results
│   └── visualizations/        # Visualization outputs
├── src/                       # Source code
│   ├── agent/                 # Agent core functionality
│   │   ├── Core.py            # Core agent implementation
│   │   ├── Memory.py          # Agent memory management
│   │   ├── Models.py          # Model definitions
│   │   ├── Planning.py        # Planning functionality
│   │   ├── Reflection.py      # Self-reflection capabilities
│   │   └── Roles.py           # Agent role definitions
│   ├── imaging/               # Image processing
│   │   └── DetectXRAY.py      # X-ray analysis
│   ├── models/                # AI model loaders
│   │   ├── LoadEmbeddingModel.py  # Embedding model initialization
│   │   └── LoadLLM.py         # LLM initialization
│   ├── reports/               # Report generation
│   │   ├── CombinedReport.py  # Combined report generation
│   │   ├── ImageReport.py     # Image-based reporting
│   │   └── TextReport.py      # Text-based reporting
│   ├── retrieval/             # Information retrieval
│   │   ├── DataLoader.py      # Data loading utilities
│   │   └── Search.py          # Search functionality
│   ├── schema/                # Schema definitions
│   │   └── Tools.py           # Tool schemas
│   └── utils/                 # Utility functions
│       ├── DiagnosisExporter.py  # Export diagnosis results
│       └── TextProcessing.py     # Text processing utilities
├── main.py                    # Main application entry point
├── pyproject.toml             # Project dependencies and metadata
├── README.md                  # Project documentation
└── uv.lock                    # Dependency lock file
```

## 🛠️ Technology Stack

- **Language Models**: Ollama (llama3.2), OpenAI
- **Embeddings**: Hugging Face, Sentence Transformers
- **Vector Database**: FAISS
- **Orchestration**: LangGraph
- **NLP**: spaCy, SciSpaCy
- **Scientific Libraries**: NumPy, Pandas, scikit-learn
- **Machine Learning**: PyTorch, Transformers
- **Retrieval**: BM25, RAGatouille, ColBERT

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-agent.git
cd medical-agent
```

2. Create a virtual environment using uv:
```bash
uv venv
```

3. Activate the virtual environment:
```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

4. Install dependencies:
```bash
uv pip install -e .
```

5. Download necessary model files and data (if applicable):
```bash
# If you have a script for this
python -m scripts.download_models
```

## 🖥️ Usage

Run the medical agent with:

```bash
python main.py
```

The application will:
1. Prompt you for a clinical query
2. Ask for an optional chest X-ray image file path
3. Process your inputs through the diagnostic workflow
4. Generate a comprehensive diagnostic report
5. Save a workflow visualization in the data/visualizations directory

### Example Session

```
🏥 Starting Enhanced Medical Agent Application

Enter your clinical query: Patient experiencing shortness of breath and chest pain for the past 3 days

🖼️ Enter chest X-ray image file path (or press Enter to skip): /path/to/xray.jpg

🔍 Chest X-ray Detection Results:
Chest X-ray Findings: Pleural Effusion, Pneumonia

🚀 Initiating diagnostic workflow...

✅ Final Diagnosis Report:
[Diagnostic report will appear here]

📊 Diagnostic Quality Assessment:
Overall Quality Score: 0.85/1.0
✓ The diagnosis met quality standards

🏥 Medical Agent completed!
```

## 📊 Diagnostic Workflows

The system supports different workflows based on the available inputs:

- **W1** (all sources): Combines patient history, X-ray, and clinical query
- **W2** (ID + image): Uses patient history and X-ray
- **W3** (ID + query): Uses patient history and clinical query
- **W4** (image + query): Uses X-ray and clinical query
- **W5** (image only): Uses only X-ray
- **W6** (query only): Uses only clinical query

## 🔄 System Architecture

The system uses a LangGraph-based architecture with the following components:

1. **Planning**: Creates a step-by-step plan for diagnostic assessment
2. **Execution**: Executes each plan step using specialized tools
3. **Reflection**: Evaluates the quality of the diagnostic report
4. **Refinement**: Improves the report based on feedback

The agent operates in different roles throughout the process, including:
- Medical Planning Agent
- Medical Action Agent
- Medical Quality Reviewer
- Medical Report Refiner

## 📄 License

[Include your license information here]

## 👥 Contributors

[List contributors here]

## 🙏 Acknowledgements

- The medical knowledge and terminology used in this system comes from public medical resources
- This project uses several open-source libraries listed in the Technology Stack section