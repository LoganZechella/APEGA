# APEGA: Automated Practice Exam Generation Agent

## Overview

The Automated Practice Exam Generation Agent (APEGA) is an advanced artificial intelligence system designed to autonomously create high-quality, interactive practice examinations. Given a natural language description of a target standardized test (such as the "Certified Licensing Professional (CLP) exam"), APEGA generates comprehensive study materials including multiple-choice questions, explanations, and interactive HTML/PDF outputs.

## System Architecture

APEGA follows a modular architecture with distinct processing stages:

1. **Knowledge Ingestion and Vectorization**
   - Processes source documents (PDFs, etc.)
   - Chunks text and generates embeddings
   - Stores vectors in a vector database (Qdrant)

2. **Contextual Understanding (RAG Engine)**
   - Performs hybrid search (dense + sparse)
   - Re-ranks results for relevance
   - Uses Gemini 2.5 Pro for deep content analysis

3. **Intelligent Prompt Engineering**
   - Dynamically generates optimal prompts
   - Uses meta-prompting to refine instructions for LLMs

4. **Question and Distractor Generation**
   - Creates MCQs with Gemini 2.5 Pro
   - Generates plausible, challenging distractors

5. **Quality Assurance**
   - Evaluates questions using OpenAI's o4-mini as LLM-as-Judge
   - Checks factual accuracy, clarity, distractor quality, and more
   - Provides feedback for iterative improvement

6. **Output Formatting**
   - Generates interactive HTML exams
   - Creates printable PDF versions
   - Produces structured JSON output

7. **Orchestration and Workflow Management**
   - Coordinates all components with LangGraph
   - Manages state and handles errors

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key
- Google AI API key
- Qdrant (local or cloud)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/apega.git
   cd apega
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     ```
     OPENAI_API_KEY=your-openai-api-key
     GOOGLE_API_KEY=your-google-api-key
     ```

5. Configure Qdrant:
   - For local installation, run:
     ```
     docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
     ```
   - For cloud, add to your `.env`:
     ```
     QDRANT_URL=your-qdrant-cloud-url
     QDRANT_API_KEY=your-qdrant-api-key
     ```

## Usage

### Ingesting Documents

Before generating exams, you need to ingest source documents:

```
python ingest.py --directory /path/to/your/documents
```

This will process documents, generate embeddings, and store them in Qdrant.

### Generating Practice Exams

Generate a practice exam with:

```
python generate.py --query "Generate a CLP practice exam focusing on intellectual property valuation" --questions 10
```

Options:
- `--query`: Natural language description of the desired exam
- `--questions`: Number of questions to generate (default: 10)
- `--output-dir`: Custom output directory
- `--verbose`: Enable detailed logging

### Outputs

The system generates:
- Interactive HTML exam with automatic scoring
- Print-friendly PDF version
- Structured JSON data

## Configuration

Edit `config/config.env` to customize settings:

- Document processing parameters
- Vector database settings
- Model parameters
- Output formatting options

## AI Models Used

- **Google Gemini 2.5 Pro**: For deep content analysis and question generation
- **OpenAI o4-mini**: For quality assurance and prompt optimization
- **OpenAI text-embedding-3-large**: For generating vector embeddings

## References

Based on the extensive development guide "Automated Practice Exam Generation Agent (APEGA): System Architecture and Development Guide"

## License

[MIT License]

## Contributors

- [Your Name]