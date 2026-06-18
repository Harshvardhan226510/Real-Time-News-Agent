# Real-Time Autonomous News Research Agent

A stateful, multi-tool autonomous AI agent engineered using the LangGraph orchestration framework, Google Gemini-2.5-flash foundational model, and local ChromaDB vector memory. The architecture utilizes a ReAct loop to dynamically execute web searches, self-index retrieved articles, and maintain a long-term context memory warehouse.

## 🧠 System Architecture Overview
The agent operates via a stateful runtime sequence:
1. **Dynamic Search Tool:** Queries live web channels using the Tavily Search API to bypass model training cutoff limitations.
2. **On-the-Fly Vectorization:** Splits fresh context payloads recursively into 500-character chunks with a 50-token overlap, indexing them dynamically inside a `Chroma` database.
3. **Retrieval Memory Tool:** Queries the local persistent directory to recall past session references during interactive text evaluation loops.

## 🛠️ Technical Stack
- **Orchestration Framework:** LangGraph (`create_react_agent`)
- **Foundational LLM Layer:** Google GenAI (`gemini-2.5-flash`)
- **Vector Storage Engine:** ChromaDB (Persisted state execution)
- **Embedding Matrix Generation:** Hugging Face Transformers (`all-MiniLM-L6-v2`)
- **Interface UI:** Gradio Web Layout Engine

## 📁 Repository Manifest
```text
├── app.py             # Autonomous agent logic, system tools, and Gradio wrapper interface
├── requirements.txt   # Complete environment package manifests
└── README.md          # Technical architectural documentation
```

## ⚙️ Environment Configuration & Architecture Verification
To instantiate this autonomous runner environment locally, execute the following commands within your deployment workflow:
- **Clone the Source Tree:**
  ```
    git clone [https://github.com/Harshvardhan226510/Real-Time-News-Agent.git](https://github.com/Harshvardhan226510/Real-Time-News-Agent.git)
    cd Real-Time-News-Agent
  ```
- **Inject Runtime Platform Environment Credentials:**
  ```
    export GOOGLE_API_KEY="your_gemini_api_key_here"
    export TAVILY_API_KEY="your_tavily_api_key_here"
  ```
- **Install Core Manifest Dependencies & Run App:**
  ```
    pip install -r requirements.txt
    python app.py
  ```
