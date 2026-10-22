# 🎬 CineGraph AI: Movie Knowledge Graph RAG

An intelligent **Retrieval-Augmented Generation (RAG)** application powered by the **Neo4j Movies Knowledge Graph** and **Large Language Models**. Converts natural language movie queries into Cypher graph traversals, retrieves verified entity relationships, and generates grounded, hallucination-free answers and recommendations.

---

## 🌟 Key Features

- **Knowledge Graph Backend:** Uses the canonical Neo4j Movies dataset (`:Movie`, `:Person`, `:ACTED_IN`, `:DIRECTED`, `:PRODUCED`, `:WROTE`, `:REVIEWED`, `:FOLLOWS`).
- **Hybrid Query Routing:**
  - **Deterministic Cypher Templates (11+ templates):** Instant, zero-latency execution for common QA and recommendation patterns.
  - **Dynamic Text-to-Cypher:** Few-shot LLM translation for novel, multi-condition queries with Cypher safety guardrails.
- **Graph-Based Recommendations:** Collaborative filtering via 2-hop co-star networks and director style matching.
- **Shortest Path & Network Analysis:** Computes Bacon Numbers and degrees of separation between any two actors.
- **Interactive Web Interface:** Streamlit web app with interactive `vis.js` graph network visualization, parameter controls, and Cypher studio.
- **Terminal CLI Prototype:** Lightweight interactive terminal runner for quick QA and automated testing.
- **Dual-Mode Execution (Zero Friction):** Seamlessly connects to **Neo4j Desktop** or **Neo4j AuraDB**, and includes an automatic **Embedded In-Memory Graph fallback** so the app runs out-of-the-box even without a running Neo4j database!

---

## 📁 Repository Structure

```
Lab 3/
├── app.py                     # Streamlit Web Application (Modern UI + vis.js Graph)
├── cli_app.py                 # Interactive Terminal CLI Prototype
├── PROJECT_REPORT.md          # Comprehensive Project Report
├── README.md                  # Quickstart & Setup Guide
├── requirements.txt           # Python package dependencies
├── .env.example               # Environment variables template
├── data/
│   ├── movies_dataset.cql     # Official Neo4j Movies dataset Cypher script
│   └── seed_data.py           # Embedded dataset definitions for offline mode
├── backend/
│   ├── __init__.py            # Package initialization
│   ├── graph_db.py            # Neo4j Driver & In-Memory Graph Engine fallback
│   ├── query_templates.py     # 11+ Curated Cypher query templates
│   ├── text2cypher.py         # Entity extraction & Text-to-Cypher generator
│   └── rag_engine.py          # RAG Orchestrator (Retrieval + Grounded LLM)
└── scripts/
    └── seed_neo4j.py          # 1-click Neo4j seeding utility (Desktop/AuraDB)
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure Python 3.10+ is installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application

Launch the modern Web UI:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### 3. Run the CLI Prototype

For terminal-based testing:

```bash
python cli_app.py
```

---

## 🔌 Connecting to Neo4j (Optional)

The application automatically defaults to an **Embedded In-Memory Graph (Neo4j Movies Dataset)** if Neo4j is offline, allowing immediate testing with zero setup.

To connect to a live Neo4j instance:

### Option A: Neo4j Desktop / Local Neo4j
1. Start your local Neo4j database (default Bolt URI: `bolt://localhost:7687`).
2. Seed the database with the Movies dataset:
   ```bash
   python scripts/seed_neo4j.py --uri bolt://localhost:7687 --user neo4j --password your_password
   ```
3. Set your credentials in `.env` or input them directly into the Web UI sidebar!

### Option B: Neo4j AuraDB (Cloud Free Tier)
1. Create a free instance at [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/).
2. Seed the AuraDB instance:
   ```bash
   python scripts/seed_neo4j.py --uri neo4j+s://<dbid>.databases.neo4j.io --user neo4j --password your_password
   ```
3. Enter your URI and credentials into the app settings.

---

## 🤖 LLM API Configuration (Optional)

- Set your Google Gemini API key in `.env`:
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
- Or input it directly into the **Settings & Controls** sidebar in the Streamlit app.
- *Note:* If no API key is provided, the engine runs its high-precision **Deterministic Graph Grounding Engine** offline.

---

## 💡 Example Queries to Test

| Query Category | Example Natural Language Prompt |
| :--- | :--- |
| **Filmography & Roles** | *"What movies did Keanu Reeves act in and what were his roles?"* |
| **Director's Works** | *"Which movies were directed by Lilly Wachowski?"* |
| **Cast & Crew Inspection** | *"Who starred in and directed The Matrix?"* |
| **Collaborative Recommendations** | *"Recommend movies for a fan of Tom Cruise"* |
| **Co-star Network** | *"Who has Tom Hanks co-starred with the most?"* |
| **Degrees of Separation** | *"How are Kevin Bacon and Keanu Reeves connected in the graph?"* |
| **Shared Filmography** | *"Did Tom Cruise and Jack Nicholson act in any movie together?"* |
| **Reviews & Ratings** | *"What are the top rated movies by user reviews?"* |
| **Multitalented Creators** | *"Which people in the database have both acted in and directed movies?"* |

---

## 📑 Project Report

For the in-depth architectural breakdown, Cypher template catalog, schema diagrams, and academic evaluation, refer to [`PROJECT_REPORT.md`](file:///c:/Users/user/Desktop/Lab%203/PROJECT_REPORT.md).
