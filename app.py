"""
CineGraph AI - Neo4j Movie Knowledge Graph RAG Application.
A modern, luxury cinema-grade Streamlit web application providing:
1. Conversational Chat UI with Grounded RAG Answers
2. 1-Click Suggestion Chips for instant discovery
3. Interactive Subgraph Network Visualizer (vis.js) with stabilized physics
4. Parameterized Cypher Query Studio with 11+ curated templates
5. Real-time Knowledge Graph Explorer with Search & Filter
6. Seamless Neo4j Desktop / AuraDB Connection & 1-Click Seeder
"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv

from backend.graph_db import get_graph_service, Neo4jGraphService
from backend.rag_engine import MovieRAGEngine
from backend.query_templates import list_templates, get_template

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CineGraph AI - Movie Knowledge Graph RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .stApp {
        background-color: #080b11;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(14, 165, 233, 0.08) 0px, transparent 50%);
        color: #e2e8f0;
    }

    /* Top Hero Header */
    .hero-container {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    .status-live {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    
    .status-embedded {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .metric-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: left;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-box:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-sub {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 2px;
    }

    /* Suggestion Chips */
    .chip-btn {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        text-align: left !important;
        padding: 8px 14px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    .chip-btn:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
    }

    /* Chat & Results Container */
    .result-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }

    .badge-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-right: 8px;
    }

    /* Sidebar Clean styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f17;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize State
if "graph_service" not in st.session_state:
    st.session_state.graph_service = get_graph_service()

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = MovieRAGEngine(graph_service=st.session_state.graph_service)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def render_visjs_graph(nodes, edges, height=440):
    """Render interactive graph network using vis.js with smooth stabilization."""
    if not nodes:
        st.info("ℹ️ No specific graph nodes to visualize for this query.")
        return

    vis_nodes = []
    for n in nodes:
        is_movie = n.get("type") == "Movie"
        vis_nodes.append({
            "id": n["id"],
            "label": n["label"],
            "shape": "box" if is_movie else "dot",
            "size": 22,
            "color": {
                "background": "#f59e0b" if is_movie else "#6366f1",
                "border": "#fbbf24" if is_movie else "#818cf8",
                "highlight": {"background": "#ec4899", "border": "#f472b6"}
            },
            "font": {"color": "#ffffff", "size": 13, "face": "Plus Jakarta Sans, sans-serif"},
            "title": f"<b>Type:</b> {n.get('type')}<br><b>Details:</b> {n.get('year', '')}"
        })

    vis_edges = []
    for e in edges:
        vis_edges.append({
            "from": e["from"],
            "to": e["to"],
            "label": e.get("label", ""),
            "arrows": "to",
            "color": {"color": "#64748b", "highlight": "#ec4899"},
            "font": {"color": "#94a3b8", "size": 10, "align": "middle"}
        })

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #network {{
                width: 100%;
                height: {height}px;
                background-color: #0b0f17;
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
            }}
            .legend {{
                display: flex;
                gap: 16px;
                margin-bottom: 8px;
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 12px;
                color: #94a3b8;
            }}
            .dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="legend">
            <div><span class="dot" style="background:#f59e0b; border-radius:2px;"></span> Movie Nodes</div>
            <div><span class="dot" style="background:#6366f1;"></span> Person Nodes</div>
            <div><span style="color:#ec4899;">➔</span> Relationships (ACTED_IN, DIRECTED, etc.)</div>
        </div>
        <div id="network"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({json.dumps(vis_nodes)});
            var edges = new vis.DataSet({json.dumps(vis_edges)});
            var container = document.getElementById('network');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{
                        gravitationalConstant: -40,
                        centralGravity: 0.01,
                        springLength: 100,
                        springConstant: 0.08
                    }},
                    stabilization: {{ iterations: 80 }}
                }},
                interaction: {{
                    hover: true,
                    zoomView: true,
                    navigationButtons: true,
                    tooltipDelay: 100
                }}
            }};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height + 35)


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.markdown("### 🎬 CineGraph Control Panel")
    st.caption("Manage database connection, seeding, and LLM.")

    with st.expander("🔌 Neo4j Database Settings", expanded=True):
        neo_uri = st.text_input("Bolt URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        neo_user = st.text_input("Username", value=os.getenv("NEO4J_USERNAME", "neo4j"))
        neo_pass = st.text_input("Password", value=os.getenv("NEO4J_PASSWORD", "password"), type="password")
        neo_db = st.text_input("Database Name", value=os.getenv("NEO4J_DATABASE", "neo4j"))

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔗 Connect", use_container_width=True):
                with st.spinner("Connecting..."):
                    srv = Neo4jGraphService(uri=neo_uri, user=neo_user, password=neo_pass, database=neo_db)
                    st.session_state.graph_service = srv
                    st.session_state.rag_engine = MovieRAGEngine(graph_service=srv)
                    st.rerun()

        with col_b2:
            if st.button("🌱 Seed Graph", use_container_width=True):
                with st.spinner("Seeding database..."):
                    ok, msg = st.session_state.graph_service.seed_database()
                    if ok:
                        st.success("Graph seeded successfully!")
                    else:
                        st.warning(msg)

    with st.expander("🤖 Gemini LLM Settings", expanded=False):
        gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password", help="Leave blank to use the high-precision deterministic graph engine offline.")
        if gemini_key:
            st.session_state.rag_engine.set_api_key(gemini_key)
            st.success("API key updated!")

    st.divider()
    status = st.session_state.graph_service.get_status()
    st.markdown("#### System Health")
    if status["status"] == "connected":
        st.markdown(f'<div class="status-pill status-live">● Live: {status["uri"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-embedded">● Embedded Graph (Offline Ready)</div>', unsafe_allow_html=True)
    
    st.write("")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.pending_query = None
        st.rerun()


# ---------------- TOP HERO BANNER ----------------
status = st.session_state.graph_service.get_status()
schema = st.session_state.graph_service.get_schema()

status_badge_html = (
    f'<div class="status-pill status-live">● Connected: {status["uri"]}</div>'
    if status["status"] == "connected"
    else '<div class="status-pill status-embedded">● Mode: Embedded Neo4j Movies Graph</div>'
)

st.markdown(f"""
<div class="hero-container">
    <div>
        <h1 class="hero-title">🎬 CineGraph AI</h1>
        <div class="hero-subtitle">
            Graph-Powered Movie Intelligence, Retrieval-Augmented Generation & Cypher Traversal
        </div>
    </div>
    <div>
        {status_badge_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- METRICS STRIP ----------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{schema.get('node_count', 171)}</div>
        <div class="metric-sub">Knowledge Graph Nodes</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{schema.get('relationship_count', 253)}</div>
        <div class="metric-sub">Graph Relationships</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">28+</div>
        <div class="metric-sub">Featured Movies</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">11+</div>
        <div class="metric-sub">Cypher Templates</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ---------------- NAVIGATION TABS ----------------
tab_chat, tab_templates, tab_explorer, tab_guide = st.tabs([
    "💬 Ask & Discover (Graph RAG)",
    "📜 Cypher Query Studio",
    "🕸️ Knowledge Graph Explorer",
    "📘 Project Documentation"
])


# ---------------- TAB 1: ASK & DISCOVER ----------------
with tab_chat:
    st.markdown("### 🍿 Instant Suggestion Prompts")
    st.caption("Click any query below to run the Graph RAG pipeline in 1 click:")

    # 1-Click Suggestion Chips
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎬 Keanu Reeves filmography & roles", use_container_width=True):
            st.session_state.pending_query = "What movies did Keanu Reeves act in and what were his roles?"
        if st.button("🍿 Recommend movies for Tom Cruise fans", use_container_width=True):
            st.session_state.pending_query = "Recommend movies for a fan of Tom Cruise"

    with c2:
        if st.button("👥 Tom Hanks most frequent co-stars", use_container_width=True):
            st.session_state.pending_query = "Who has Tom Hanks co-starred with the most?"
        if st.button("🎥 Who directed and wrote A Few Good Men?", use_container_width=True):
            st.session_state.pending_query = "Who starred in, directed, and wrote A Few Good Men?"

    with c3:
        if st.button("🔗 Kevin Bacon to Keanu Reeves path", use_container_width=True):
            st.session_state.pending_query = "How are Kevin Bacon and Keanu Reeves connected in the graph?"
        if st.button("⭐ Highest rated movies by reviews", use_container_width=True):
            st.session_state.pending_query = "What are the top rated movies by user reviews?"

    st.write("")

    # Process prompt from chat_input OR clicked chip
    chat_input_val = st.chat_input("Ask a question about movies, actors, directors, or recommendations...")
    active_prompt = chat_input_val or st.session_state.pending_query

    if active_prompt:
        # Reset pending query
        st.session_state.pending_query = None

        with st.spinner(f"Analyzing graph for: \"{active_prompt}\"..."):
            response_data = st.session_state.rag_engine.answer_question(active_prompt)
            st.session_state.chat_messages.append(response_data)

    # Render Conversation Messages
    if not st.session_state.chat_messages:
        st.info("💡 Try clicking one of the suggestion buttons above or type your question in the chat input below!")
    else:
        for idx, item in enumerate(st.session_state.chat_messages):
            # User Message
            with st.chat_message("user", avatar="👤"):
                st.write(item["question"])

            # Assistant RAG Message
            with st.chat_message("assistant", avatar="🎬"):
                st.markdown(f'<span class="badge-tag">{item["routing_mode"]}</span> <span class="badge-tag">⏱️ {item["execution_time_ms"]} ms</span> <span class="badge-tag">📊 {item["record_count"]} Records</span>', unsafe_allow_html=True)
                
                # Grounded Answer
                st.markdown(item["answer"])

                # Expandable Subgraph & Cypher Inspector
                col_sub1, col_sub2 = st.columns([1, 1])
                with col_sub1:
                    with st.expander("🕸️ View Interactive Subgraph", expanded=True):
                        vis_data = item.get("graph_visualization", {})
                        render_visjs_graph(vis_data.get("nodes", []), vis_data.get("edges", []), height=320)

                with col_sub2:
                    with st.expander("🔍 Executed Cypher & Retrieved Records", expanded=True):
                        st.markdown("**Executed Cypher Query:**")
                        st.code(item["cypher_query"], language="cypher")
                        if item["records"]:
                            st.markdown("**Retrieved Records:**")
                            st.dataframe(pd.DataFrame(item["records"]), use_container_width=True, height=200)


# ---------------- TAB 2: CYPHER QUERY STUDIO ----------------
with tab_templates:
    st.markdown("### 📜 Curated Cypher Query Templates")
    st.write("Browse, customize parameters, and execute verified Cypher query templates against the knowledge graph.")

    templates = list_templates()
    template_labels = [f"[{t['category']}] {t['title']}" for t in templates]
    t_idx = st.selectbox("Select Template to Inspect & Run:", range(len(templates)), format_func=lambda i: template_labels[i])
    t_curr = templates[t_idx]

    col_tp1, col_tp2 = st.columns([1.1, 0.9])

    with col_tp1:
        st.markdown(f"#### {t_curr['title']}")
        st.write(t_curr["description"])
        st.info(f"**Sample Question:** \"{t_curr['sample_question']}\"")
        st.markdown("**Cypher Query Code:**")
        st.code(t_curr["cypher"].strip(), language="cypher")

    with col_tp2:
        st.markdown("#### Query Parameters")
        p_values = {}
        for k, v in t_curr.get("params", {}).items():
            p_values[k] = st.text_input(f"${k}", value=str(v))

        if st.button("⚡ Execute Template", type="primary", use_container_width=True):
            with st.spinner("Running Cypher query..."):
                t_results = st.session_state.graph_service.execute_template(
                    t_curr["id"],
                    p_values,
                    t_curr["cypher"]
                )
                st.success(f"Execution finished: Retrieved {len(t_results)} record(s)")
                if t_results:
                    st.dataframe(pd.DataFrame(t_results), use_container_width=True)
                else:
                    st.warning("No records matched the provided parameter values.")


# ---------------- TAB 3: GRAPH EXPLORER ----------------
with tab_explorer:
    st.markdown("### 🕸️ Knowledge Graph Catalog & Schema")
    st.write("Explore canonical movies, actors, and directors indexed in the Neo4j Movies Knowledge Graph.")

    tab_m, tab_p = st.tabs(["🎬 Movies Directory", "👤 Persons Directory"])

    with tab_m:
        movies_dict = st.session_state.graph_service.fallback_engine.movies
        df_movies = pd.DataFrame(list(movies_dict.values()))
        search_m = st.text_input("Search Movies by title or tagline:", placeholder="e.g. Matrix, Speed, Love...")
        if search_m:
            df_movies = df_movies[
                df_movies["title"].str.contains(search_m, case=False, na=False) |
                df_movies["tagline"].str.contains(search_m, case=False, na=False)
            ]
        st.dataframe(df_movies[["title", "released", "tagline"]], use_container_width=True, height=400)

    with tab_p:
        persons_dict = st.session_state.graph_service.fallback_engine.persons
        df_persons = pd.DataFrame(list(persons_dict.values()))
        search_p = st.text_input("Search Persons by name:", placeholder="e.g. Keanu, Tom, Carrie...")
        if search_p:
            df_persons = df_persons[df_persons["name"].str.contains(search_p, case=False, na=False)]
        st.dataframe(df_persons[["name", "born"]], use_container_width=True, height=400)


# ---------------- TAB 4: PROJECT DOCUMENTATION ----------------
with tab_guide:
    st.markdown("### 📘 System Architecture & Project Documentation")
    st.markdown("""
    #### 1. Why Graph RAG for Movies?
    Traditional RAG relies on vector similarity over text chunks. However, cinema queries often require:
    - Multi-hop traversal (e.g. *Person -> ACTED_IN -> Movie <- ACTED_IN <- Person*)
    - Exact relationship attribution (Actor vs. Director vs. Producer)
    - Shortest path calculations (Bacon Number)
    
    **CineGraph AI** addresses this by converting questions into Cypher graph queries and using LLMs exclusively for context synthesis.

    #### 2. Key Components Delivered
    - **Neo4j Movie Graph Seeder:** `scripts/seed_neo4j.py` & `data/movies_dataset.cql`
    - **Cypher Query Templates:** 11+ verified query templates for QA, Recommendations, and Network Analysis
    - **Interactive Subgraph Visualizer:** Physics-stabilized `vis.js` graph network view
    - **Dual-Mode Backend:** Seamlessly switches between live Neo4j and embedded in-memory graph
    
    *For full technical details, consult [`PROJECT_REPORT.md`](file:///c:/Users/user/Desktop/Lab%203/PROJECT_REPORT.md).*
    """)
