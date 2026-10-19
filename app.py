
import os
import streamlit as st
from dotenv import load_dotenv
from graph_qa import MovieGraphQA

load_dotenv()

# Load Streamlit Cloud secrets into environment variables.
try:
    for key in [
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    # Local .env configuration can still be used.
    pass


st.set_page_config(
    page_title="Movie Knowledge Graph RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: clamp(1.7rem, 3vw, 2.5rem) !important;
        line-height: 1.25 !important;
        overflow-wrap: anywhere;
    }

    [data-testid="stSidebar"] {
        min-width: 270px;
    }

    [data-testid="stChatInput"] {
        max-width: 100%;
    }

    .subtitle {
        color: #777f8b;
        font-size: 1rem;
        line-height: 1.6;
    }

    .small-note {
        color: #777f8b;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_qa_engine():
    return MovieGraphQA()


def get_engine():
    """Return a cached engine, with configuration errors handled by the UI."""
    return get_qa_engine()


def reset_chat():
    st.session_state.messages = []


# Initialize session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# Sidebar.
with st.sidebar:
    st.title("🎬 Movie RAG")
    st.caption("Neo4j Knowledge Graph Assistant")

    st.divider()

    st.subheader("About this project")
    st.write("**Backend:** Neo4j Movies graph")
    st.write("**Retrieval:** Parameterized Cypher")
    st.write("**Answer generation:** OpenAI API")

    st.divider()

    st.subheader("Database status")

    if st.button(
        "Check database connection",
        use_container_width=True,
    ):
        try:
            engine = get_engine()
            status = engine.check_connection()
            st.success(status)
        except Exception:
            st.error("The request failed.")

            with st.expander("Technical details", expanded=True):
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    st.subheader("Try a question")

    suggestions = [
        "Who directed The Matrix?",
        "Who acted in The Matrix?",
        "What movies did Keanu Reeves act in?",
        "Recommend movies like The Matrix",
    ]

    for suggestion in suggestions:
        if st.button(
            suggestion,
            key=f"suggestion_{suggestion}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggestion

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        reset_chat()
        st.session_state.pending_question = None
        st.rerun()

    st.caption("Answers depend on data available in your graph.")


# Main content.
st.title("🎬 Movie Knowledge Graph RAG")
st.markdown(
    '<div class="subtitle">'
    "Ask about movies, actors, directors, and recommendations. "
    "Answers are grounded in retrieved Neo4j graph records."
    "</div>",
    unsafe_allow_html=True,
)

st.write("")


# Display previous messages.
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("route"):
            st.caption(f"Query type: {message['route']}")

        if message.get("records"):
            with st.expander("View retrieved graph records"):
                st.json(message["records"])


# Chat input.
question = st.chat_input(
    "Ask a question about a movie, actor, or director..."
)

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            engine = get_engine()

            with st.spinner(
                "Retrieving graph data and generating an answer..."
            ):
                result = engine.answer(question)

            st.markdown(result["answer"])
            st.caption(f"Query type: {result['route']}")

            if result["records"]:
                with st.expander("View retrieved graph records"):
                    st.json(result["records"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "route": result["route"],
                "records": result["records"],
            })

        except Exception:
            st.error(
                "The request failed. Check your Neo4j connection, "
                "API configuration, and terminal logs."
            )

            with st.expander("Technical details"):
                import traceback
                st.code(traceback.format_exc())
