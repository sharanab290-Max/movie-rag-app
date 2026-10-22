"""
CLI Prototype for Neo4j Movies Graph RAG Application.
Allows interactive natural language QA, template execution, and connection inspection.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.graph_db import get_graph_service
from backend.rag_engine import MovieRAGEngine
from backend.query_templates import list_templates, get_template


def print_banner():
    banner = r"""
========================================================================
   ____ _             ____                 _       _    ___ 
  / ___(_)_ __   ___ / ___|_ __ __ _ _ __ | |__   / \  |_ _|
 | |   | | '_ \ / _ \ |  _| '__/ _` | '_ \| '_ \ / _ \  | | 
 | |___| | | | |  __/ |_| | | | (_| | |_) | | | / ___ \ | | 
  \____|_|_| |_|\___|\____|_|  \__,_| .__/|_| |_/_/   \_\___|
                                    |_|                      
   Movie Knowledge Graph RAG (Neo4j & LLM)
========================================================================
"""
    print(banner)


def display_menu():
    print("\nOptions:")
    print("  [1] Ask a natural language question (e.g., 'What movies did Keanu Reeves star in?')")
    print("  [2] Ask for movie recommendations (e.g., 'Recommend movies for a fan of Tom Cruise')")
    print("  [3] Explore Bacon Number / Connection (e.g., 'How are Kevin Bacon and Keanu Reeves connected?')")
    print("  [4] Browse & execute Cypher query templates")
    print("  [5] Check Neo4j connection & schema status")
    print("  [6] Exit")
    print("-" * 72)


def handle_template_browser(engine: MovieRAGEngine):
    templates = list_templates()
    print("\n--- Available Cypher Query Templates ---")
    for idx, t in enumerate(templates, 1):
        print(f"[{idx}] {t['title']} ({t['category']}) - {t['description']}")
    
    choice = input("\nEnter template number to run (or 'b' to go back): ").strip()
    if choice.lower() == 'b':
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(templates):
            selected = templates[idx]
            print(f"\nSelected: {selected['title']}")
            print(f"Sample Question: \"{selected['sample_question']}\"")
            print(f"Cypher Query Template:\n{selected['cypher'].strip()}")
            
            run_sample = input("\nRun with sample parameters? (y/n): ").strip().lower()
            if run_sample == 'y':
                res = engine.answer_question(selected['sample_question'])
                print("\n" + "=" * 50)
                print(f"Executed Cypher: {res['cypher_query']}")
                print(f"Engine: {res['synthesis_source']} ({res['execution_time_ms']} ms)")
                print("=" * 50)
                print(res['answer'])
                print("=" * 50)
    except Exception as e:
        print(f"Invalid selection: {e}")


def main():
    load_dotenv()
    print_banner()

    graph_service = get_graph_service()
    status = graph_service.get_status()
    print(f"Active Backend: {status['backend']}")
    if status['status'] == 'connected':
        print(f"Connected to Neo4j instance at {status['uri']}")
    else:
        print("Note: Running in Embedded Neo4j Movies Graph mode (no external Neo4j required).")
        print("You can connect to a live Neo4j Desktop or AuraDB database at any time.")

    engine = MovieRAGEngine(graph_service=graph_service)

    while True:
        display_menu()
        choice = input("Enter choice (1-6): ").strip()

        if choice in ['1', '2', '3']:
            sample_prompts = {
                '1': "What movies did Keanu Reeves act in and what were his roles?",
                '2': "Recommend movies for a fan of Tom Cruise",
                '3': "How are Kevin Bacon and Keanu Reeves connected in the graph?"
            }
            default_q = sample_prompts[choice]
            user_q = input(f"\nEnter question [Press Enter for: '{default_q}']: ").strip()
            if not user_q:
                user_q = default_q

            print(f"\n[Processing] Analyzing: \"{user_q}\"...")
            res = engine.answer_question(user_q)

            print("\n" + "=" * 72)
            print(f"Intent / Routing: {res['routing_mode']}")
            print(f"Cypher Query:\n{res['cypher_query']}")
            print(f"Retrieved Records: {res['record_count']}")
            print(f"Synthesis Source: {res['synthesis_source']} | Latency: {res['execution_time_ms']} ms")
            print("=" * 72)
            print("\n" + res['answer'] + "\n")
            print("=" * 72)

        elif choice == '4':
            handle_template_browser(engine)

        elif choice == '5':
            schema = graph_service.get_schema()
            st = graph_service.get_status()
            print("\n--- Neo4j Knowledge Graph Status ---")
            print(f"Backend: {st['backend']}")
            print(f"URI: {st['uri']}")
            print(f"Total Nodes: {schema.get('node_count')}")
            print(f"Total Relationships: {schema.get('relationship_count')}")
            print(f"Labels: {schema.get('labels')}")
            print(f"Relationship Types: {schema.get('relationship_types')}")

        elif choice == '6':
            print("\nExiting CineGraph AI. Goodbye!\n")
            break
        else:
            print("Invalid option. Please enter 1-6.")


if __name__ == "__main__":
    main()
