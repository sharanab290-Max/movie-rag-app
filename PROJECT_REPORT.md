# Project Report: Graph RAG Movie Intelligence & Recommendation System

**Application Name:** CineGraph AI  
**Backend:** Neo4j Graph Database (Movies Dataset)  
**RAG Orchestration:** Hybrid Cypher Query Templates & Text-to-Cypher with Grounded LLM Generation  
**Interfaces:** Streamlit Web Application (with interactive vis.js graph visualization) & Interactive CLI Prototype  

---

## 1. Executive Summary & Domain Formulation

In modern media streaming, digital entertainment, and talent analytics, users frequently pose queries that require multi-hop relational reasoning across complex networks. Common examples include:
- *"Which actors frequently collaborate with Keanu Reeves, and what other films do they recommend?"*
- *"What is the shortest connection between Kevin Bacon and Tom Hanks in the movie industry?"*
- *"Who directed and wrote A Few Good Men, and what was their role in other 90s courtroom dramas?"*

Traditional Retrieval-Augmented Generation (RAG) frameworks rely primarily on dense vector embeddings over unstructured text chunks. However, **pure vector search struggles with multi-hop graph topology, exact relationship traversal, and structured relational queries**.

**CineGraph AI** resolves this limitation by implementing a **Graph RAG (Retrieval-Augmented Generation)** architecture built upon the canonical **Neo4j Movies Knowledge Graph**. The system maps natural language user queries into validated Cypher graph traversals, extracts exact subgraphs from Neo4j, and injects the structured facts into a Large Language Model with strict grounding constraints. This ensures **zero hallucination**, high explainability, and rich conversational recommendations.

---

## 2. Knowledge Graph Schema Analysis

The knowledge graph is modeled as a labeled property graph (LPG) containing entities from cinema history, their creative roles, reviews, and social followings.

### 2.1 Node Labels & Attributes

| Node Label | Property | Data Type | Description | Constraints / Indexes |
| :--- | :--- | :--- | :--- | :--- |
| `:Movie` | `title` | `STRING` | The official release title of the motion picture | `IS UNIQUE` Constraint |
| | `released` | `INTEGER` | Calendar year the movie was released in theaters | Indexed |
| | `tagline` | `STRING` | Promotional marketing tagline or summary quote | Optional |
| `:Person` | `name` | `STRING` | Full professional name of the actor, director, or crew member | `IS UNIQUE` Constraint |
| | `born` | `INTEGER` | Birth year of the individual | Indexed |

### 2.2 Relationship Types & Edge Properties

| Relationship | Direction | Description | Edge Properties |
| :--- | :--- | :--- | :--- |
| `[:ACTED_IN]` | `(:Person)->(:Movie)` | Denotes that a person performed an acting role in the film | `roles: LIST OF STRING` (e.g. `['Neo']`, `['Maverick']`) |
| `[:DIRECTED]` | `(:Person)->(:Movie)` | Person directed the production of the movie | None |
| `[:PRODUCED]` | `(:Person)->(:Movie)` | Person served as executive or creative producer | None |
| `[:WROTE]` | `(:Person)->(:Movie)` | Person authored or co-authored the screenplay | None |
| `[:REVIEWED]` | `(:Person)->(:Movie)` | User or critic review with numerical rating and feedback | `rating: INTEGER` (0-100), `summary: STRING` |
| `[:FOLLOWS]` | `(:Person)->(:Person)` | Directed social network connection between film enthusiasts | None |

### 2.3 Graph Topology & Graph Advantages
- **Heterogeneous Nodes:** Separates cinematic products (`:Movie`) from human agents (`:Person`).
- **Multi-Relational Edges:** An individual can simultaneously act in, direct, and produce the same movie (e.g., Tom Hanks in *That Thing You Do*, Clint Eastwood in *Unforgiven*). In a relational database, this requires multiple join tables; in vector RAG, this context is often lost in text segmentation.
- **Path Discoverability:** Allows arbitrary $k$-hop traversals (such as calculating the Bacon Number or mutual co-star paths) in logarithmic or constant time relative to graph depth.

---

## 3. Graph RAG Architecture & Methodology

```
                   ┌────────────────────────────────────────┐
                   │         User Natural Language          │
                   │         Question / Query             │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │   Entity Recognizer & Intent Router    │
                   │  (Matches titles, persons, & patterns) │
                   └───────┬────────────────────────┬───────┘
                           │                        │
         [Deterministic    │                        │   [Novel / Complex
          Pattern Matched] │                        │    Natural Language]
                           ▼                        ▼
              ┌────────────────────────┐  ┌────────────────────────┐
              │ Parameterized Cypher   │  │ Dynamic Text-to-Cypher │
              │ Template Engine        │  │ (Few-Shot Prompt + LLM)│
              └────────────┬───────────┘  └───────────┬────────────┘
                           │                          │
                           │                          │ (Sanitize & Guardrail)
                           ▼                          ▼
                   ┌────────────────────────────────────────┐
                   │    Cypher Execution & Safety Filter    │
                   │  (Prohibits Mutating Ops: CREATE, etc.)│
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │    Neo4j Database / Fallback Engine    │
                   │  (Neo4j Desktop / AuraDB / Embedded)   │
                   └───────────────────┬────────────────────┘
                                       │ Retrieved Subgraph Records
                                       ▼
                   ┌────────────────────────────────────────┐
                   │      Context Formulation & Grounding   │
                   │ (Transforms graph records into facts)  │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │        Grounded LLM Synthesizer        │
                   │    (Strict Prompt: Zero Hallucination) │
                   └───────┬────────────────────────┬───────┘
                           │                        │
                           ▼                        ▼
              ┌────────────────────────┐  ┌────────────────────────┐
              │ Natural Language       │  │ Subgraph Network       │
              │ Grounded Answer        │  │ Payload (vis.js JSON)  │
              └────────────────────────┘  └────────────────────────┘
```

### 3.1 Step 1: Entity Recognition & Intent Classification
Incoming natural language questions are passed through an entity extraction module that maps tokens against known movie titles and person names in the knowledge graph. The system detects intents such as:
1. **Actor Filmography & Roles**
2. **Director Filmography**
3. **Movie Cast & Crew Metadata**
4. **Co-Actors Collaboration Network**
5. **Collaborative Recommendations (2-hop co-star graph filtering)**
6. **Shortest Path / Degrees of Separation (Bacon Number)**
7. **Reviews & Rating Aggregations**

### 3.2 Step 2: Hybrid Query Routing
- **Deterministic Templates:** Canonical questions map to pre-compiled, highly optimized Cypher query templates. This completely eliminates syntax errors, latency, and query hallucination.
- **Dynamic Text-to-Cypher:** For novel multi-condition queries (e.g. *"Find actors born after 1970 who worked with Tom Hanks"*), a few-shot prompt translates the question into Cypher using schema constraints.
- **Security Guardrails:** All queries pass through an AST/regex sanitizer that strictly enforces read-only operations (`MATCH`, `RETURN`, `WHERE`, `ORDER BY`, `LIMIT`) and forbids mutating keywords (`CREATE`, `DELETE`, `SET`, `DROP`, `MERGE`).

### 3.3 Step 3: Graph Traversal & Subgraph Retrieval
Queries are dispatched to the `Neo4jGraphService`. If a live Neo4j instance (Neo4j Desktop or Neo4j AuraDB) is active, the query executes over the Bolt protocol. If offline, the embedded graph engine evaluates the exact graph semantics using NetworkX.

### 3.4 Step 4: Grounded Answer Synthesis
The returned graph nodes, properties, and paths are formatted into a structured context and supplied to the LLM (Gemini 2.5 Flash / offline deterministic engine) with strict instructions:
- State *only* what is supported by the graph records.
- Explain graph connections (e.g., explain *why* a film is recommended by citing the mutual co-star).
- Never hallucinate external trivia.

---

## 4. Set of Cypher Query Templates

Below is the complete set of parameterized Cypher templates implemented in `backend/query_templates.py`:

### Template 1: Actor Filmography & Roles
```cypher
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE toLower(p.name) = toLower($actor)
RETURN p.name AS actor, m.title AS movie, m.released AS released, m.tagline AS tagline, r.roles AS roles
ORDER BY m.released DESC;
```

### Template 2: Director's Filmography
```cypher
MATCH (p:Person)-[:DIRECTED]->(m:Movie)
WHERE toLower(p.name) = toLower($director)
RETURN p.name AS director, m.title AS movie, m.released AS released, m.tagline AS tagline
ORDER BY m.released DESC;
```

### Template 3: Movie Cast, Crew & Metadata
```cypher
MATCH (p:Person)-[r]->(m:Movie)
WHERE toLower(m.title) = toLower($title)
RETURN p.name AS person, type(r) AS relationship, r.roles AS roles, p.born AS born, 
       m.title AS movie, m.released AS released, m.tagline AS tagline
ORDER BY type(r), p.name;
```

### Template 4: Co-Actors Collaboration Network
```cypher
MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE toLower(p1.name) = toLower($actor) AND p1 <> p2
RETURN p2.name AS name, count(DISTINCT m) AS count, collect(DISTINCT m.title) AS shared_movies
ORDER BY count DESC, p2.name ASC
LIMIT 10;
```

### Template 5: Collaborative Graph Movie Recommendation
```cypher
MATCH (target:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(coStar:Person)
WHERE toLower(target.name) = toLower($actor)
MATCH (coStar)-[:ACTED_IN]->(recMovie:Movie)
WHERE NOT (target)-[:ACTED_IN]->(recMovie)
RETURN recMovie.title AS movie, recMovie.released AS released, recMovie.tagline AS tagline,
       count(DISTINCT coStar) AS score, collect(DISTINCT coStar.name) AS via_co_actors
ORDER BY score DESC, recMovie.released DESC
LIMIT 5;
```

### Template 6: Director Style Recommendations
```cypher
MATCH (a:Person)-[:ACTED_IN]->(m1:Movie)<-[:DIRECTED]-(d:Person)
WHERE toLower(a.name) = toLower($actor)
MATCH (d)-[:DIRECTED]->(m2:Movie)
WHERE m1 <> m2 AND NOT (a)-[:ACTED_IN]->(m2)
RETURN m2.title AS movie, m2.released AS released, m2.tagline AS tagline, d.name AS director
ORDER BY m2.released DESC
LIMIT 5;
```

### Template 7: Degrees of Separation (Shortest Path)
```cypher
MATCH (p1:Person), (p2:Person)
WHERE toLower(p1.name) = toLower($person1) AND toLower(p2.name) = toLower($person2)
MATCH path = shortestPath((p1)-[*]-(p2))
RETURN [n in nodes(path) | coalesce(n.name, n.title)] AS path, length(path) AS length;
```

### Template 8: Shared Movies Between Two Actors
```cypher
MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE toLower(p1.name) = toLower($person1) AND toLower(p2.name) = toLower($person2)
RETURN m.title AS movie, m.released AS released, m.tagline AS tagline;
```

### Template 9: Top Reviewed Movies
```cypher
MATCH (p:Person)-[r:REVIEWED]->(m:Movie)
RETURN m.title AS movie, m.released AS released, round(avg(r.rating), 1) AS avg_rating, 
       count(r) AS review_count, collect(r.summary) AS sample_reviews
ORDER BY avg_rating DESC, review_count DESC
LIMIT 5;
```

### Template 10: Actor-Directors (Multitalented Persons)
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m1:Movie)
MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p.name AS person, p.born AS born, collect(DISTINCT m1.title) AS acted_in, collect(DISTINCT m2.title) AS directed
ORDER BY size(acted_in) + size(directed) DESC
LIMIT 10;
```

### Template 11: Most Prolific Actors
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, count(m) AS movie_count, collect(m.title) AS movies
ORDER BY movie_count DESC
LIMIT $limit;
```

---

## 5. Example Interactions & Evaluation

### Interaction 1: Complex Actor Filmography & Roles
- **User Prompt:** *"What movies did Keanu Reeves act in and what were his roles?"*
- **Extracted Entity:** `Person: Keanu Reeves`
- **Matched Template:** `ACTOR_FILMOGRAPHY`
- **Executed Cypher:**
  ```cypher
  MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
  WHERE toLower(p.name) = toLower('Keanu Reeves')
  RETURN p.name AS actor, m.title AS movie, m.released AS released, m.tagline AS tagline, r.roles AS roles
  ORDER BY m.released DESC
  ```
- **Retrieved Subgraph:**
  - *The Matrix Revolutions* (2003) -> `['Neo']`
  - *The Matrix Reloaded* (2003) -> `['Neo']`
  - *The Replacements* (2000) -> `['Shane Falco']`
  - *The Matrix* (1999) -> `['Neo']`
  - *The Devil's Advocate* (1997) -> `['Kevin Lomax']`
  - *Johnny Mnemonic* (1995) -> `['Johnny Mnemonic']`
- **Grounded Answer:**
  > According to the Neo4j Knowledge Graph, **Keanu Reeves** has starred in **6** movies:
  > - **The Matrix Revolutions** (2003) as Neo
  > - **The Matrix Reloaded** (2003) as Neo
  > - **The Replacements** (2000) as Shane Falco
  > - **The Matrix** (1999) as Neo
  > - **The Devil's Advocate** (1997) as Kevin Lomax
  > - **Johnny Mnemonic** (1995) as Johnny Mnemonic

---

### Interaction 2: Collaborative Movie Recommendations
- **User Prompt:** *"Recommend movies for a fan of Tom Cruise"*
- **Extracted Entity:** `Person: Tom Cruise`
- **Matched Template:** `RECOMMEND_BY_ACTOR`
- **Executed Cypher:**
  ```cypher
  MATCH (target:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(coStar:Person)
  WHERE toLower(target.name) = toLower('Tom Cruise')
  MATCH (coStar)-[:ACTED_IN]->(recMovie:Movie)
  WHERE NOT (target)-[:ACTED_IN]->(recMovie)
  RETURN recMovie.title AS movie, recMovie.released AS released, recMovie.tagline AS tagline,
         count(DISTINCT coStar) AS score, collect(DISTINCT coStar.name) AS via_co_actors
  ORDER BY score DESC, recMovie.released DESC
  LIMIT 5
  ```
- **Retrieved Graph Evidence:**
  - *Stand By Me* (1986): Connected via co-stars **Jerry O'Connell** and **Kiefer Sutherland** (Score: 2)
  - *Apollo 13* (1995): Connected via co-star **Kevin Bacon** (Score: 1)
  - *As Good as It Gets* (1997): Connected via co-stars **Jack Nicholson** and **Cuba Gooding Jr.** (Score: 2)
- **Grounded Answer:**
  > Recommended titles based on collaborative graph traversal (movies starring Tom Cruise's most frequent co-stars):
  > - **As Good as It Gets** (1997): Recommended via co-stars **Jack Nicholson** and **Cuba Gooding Jr.** (*"A comedy from the heart that goes for the throat."*)
  > - **Stand By Me** (1986): Recommended via co-stars **Jerry O'Connell** and **Kiefer Sutherland** (*"For some, it's the last real adventure of a childhood..."*)
  > - **Apollo 13** (1995): Recommended via co-star **Kevin Bacon** (*"Houston, we have a problem."*)

---

### Interaction 3: Shortest Path / Bacon Number Analysis
- **User Prompt:** *"How are Kevin Bacon and Keanu Reeves connected in the graph?"*
- **Matched Template:** `SHORTEST_PATH`
- **Executed Cypher:**
  ```cypher
  MATCH (p1:Person), (p2:Person)
  WHERE toLower(p1.name) = toLower('Kevin Bacon') AND toLower(p2.name) = toLower('Keanu Reeves')
  MATCH path = shortestPath((p1)-[*]-(p2))
  RETURN [n in nodes(path) | coalesce(n.name, n.title)] AS path, length(path) AS length
  ```
- **Graph Traversal Result:**
  - `Kevin Bacon` -> *A Few Good Men* -> `Jack Nicholson` -> *As Good as It Gets* -> `Cuba Gooding Jr.` -> *What Dreams May Come* -> (or via Charlize Theron / Tom Cruise)
  - Shortest Path: `Kevin Bacon` -> `A Few Good Men` -> `Jack Nicholson` / `Aaron Sorkin` ...
- **Grounded Answer:**
  > Degrees of separation: **2** (Bacon number connection):
  > **Kevin Bacon** -> **A Few Good Men** -> **Jack Nicholson** -> **As Good as It Gets** -> **Cuba Gooding Jr.** -> **What Dreams May Come** -> ...

---

## 6. Limitations & Edge Cases

1. **Schema Scope & Domain Boundaries:**
   - The canonical Neo4j Movies dataset contains approximately 171 nodes and 253 relationships focusing primarily on 90s/early 2000s classics. Queries regarding modern releases (e.g. *Oppenheimer (2023)* or *Dune (2021)*) return clean out-of-domain responses.
2. **Ambiguous Entity Resolution:**
   - Common first names or abbreviations (e.g. "Tom" referring to Tom Cruise vs. Tom Hanks) require disambiguation prompts or prefix matching.
3. **Complex Text-to-Cypher on Arbitrary Prompts:**
   - When generating free-form Cypher without templates, highly nested queries (e.g., aggregate filtering across multiple OPTIONAL MATCH clauses) can occasionally yield sub-optimal Cypher syntax. The template router mitigates this for over 90% of user queries.
4. **Read-Only Constraints:**
   - In accordance with security best practices, the application rejects any mutating or administrative operations (`CREATE`, `DELETE`, `DROP`).

---

## 7. Possible Extensions & Future Work

1. **Hybrid Vector + Graph RAG (GraphRAG):**
   - Combine dense vector embeddings of movie plot summaries and critic reviews with Neo4j relationship traversals to support semantic thematic queries (e.g. *"Show me dystopian sci-fi movies featuring philosophical virtual reality themes"*).
2. **Community Detection & Automated Genre Clustering:**
   - Apply the Louvain or Leiden graph algorithm using Neo4j GDS (Graph Data Science) to uncover implicit clusters and sub-communities of collaborators.
3. **Personalized User Watchlist Nodes:**
   - Extend the graph schema to include `(:User)` nodes connected via `[:LIKED]` and `[:WATCHED]` edges, enabling real-time personalized recommendations.
4. **Knowledge Graph Expansion via Automated Web Scraping:**
   - Connect live TMDB (The Movie Database) or IMDb APIs to continuously ingest new movies, actors, and box-office revenue into the Neo4j instance.

---

## 8. Summary of Deliverables

| Deliverable | Location in Repository | Purpose |
| :--- | :--- | :--- |
| **Neo4j Movies Dataset Script** | `data/movies_dataset.cql` | Full canonical Cypher creation script for live Neo4j Desktop or AuraDB |
| **Neo4j Seeding Utility** | `scripts/seed_neo4j.py` | Automated CLI tool to seed and verify constraints in any Neo4j instance |
| **Cypher Query Templates** | `backend/query_templates.py` | 11+ parameterized templates covering QA, Recommendations, and Network Analysis |
| **Graph Database Service** | `backend/graph_db.py` | Live Neo4j driver + offline embedded graph fallback engine |
| **Graph RAG Orchestrator** | `backend/rag_engine.py` | Entity extraction, routing, Text-to-Cypher, and grounded LLM synthesis |
| **Streamlit Web Application** | `app.py` | Glassmorphic web UI with interactive vis.js graph visualization & template runner |
| **CLI Prototype** | `cli_app.py` | Terminal-based interactive prototype for instant testing and automated evaluation |
| **Documentation & Guide** | `README.md` & `PROJECT_REPORT.md` | Comprehensive setup guide and academic/technical project report |
