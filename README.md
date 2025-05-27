# RAG_Types
Different Types of RAG, Source code of different types of RAG with detailed expalination how its implemented

Simple RAG
A straightforward pipeline: retrieve the top K relevant document snippets for a user query, then feed them plus the query into an LLM to generate an informed answer. Fast to implement, balances LLM fluency with up-to-date or domain-specific facts, but may miss deeper context or multi-step reasoning.

┌────────────┐       ┌───────────────┐       ┌────────────┐
│   User     │─query─▶  Retriever   │─docs──▶│  Generator │─answer─▶ User
└────────────┘       └───────────────┘       └────────────┘

Self RAG
The model alternates between “thinking” and “looking up.” It drafts an outline or partial answer, identifies knowledge gaps, retrieves targeted documents, then refines its response. This iterative loop yields richer, more accurate answers by letting the LLM proactively fetch evidence as it writes.

┌─────────┐   ┌────────────┐   ┌───────────────┐   ┌────────────┐
│  User   │▶▶▶│  LLM think ├▶▶▶│  Retriever    ├▶▶▶│  LLM refine│──▶ User
└─────────┘   └────────────┘   └───────────────┘   └────────────┘

Fusion RAG
Combine multiple knowledge sources—e.g., web articles, internal databases, knowledge graphs—by retrieving from each independently. Then either concatenate all results for one LLM pass or generate per-source answers and merge them. Fusion RAG delivers broad coverage plus deep, specialized insight in a single response.

                ┌───────────────┐
Retriever A ───▶│  LLM answer A │
                └───────────────┘           ┌────────────┐
Retriever B ───▶│  LLM answer B │───merge──▶│ Fusion LLM │─▶ User
                └───────────────┘           └────────────┘

Corrective RAG
First, the LLM drafts an answer from its internal knowledge. Next, extract each factual claim and retrieve evidence to support or contradict it. Flag inconsistencies, prompt the LLM with corrections, and generate a final, fact-checked response with inline citations—melding generative fluency with rigorous verification.

flowchart TD
    User["💬 User Query"] --> InitialRetriever
    InitialRetriever --> FirstLLM["🧠 Initial Answer"]
    FirstLLM --> Check["🔎 Self-Evaluation"]
    Check -- Incomplete --> Refine["🔁 Refine Query"]
    Refine --> SecondRetriever
    SecondRetriever --> FinalLLM["✅ Final Answer"]
    Check -- Sufficient --> FinalLLM
