# FMCG Product Launch Planner — Multi-Agent System Design

**Date:** 2026-10-10  
**Stack:** Python · CrewAI · Groq (`openai/gpt-oss-120b`) · SerpAPI · Streamlit

---

## Overview

A multi-agent AI system that helps FMCG companies plan product launches. Users can either submit a product brief to get a full launch readiness report (report mode), or ask free-text questions to a relevant domain expert agent (chat mode). Both modes run through the same Streamlit interface.

---

## Architecture

```
User (Streamlit UI)
      │
      ├── Report Mode: product brief form
      │     → CrewAI Crew (sequential process)
      │         Marketing Agent → Regulatory Agent → Supply Chain Agent → Finance Agent → Synthesizer Agent
      │         → Full launch readiness report rendered in chat panel
      │
      └── Chat Mode: free-text input
            → Router logic (keyword/LLM-based) picks one specialist agent
            → That agent responds directly
            → Response rendered in chat panel
```

---

## File Structure

```
shaiviProject/
├── app.py          # Streamlit UI — chat panel, sidebar form, session state
├── agents.py       # All 5 CrewAI Agent definitions + shared LLM config
├── tasks.py        # CrewAI Task definitions for report mode
├── crew.py         # Crew assembly and .kickoff() for report mode
├── router.py       # Chat mode routing — picks agent from user query
├── tools.py        # SerpAPI tool instance
├── .env            # GROQ_API_KEY, SERPAPI_API_KEY
└── requirements.txt
```

---

## LLM Configuration

```python
import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="openai/gpt-oss-120b",
    custom_openai=True,
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
```

`custom_openai=True` bypasses LiteLLM and uses CrewAI's native OpenAI SDK pointed at Groq's endpoint. The model name is passed through untouched.

---

## Agents

All agents share the same `llm` instance defined above.

### 1. Marketing Agent
- **Role:** FMCG Marketing & Pricing Strategist  
- **Goal:** Analyze competitor landscape, suggest pricing strategy and brand positioning  
- **Tools:** SerpAPI  
- **Searches for:** competitor products, price points, market trends for the product category

### 2. Regulatory Agent
- **Role:** FMCG Regulatory & Compliance Specialist  
- **Goal:** Identify applicable regulations, labeling requirements, certifications needed  
- **Tools:** SerpAPI  
- **Searches for:** FSSAI regulations, import/export rules, packaging standards for product type

### 3. Supply Chain Agent
- **Role:** Supply Chain & Manufacturing Advisor  
- **Goal:** Assess distribution feasibility, manufacturing readiness, logistics considerations  
- **Tools:** None (domain knowledge)  

### 4. Finance Agent
- **Role:** FMCG Finance & P&L Analyst  
- **Goal:** Estimate rough P&L, margins, break-even, and capital requirements  
- **Tools:** None (domain knowledge)  

### 5. Synthesizer Agent
- **Role:** Launch Readiness Coordinator  
- **Goal:** Consolidate all domain analyses into a structured launch readiness report  
- **Tools:** None  
- **Output:** Markdown report with sections per domain + overall go/no-go recommendation

---

## Tasks (Report Mode)

Each task maps 1:1 to an agent. Tasks run sequentially. Each task receives the product brief as context, and later tasks also receive prior task outputs via `context=[]`.

1. `market_analysis_task` → Marketing Agent
2. `regulatory_check_task` → Regulatory Agent
3. `supply_chain_task` → Supply Chain Agent
4. `finance_task` → Finance Agent
5. `synthesis_task` → Synthesizer Agent (context = all 4 prior tasks)

---

## Chat Mode Routing

A simple keyword + fallback LLM router in `router.py`:

- Keywords like "price", "competitor", "brand" → Marketing Agent
- Keywords like "regulation", "FSSAI", "label", "certificate" → Regulatory Agent
- Keywords like "factory", "supply", "distribution", "logistics" → Supply Chain Agent
- Keywords like "profit", "margin", "cost", "P&L", "break-even" → Finance Agent
- No match → Marketing Agent as default

The matched agent is run as a single-task mini-crew with the user's question as the task description.

---

## Streamlit UI

- **Sidebar:** Product brief form (product name, category, target market, USP) + "Generate Report" button
- **Main panel:** Chat history (st.session_state messages list), streamed agent responses
- **Input bar:** Free-text chat input at bottom for chat mode
- **Report display:** Full synthesized report rendered as markdown in the chat panel

---

## Environment Variables

```
GROQ_API_KEY=your_groq_key
SERPAPI_API_KEY=your_serpapi_key
```

---

## Dependencies

```
crewai
crewai-tools
streamlit
python-dotenv
```

---

## Constraints & Scope

- No database or persistence — session state only
- No authentication
- No streaming token-by-token — full agent response displayed when complete
- SerpAPI free tier (100 searches/month) — only Marketing and Regulatory agents search
- Report mode runs agents sequentially (not parallel) to keep implementation simple
