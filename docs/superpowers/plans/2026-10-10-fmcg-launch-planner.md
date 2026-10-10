# FMCG Launch Planner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit multi-agent app where CrewAI agents (powered by Groq) help plan FMCG product launches via a report mode and a chat mode.

**Architecture:** Five CrewAI agents (Marketing, Regulatory, Supply Chain, Finance, Synthesizer) share one Groq LLM instance. In report mode they run sequentially as a Crew producing a markdown launch readiness report. In chat mode a keyword router picks the best single agent to answer the user's question.

**Tech Stack:** Python 3.11+, CrewAI, crewai-tools (SerpApiGoogleSearchTool), Streamlit, python-dotenv, Groq (`openai/gpt-oss-120b` via custom OpenAI endpoint)

**Spec:** `docs/superpowers/specs/2026-10-10-fmcg-launch-planner-design.md`

---

## Global Constraints

- LLM must use `custom_openai=True`, `base_url="https://api.groq.com/openai/v1"`, `model="openai/gpt-oss-120b"` — never use `groq/` prefix (LiteLLM is deprecated in CrewAI)
- All API keys loaded via `python-dotenv` from `.env` — never hardcoded
- `.env` keys: `GROQ_API_KEY`, `SERPAPI_API_KEY`
- Only Marketing Agent and Regulatory Agent use SerpAPI — other agents use domain knowledge only
- Crew process: `Process.sequential` — no parallel execution
- No database, no auth, no token streaming — session state + full response display only
- Dependencies: `crewai`, `crewai-tools`, `streamlit`, `python-dotenv`

## Review Focus

- **Empty product brief submitted** — report mode should show a validation warning, not crash the crew
- **Unrecognised chat query** — router falls back to Marketing Agent, user still gets a response
- **SerpAPI quota exhausted** — agents with search tools should degrade gracefully with an error message rather than traceback
- **Very long synthesizer output** — Streamlit must render full markdown without truncation
- **Session state cleared on rerun** — chat history must persist across Streamlit reruns via `st.session_state`

---

### Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: installable environment for all subsequent tasks

- [ ] **Step 1: Create `requirements.txt`**

```
crewai
crewai-tools
streamlit
python-dotenv
pytest
```

- [ ] **Step 2: Create `.env.example`**

```
GROQ_API_KEY=your_groq_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

- [ ] **Step 3: Create `tests/__init__.py`** (empty file — marks tests as a package)

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt .env.example tests/__init__.py
git commit -m "chore: project scaffold and dependencies"
```

---

### Task 2: LLM Config + SerpAPI Tool

**Files:**
- Create: `tools.py`
- Create: `llm_config.py`
- Create: `tests/test_llm_config.py`

**Interfaces:**
- Produces:
  - `llm_config.llm` — a `crewai.LLM` instance, importable by agents.py
  - `tools.search_tool` — a `SerpApiGoogleSearchTool` instance, importable by agents.py

- [ ] **Step 1: Write failing test for LLM config**

```python
# tests/test_llm_config.py
import os
import pytest

def test_llm_config_has_correct_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from llm_config import llm
    assert llm.model == "openai/gpt-oss-120b"

def test_llm_config_custom_openai(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from llm_config import llm
    assert llm.base_url == "https://api.groq.com/openai/v1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_llm_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'llm_config'`

- [ ] **Step 3: Create `llm_config.py`**

```python
import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="openai/gpt-oss-120b",
    custom_openai=True,
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)
```

- [ ] **Step 4: Create `tools.py`**

```python
from crewai_tools import SerpApiGoogleSearchTool

search_tool = SerpApiGoogleSearchTool()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_llm_config.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add llm_config.py tools.py tests/test_llm_config.py
git commit -m "feat: Groq LLM config and SerpAPI tool"
```

---

### Task 3: Agent Definitions

**Files:**
- Create: `agents.py`
- Create: `tests/test_agents.py`

**Interfaces:**
- Consumes: `llm_config.llm`, `tools.search_tool`
- Produces:
  - `agents.marketing_agent` — `crewai.Agent`
  - `agents.regulatory_agent` — `crewai.Agent`
  - `agents.supply_chain_agent` — `crewai.Agent`
  - `agents.finance_agent` — `crewai.Agent`
  - `agents.synthesizer_agent` — `crewai.Agent`

- [ ] **Step 1: Write failing tests for agents**

```python
# tests/test_agents.py
import os
import pytest

def setup_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

def test_marketing_agent_has_search_tool(monkeypatch):
    setup_env(monkeypatch)
    from agents import marketing_agent
    tool_names = [type(t).__name__ for t in marketing_agent.tools]
    assert "SerpApiGoogleSearchTool" in tool_names

def test_regulatory_agent_has_search_tool(monkeypatch):
    setup_env(monkeypatch)
    from agents import regulatory_agent
    tool_names = [type(t).__name__ for t in regulatory_agent.tools]
    assert "SerpApiGoogleSearchTool" in tool_names

def test_supply_chain_agent_has_no_tools(monkeypatch):
    setup_env(monkeypatch)
    from agents import supply_chain_agent
    assert supply_chain_agent.tools == [] or supply_chain_agent.tools is None

def test_finance_agent_has_no_tools(monkeypatch):
    setup_env(monkeypatch)
    from agents import finance_agent
    assert finance_agent.tools == [] or finance_agent.tools is None

def test_synthesizer_agent_has_no_tools(monkeypatch):
    setup_env(monkeypatch)
    from agents import synthesizer_agent
    assert synthesizer_agent.tools == [] or synthesizer_agent.tools is None

def test_all_agents_use_same_llm_model(monkeypatch):
    setup_env(monkeypatch)
    from agents import marketing_agent, regulatory_agent, supply_chain_agent, finance_agent, synthesizer_agent
    for agent in [marketing_agent, regulatory_agent, supply_chain_agent, finance_agent, synthesizer_agent]:
        assert agent.llm.model == "openai/gpt-oss-120b"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agents.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents'`

- [ ] **Step 3: Create `agents.py`**

```python
from crewai import Agent
from llm_config import llm
from tools import search_tool

marketing_agent = Agent(
    role="FMCG Marketing & Pricing Strategist",
    goal="Analyze the competitor landscape and recommend pricing strategy and brand positioning for the product.",
    backstory=(
        "You are a seasoned FMCG marketing strategist with 15 years of experience "
        "launching consumer goods across India and Southeast Asia. You understand "
        "retail dynamics, consumer psychology, and how to position products against "
        "established competitors."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

regulatory_agent = Agent(
    role="FMCG Regulatory & Compliance Specialist",
    goal="Identify all regulatory requirements, certifications, and labeling standards the product must meet before launch.",
    backstory=(
        "You are a regulatory expert specializing in FMCG compliance in India. "
        "You know FSSAI requirements, packaging standards, import/export rules, "
        "and how to navigate certification timelines efficiently."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

supply_chain_agent = Agent(
    role="Supply Chain & Manufacturing Advisor",
    goal="Assess manufacturing readiness, distribution channel feasibility, and logistics considerations for the product launch.",
    backstory=(
        "You are a supply chain specialist with deep expertise in FMCG manufacturing "
        "and distribution across modern trade, general trade, and e-commerce channels. "
        "You help brands avoid launch delays caused by operational bottlenecks."
    ),
    tools=[],
    llm=llm,
    verbose=True,
)

finance_agent = Agent(
    role="FMCG Finance & P&L Analyst",
    goal="Estimate a rough P&L, gross margins, break-even point, and capital requirements for the product launch.",
    backstory=(
        "You are an FMCG financial analyst who has modelled P&Ls for dozens of "
        "consumer product launches. You provide realistic financial projections "
        "based on category benchmarks and cost structures."
    ),
    tools=[],
    llm=llm,
    verbose=True,
)

synthesizer_agent = Agent(
    role="Launch Readiness Coordinator",
    goal="Synthesize all domain analyses into a structured, actionable launch readiness report with an overall go/no-go recommendation.",
    backstory=(
        "You are a senior product launch manager who has coordinated cross-functional "
        "launches for major FMCG brands. You distil complex inputs from marketing, "
        "regulatory, supply chain, and finance into clear executive summaries."
    ),
    tools=[],
    llm=llm,
    verbose=True,
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agents.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add agents.py tests/test_agents.py
git commit -m "feat: define all 5 CrewAI agents"
```

---

### Task 4: Task Definitions (Report Mode)

**Files:**
- Create: `tasks.py`
- Create: `tests/test_tasks.py`

**Interfaces:**
- Consumes: all agents from `agents.py`
- Produces: `tasks.build_tasks(product_name, category, target_market, usp)` → `list[crewai.Task]` of length 5

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tasks.py
import os
import pytest

def setup_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

def test_build_tasks_returns_five_tasks(monkeypatch):
    setup_env(monkeypatch)
    from tasks import build_tasks
    result = build_tasks("CoolDrink", "Energy Drink", "Urban youth 18-30", "Zero sugar, natural caffeine")
    assert len(result) == 5

def test_synthesis_task_has_context_from_all_others(monkeypatch):
    setup_env(monkeypatch)
    from tasks import build_tasks
    result = build_tasks("CoolDrink", "Energy Drink", "Urban youth 18-30", "Zero sugar, natural caffeine")
    synthesis_task = result[4]
    assert len(synthesis_task.context) == 4

def test_product_name_in_task_description(monkeypatch):
    setup_env(monkeypatch)
    from tasks import build_tasks
    result = build_tasks("CoolDrink", "Energy Drink", "Urban youth 18-30", "Zero sugar")
    for task in result:
        assert "CoolDrink" in task.description
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tasks.py -v
```

Expected: `ModuleNotFoundError: No module named 'tasks'`

- [ ] **Step 3: Create `tasks.py`**

```python
from crewai import Task
from agents import marketing_agent, regulatory_agent, supply_chain_agent, finance_agent, synthesizer_agent


def build_tasks(product_name: str, category: str, target_market: str, usp: str) -> list:
    brief = (
        f"Product: {product_name}\n"
        f"Category: {category}\n"
        f"Target Market: {target_market}\n"
        f"USP: {usp}"
    )

    market_task = Task(
        description=(
            f"Analyze the competitive landscape for {product_name}.\n{brief}\n\n"
            "Search for: top competitors in this category, their price points, "
            "market share, and positioning. Recommend a pricing strategy and "
            "brand positioning angle for this product."
        ),
        expected_output=(
            "A market analysis covering: top 3-5 competitors with price points, "
            "recommended retail price range, suggested brand positioning statement, "
            "and key differentiation opportunities."
        ),
        agent=marketing_agent,
    )

    regulatory_task = Task(
        description=(
            f"Identify regulatory and compliance requirements for {product_name}.\n{brief}\n\n"
            "Search for: FSSAI regulations for this product category, mandatory "
            "labeling requirements, certifications needed, and typical approval timelines."
        ),
        expected_output=(
            "A compliance checklist covering: required FSSAI licenses/registrations, "
            "mandatory label elements, certifications needed, and estimated time "
            "to obtain all approvals."
        ),
        agent=regulatory_agent,
    )

    supply_chain_task = Task(
        description=(
            f"Assess supply chain and manufacturing readiness for launching {product_name}.\n{brief}\n\n"
            "Cover: manufacturing considerations for this product category, "
            "recommended distribution channels (modern trade, general trade, e-commerce), "
            "cold chain requirements if any, and key logistics risks."
        ),
        expected_output=(
            "A supply chain assessment covering: manufacturing readiness checklist, "
            "recommended distribution channel mix, key logistics risks, and "
            "suggested warehouse/distribution strategy."
        ),
        agent=supply_chain_agent,
    )

    finance_task = Task(
        description=(
            f"Estimate the P&L and financial requirements for launching {product_name}.\n{brief}\n\n"
            "Provide: estimated cost of goods sold (COGS) as % of MRP for this category, "
            "gross margin benchmarks, break-even volume estimate, and capital required "
            "for a 6-month launch runway."
        ),
        expected_output=(
            "A financial summary covering: estimated COGS %, gross margin %, "
            "break-even monthly volume, estimated launch capital required (INR), "
            "and key financial risks."
        ),
        agent=finance_agent,
    )

    synthesis_task = Task(
        description=(
            f"Create a comprehensive launch readiness report for {product_name}.\n{brief}\n\n"
            "Using the analyses from marketing, regulatory, supply chain, and finance teams, "
            "write a structured report with an overall GO / NO-GO / CONDITIONAL GO recommendation."
        ),
        expected_output=(
            "A structured markdown report with sections:\n"
            "1. Executive Summary & Recommendation (GO / NO-GO / CONDITIONAL GO)\n"
            "2. Market & Competitive Position\n"
            "3. Regulatory Compliance Status\n"
            "4. Supply Chain Readiness\n"
            "5. Financial Viability\n"
            "6. Top 3 Risks & Mitigation\n"
            "7. Suggested Launch Timeline (12-week view)"
        ),
        agent=synthesizer_agent,
        context=[market_task, regulatory_task, supply_chain_task, finance_task],
    )

    return [market_task, regulatory_task, supply_chain_task, finance_task, synthesis_task]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tasks.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add tasks.py tests/test_tasks.py
git commit -m "feat: define CrewAI tasks for report mode"
```

---

### Task 5: Crew Assembly

**Files:**
- Create: `crew.py`
- Create: `tests/test_crew.py`

**Interfaces:**
- Consumes: `tasks.build_tasks()`
- Produces: `crew.run_report(product_name, category, target_market, usp)` → `str` (markdown report)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_crew.py
import os
import pytest

def setup_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

def test_run_report_signature_exists(monkeypatch):
    setup_env(monkeypatch)
    from crew import run_report
    import inspect
    sig = inspect.signature(run_report)
    params = list(sig.parameters.keys())
    assert params == ["product_name", "category", "target_market", "usp"]

def test_crew_uses_sequential_process(monkeypatch):
    setup_env(monkeypatch)
    from crew import build_crew
    from crewai import Process
    crew = build_crew("TestProduct", "Snack", "Adults", "Healthy")
    assert crew.process == Process.sequential
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_crew.py -v
```

Expected: `ModuleNotFoundError: No module named 'crew'`

- [ ] **Step 3: Create `crew.py`**

```python
from crewai import Crew, Process
from agents import marketing_agent, regulatory_agent, supply_chain_agent, finance_agent, synthesizer_agent
from tasks import build_tasks


def build_crew(product_name: str, category: str, target_market: str, usp: str) -> Crew:
    task_list = build_tasks(product_name, category, target_market, usp)
    return Crew(
        agents=[marketing_agent, regulatory_agent, supply_chain_agent, finance_agent, synthesizer_agent],
        tasks=task_list,
        process=Process.sequential,
        verbose=True,
    )


def run_report(product_name: str, category: str, target_market: str, usp: str) -> str:
    crew = build_crew(product_name, category, target_market, usp)
    result = crew.kickoff()
    return str(result)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_crew.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add crew.py tests/test_crew.py
git commit -m "feat: crew assembly and report runner"
```

---

### Task 6: Chat Mode Router

**Files:**
- Create: `router.py`
- Create: `tests/test_router.py`

**Interfaces:**
- Consumes: all agents from `agents.py`
- Produces: `router.route_and_respond(query: str) -> str` — runs the matched agent as a mini-crew, returns response string

- [ ] **Step 1: Write failing tests**

```python
# tests/test_router.py
import os
import pytest

def setup_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

def test_marketing_keywords_route_to_marketing(monkeypatch):
    setup_env(monkeypatch)
    from router import pick_agent
    from agents import marketing_agent
    assert pick_agent("What price should I set for my energy drink?") == marketing_agent

def test_regulatory_keywords_route_to_regulatory(monkeypatch):
    setup_env(monkeypatch)
    from router import pick_agent
    from agents import regulatory_agent
    assert pick_agent("What FSSAI certification do I need?") == regulatory_agent

def test_supply_chain_keywords_route_to_supply_chain(monkeypatch):
    setup_env(monkeypatch)
    from router import pick_agent
    from agents import supply_chain_agent
    assert pick_agent("Can our factory handle the distribution?") == supply_chain_agent

def test_finance_keywords_route_to_finance(monkeypatch):
    setup_env(monkeypatch)
    from router import pick_agent
    from agents import finance_agent
    assert pick_agent("What is the expected profit margin?") == finance_agent

def test_unknown_query_falls_back_to_marketing(monkeypatch):
    setup_env(monkeypatch)
    from router import pick_agent
    from agents import marketing_agent
    assert pick_agent("xyzzy random words with no domain keywords") == marketing_agent

def test_route_and_respond_returns_string(monkeypatch):
    setup_env(monkeypatch)
    from router import route_and_respond
    assert callable(route_and_respond)
    import inspect
    sig = inspect.signature(route_and_respond)
    assert "query" in sig.parameters
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'router'`

- [ ] **Step 3: Create `router.py`**

```python
from crewai import Crew, Task, Process, Agent

KEYWORD_MAP = {
    "marketing": ["price", "pricing", "competitor", "brand", "positioning", "market", "consumer", "retail", "channel", "launch strategy"],
    "regulatory": ["fssai", "regulation", "regulatory", "label", "labeling", "certificate", "certification", "compliance", "approval", "license", "import", "export", "standard"],
    "supply_chain": ["factory", "supply", "distribution", "logistics", "manufacturing", "warehouse", "inventory", "delivery", "cold chain", "vendor"],
    "finance": ["profit", "margin", "p&l", "cost", "revenue", "break-even", "breakeven", "capital", "investment", "budget", "financial", "roi"],
}


def pick_agent(query: str):
    from agents import marketing_agent, regulatory_agent, supply_chain_agent, finance_agent
    agent_map = {
        "marketing": marketing_agent,
        "regulatory": regulatory_agent,
        "supply_chain": supply_chain_agent,
        "finance": finance_agent,
    }
    query_lower = query.lower()
    scores = {domain: 0 for domain in KEYWORD_MAP}
    for domain, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in query_lower:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return marketing_agent  # default fallback
    return agent_map[best]


def route_and_respond(query: str) -> str:
    agent = pick_agent(query)
    task = Task(
        description=query,
        expected_output="A clear, expert answer to the question from your domain perspective.",
        agent=agent,
    )
    mini_crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    result = mini_crew.kickoff()
    return str(result)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_router.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add router.py tests/test_router.py
git commit -m "feat: keyword-based chat router with fallback"
```

---

### Task 7: Streamlit UI

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `crew.run_report()`, `router.route_and_respond()`
- Produces: runnable Streamlit app

Note: Streamlit apps are not unit-testable in the standard sense. This task uses manual verification steps instead.

- [ ] **Step 1: Create `app.py`**

```python
import streamlit as st
from crew import run_report
from router import route_and_respond

st.set_page_config(
    page_title="FMCG Launch Planner",
    page_icon="🚀",
    layout="wide",
)

# ── Session state init ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar: product brief form ─────────────────────────────────
with st.sidebar:
    st.title("🚀 FMCG Launch Planner")
    st.markdown("---")
    st.subheader("Generate Launch Report")

    product_name = st.text_input("Product Name", placeholder="e.g. ZestUp Energy Drink")
    category = st.text_input("Category", placeholder="e.g. Energy Drink")
    target_market = st.text_input("Target Market", placeholder="e.g. Urban youth 18-30")
    usp = st.text_input("USP / Key Differentiator", placeholder="e.g. Zero sugar, natural caffeine")

    generate_clicked = st.button("Generate Report", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Or ask any question in the chat →")

# ── Main panel: chat history ────────────────────────────────────
st.title("FMCG Product Launch Planner")
st.markdown("Use the sidebar to generate a full launch report, or ask a question below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Report mode ─────────────────────────────────────────────────
if generate_clicked:
    if not all([product_name, category, target_market, usp]):
        st.warning("Please fill in all fields in the sidebar before generating a report.")
    else:
        user_msg = (
            f"**Generate full launch report for:**\n"
            f"- Product: {product_name}\n"
            f"- Category: {category}\n"
            f"- Target Market: {target_market}\n"
            f"- USP: {usp}"
        )
        st.session_state.messages.append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Running launch analysis across all agents... this may take a minute."):
                try:
                    report = run_report(product_name, category, target_market, usp)
                    st.markdown(report)
                    st.session_state.messages.append({"role": "assistant", "content": report})
                except Exception as e:
                    error_msg = f"⚠️ Error running report: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ── Chat mode ───────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your product launch..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting domain expert..."):
            try:
                response = route_and_respond(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
```

- [ ] **Step 2: Verify the app starts without error**

```bash
streamlit run app.py
```

Expected: Streamlit opens in browser with sidebar showing form fields and empty chat panel.

- [ ] **Step 3: Manual — verify session state persists**

1. Type a question in the chat input
2. Wait for response
3. Type a second question
Expected: both messages and both responses visible in the chat panel.

- [ ] **Step 4: Manual — verify report mode validation**

1. Click "Generate Report" with empty fields
Expected: yellow warning message "Please fill in all fields", no crash.

- [ ] **Step 5: Manual — verify report mode end-to-end (requires real API keys in .env)**

1. Fill in: Product Name="ZestUp", Category="Energy Drink", Target Market="Urban youth 18-30", USP="Zero sugar"
2. Click "Generate Report"
Expected: spinner appears, then a markdown report with 7 sections appears in the chat panel.

- [ ] **Step 6: Manual — verify chat mode end-to-end**

1. Type: "What FSSAI certifications do I need for an energy drink?"
Expected: Regulatory Agent responds with compliance information.

2. Type: "What price should I charge?"
Expected: Marketing Agent responds with pricing guidance.

- [ ] **Step 7: Run full test suite to confirm nothing broken**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit UI with report mode and chat mode"
```
