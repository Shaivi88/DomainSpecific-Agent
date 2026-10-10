from crewai import Crew, Task, Process

KEYWORD_MAP = {
    "marketing": [
        "price", "pricing", "competitor", "brand", "positioning",
        "market", "consumer", "retail", "channel", "launch strategy",
        "advertising", "promotion", "campaign",
    ],
    "regulatory": [
        "fssai", "regulation", "regulatory", "label", "labeling",
        "certificate", "certification", "compliance", "approval",
        "license", "import", "export", "standard", "permit",
    ],
    "supply_chain": [
        "factory", "supply", "distribution", "logistics", "manufacturing",
        "warehouse", "inventory", "delivery", "cold chain", "vendor",
        "supplier", "production", "capacity",
    ],
    "finance": [
        "profit", "margin", "p&l", "cost", "revenue", "break-even",
        "breakeven", "capital", "investment", "budget", "financial",
        "roi", "expense", "funding",
    ],
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
