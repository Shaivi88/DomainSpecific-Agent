from crewai import Crew, Process
from agents import (
    marketing_agent,
    regulatory_agent,
    supply_chain_agent,
    finance_agent,
    synthesizer_agent,
)
from tasks import build_tasks


def build_crew(product_name: str, category: str, target_market: str, usp: str) -> Crew:
    task_list = build_tasks(product_name, category, target_market, usp)
    return Crew(
        agents=[
            marketing_agent,
            regulatory_agent,
            supply_chain_agent,
            finance_agent,
            synthesizer_agent,
        ],
        tasks=task_list,
        process=Process.sequential,
        verbose=True,
    )


def run_report(product_name: str, category: str, target_market: str, usp: str) -> str:
    crew = build_crew(product_name, category, target_market, usp)
    result = crew.kickoff()
    return str(result)
