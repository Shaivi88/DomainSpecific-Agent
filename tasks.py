from crewai import Task
from agents import (
    marketing_agent,
    regulatory_agent,
    supply_chain_agent,
    finance_agent,
    synthesizer_agent,
)


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
