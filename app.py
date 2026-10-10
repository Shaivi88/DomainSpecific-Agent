import streamlit as st
from crew import run_report
from router import route_and_respond

st.set_page_config(
    page_title="FMCG Launch Planner",
    page_icon="🚀",
    layout="wide",
)

# ── Session state init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar: product brief form ──────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 FMCG Launch Planner")
    st.markdown("---")
    st.subheader("Generate Launch Report")
    st.caption("Fill in the product details and click Generate to run a full multi-agent analysis.")

    product_name = st.text_input("Product Name", placeholder="e.g. ZestUp Energy Drink")
    category = st.text_input("Category", placeholder="e.g. Energy Drink")
    target_market = st.text_input("Target Market", placeholder="e.g. Urban youth 18-30")
    usp = st.text_input("USP / Key Differentiator", placeholder="e.g. Zero sugar, natural caffeine")

    generate_clicked = st.button("⚡ Generate Report", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Or ask any question in the chat on the right →")
    st.markdown("**Domain experts available:**")
    st.markdown("- 📊 Marketing & Pricing")
    st.markdown("- 📋 Regulatory & Compliance")
    st.markdown("- 🏭 Supply Chain")
    st.markdown("- 💰 Finance & P&L")

# ── Main panel header ────────────────────────────────────────────────────────
st.title("FMCG Product Launch Planner")
st.markdown(
    "Use the **sidebar** to generate a full AI-powered launch report, "
    "or **ask a question below** to consult a domain expert directly."
)
st.markdown("---")

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Report mode ───────────────────────────────────────────────────────────────
if generate_clicked:
    if not all([product_name.strip(), category.strip(), target_market.strip(), usp.strip()]):
        st.warning("Please fill in all four fields in the sidebar before generating a report.")
    else:
        user_msg = (
            f"**Generate full launch report for:**\n\n"
            f"- **Product:** {product_name}\n"
            f"- **Category:** {category}\n"
            f"- **Target Market:** {target_market}\n"
            f"- **USP:** {usp}"
        )
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Running launch analysis across all 5 agents... this may take a minute or two."):
                try:
                    report = run_report(product_name, category, target_market, usp)
                    st.markdown(report)
                    st.session_state.messages.append({"role": "assistant", "content": report})
                except Exception as e:
                    error_msg = f"⚠️ Error running report: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ── Chat mode ─────────────────────────────────────────────────────────────────
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
