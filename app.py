import streamlit as st
from scripts.ask_bot import ask_legal_bot

st.set_page_config(page_title="Legal Bot (RAG Based)", page_icon="⚖️")

st.title("⚖️ Legal Bot")
st.markdown("Ask anything about Indian laws ")

query = st.text_input("📩 Ask your question:", placeholder="e.g. What is Section 302 of IPC?")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        st.write("⏳ Running `ask_legal_bot`...")
        print("📩 Query Received:", query)

        try:
            answer, _ = ask_legal_bot(query)  # discard sources
            print("✅ Answer:", answer)
        except Exception as e:
            st.error(f"Error: {e}")
            print("❌ Error occurred:", e)
            answer = None

    if answer:
        st.success("✅ Answer:")
        st.markdown(answer)  # use markdown to support formatting like **bold**, lists, etc.
    else:
        st.warning("🤖 I couldn't find an answer. Try rephrasing or uploading more relevant documents.")

