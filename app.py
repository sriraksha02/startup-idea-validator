import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import pandas as pd

# ---------------- DATABASE ----------------
conn = sqlite3.connect("ideas.db", check_same_thread=False)
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS ideas (
    idea TEXT,
    audience TEXT,
    score INTEGER
)
''')
conn.commit()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Startup Validator", page_icon="🚀", layout="wide")

# ---------------- HEADER ----------------
st.title("🚀 AI Startup Idea Validator")
st.markdown("### Evaluate your startup idea using Natural Language Processing")
st.markdown("---")

# ---------------- DATASET ----------------
startup_ideas = [
    "AI-powered food delivery optimization system",
    "Online learning platform for students",
    "Remote healthcare monitoring system",
    "E-commerce platform for fashion",
    "Fitness tracking mobile application",
    "AI chatbot for mental health support",
    "Smart home automation system",
    "Blockchain-based voting system",
    "AI resume screening tool",
    "Online marketplace for freelancers",
    "Virtual reality learning platform",
    "IoT-based agriculture monitoring",
    "AI-based stock prediction tool",
    "Personal finance management app",
    "On-demand tutoring platform",
    "Food waste reduction app",
    "AI-based career guidance system",
    "Smart parking management system",
    "Pet care service platform",
    "Subscription-based grocery delivery",
    "AI-based language translation tool",
    "Online event management system",
    "Doctor appointment booking platform",
    "AI-powered fitness coach",
    "Crowdfunding platform for startups"
]

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    idea = st.text_area("💡 Startup Idea", placeholder="Describe your startup idea clearly...")

with col2:
    audience = st.text_input("🎯 Target Audience", placeholder="e.g. Students, Professionals")

analyze_btn = st.button("🔍 Analyze Idea")

# ---------------- ANALYSIS ----------------
if analyze_btn:

    # 🔴 Improved validation
    if len(idea.strip()) < 15:
        st.warning("Please provide a more detailed startup idea (minimum 15 characters).")

    elif not any(char.isalpha() for char in idea):
        st.warning("Please enter a meaningful idea using proper words.")

    else:
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(startup_ideas + [idea])

            similarity = cosine_similarity(vectors[-1], vectors[:-1])
            max_similarity = similarity.max()

            # 🔥 Better scoring logic
            score = max(3, min(10, int((1 - max_similarity) * 10)))

            if len(idea.split()) < 4:
                score -= 2

            score = max(1, score)

            # Save to DB
            c.execute("INSERT INTO ideas VALUES (?, ?, ?)", (idea, audience, score))
            conn.commit()

            # ---------------- OUTPUT ----------------
            st.markdown("---")
            st.subheader("📊 Evaluation Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Idea Score", f"{score}/10")

            with col2:
                st.metric("Similarity Index", f"{round(max_similarity, 2)}")

            # Feedback
            if max_similarity > 0.7:
                st.error("This idea is highly similar to existing solutions. Consider adding unique features.")
            elif max_similarity > 0.4:
                st.warning("This idea shows moderate similarity. Some improvements are recommended.")
            else:
                st.success("The idea demonstrates uniqueness with good potential for innovation.")

            # Suggestions
            st.markdown("### 💡 Recommendations")
            st.markdown("""
            - Clearly define the problem you are solving  
            - Focus on a niche audience  
            - Add innovative or differentiating features  
            - Validate the idea with real users  
            """)

        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ---------------- DATABASE DISPLAY ----------------
st.markdown("---")
st.subheader("📂 Stored Startup Ideas")

rows = c.execute("SELECT * FROM ideas").fetchall()

if len(rows) > 0:
    df = pd.DataFrame(rows, columns=["Idea", "Audience", "Score"])

    st.download_button(
        label="📥 Download Data as CSV",
        data=df.to_csv(index=False),
        file_name="startup_ideas.csv",
        mime="text/csv"
    )

    for row in rows:
        st.markdown(f"""
        **💡 Idea:** {row[0]}  
        **🎯 Audience:** {row[1]}  
        **⭐ Score:** {row[2]}/10  
        ---
        """)
else:
    st.info("No startup ideas have been analyzed yet.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed using Streamlit, NLP (TF-IDF), and SQLite Database")