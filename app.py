import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load a healthcare-specific model (optimized for Q&A tasks)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define chatbot function
def healthcare_chatbot(user_input):
    common_questions = {
        "symptom": "If you're experiencing symptoms, consider seeking medical attention.",
        "appointment": "Would you like to schedule a consultation with a doctor?",
        "medication": "Always take medications as prescribed. Contact a doctor for guidance."
    }
    
    for keyword, response in common_questions.items():
        if keyword in user_input.lower():
            return response
    
    # Use model for other questions
    context = "Healthcare involves patient care, disease prevention, treatments, and medications. Always consult a doctor for serious health concerns."
    response = qa_pipeline(question=user_input, context=context)
    return response['answer']

# Streamlit app
def main():
    st.set_page_config(page_title="Healthcare Chatbot", layout="wide")
    st.markdown("""
        <style>
            .chat-container {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
            }
            .user-message {
                color: #0052cc;
                font-weight: bold;
            }
            .bot-message {
                color: #008000;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💬 Healthcare Assistant Chatbot")
    st.markdown("I'm here to help with health-related queries. Always consult a doctor for medical advice.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # User input
    user_input = st.text_input("### Ask me a health-related question:", "")
    if st.button("Submit") and user_input:
        with st.spinner("Thinking..."):
            response = healthcare_chatbot(user_input)
        
        # Save chat history
        st.session_state.chat_history.append((user_input, response))
    
    # Display chat history
    st.markdown("---")
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"<div class='chat-container'><p class='user-message'>**You:** {user_msg}</p><p class='bot-message'>**Healthcare Assistant:** {bot_msg}</p></div>", unsafe_allow_html=True)
        st.markdown("---")

if __name__ == "__main__":
    main()
