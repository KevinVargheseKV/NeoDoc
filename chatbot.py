# type: ignore
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import gradio as gr

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Config
CHROMA_PATH = "chroma_db"

# Initialize embeddings and LLM
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,  # Lower temperature for focused answers
    google_api_key=api_key
)

# Connect to Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# List of symptoms we want to ask about
SYMPTOM_QUESTIONS = [
    "headache",
    "fever",
    "cough",
    "stomach pain",
    "difficulty breathing",
    "chest pain",
    "bleeding",
    "dizziness"
]

# We'll keep track of symptoms and conversation history
conversation_state = {
    "symptoms": {},  # symptom: True/False
    "awaiting_symptom": None,  # which symptom are we waiting for user to answer
    "history": []  # chat history for context
}

def stream_response(user_input, history):
    # Update history with user input
    conversation_state["history"] = history

    # If waiting for symptom confirmation, record answer
    if conversation_state["awaiting_symptom"]:
        symptom = conversation_state["awaiting_symptom"]
        answer = user_input.strip().lower()
        if answer in ["yes", "y"]:
            conversation_state["symptoms"][symptom] = True
        else:
            conversation_state["symptoms"][symptom] = False
        conversation_state["awaiting_symptom"] = None

    # If not waiting for symptom, check if user just typed new symptom(s)
    if not conversation_state["awaiting_symptom"]:
        # Check if user mentioned symptoms directly (simple keyword check)
        # Here you can expand this part with NLP if needed
        for symptom in SYMPTOM_QUESTIONS:
            if symptom in user_input.lower():
                conversation_state["symptoms"][symptom] = True

    # Now find next symptom to ask that user hasn't answered yet
    for symptom in SYMPTOM_QUESTIONS:
        if symptom not in conversation_state["symptoms"]:
            conversation_state["awaiting_symptom"] = symptom
            yield f"Do you have {symptom.replace('_',' ')}? (yes/no)"
            return

    # All symptoms gathered, now prepare knowledge + prompt for LLM

    # Retrieve relevant docs for the current question (all symptoms as text)
    symptom_text = ", ".join([s for s,v in conversation_state["symptoms"].items() if v])
    if not symptom_text:
        symptom_text = user_input  # fallback

    docs = retriever.invoke(symptom_text)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are NeoDoc, a highly knowledgeable AI doctor. Based ONLY on the medical knowledge below and the reported symptoms, do the following:

- Analyze all the symptoms listed.
- Suggest a list of possible diseases that match the symptoms.
- Recommend medicines or first-aid treatments appropriate for these diseases.
- Keep your answer concise and clear.
- Format your answer exactly as:
Possible Diseases: disease1, disease2, ...
Medicines: medicine1, medicine2, ...

If the knowledge is insufficient, reply: 'Please consult a doctor for an accurate diagnosis.'
If you are unsure, say: 'I don't know.'

Reported Symptoms:
{', '.join([f"{k}: {'Yes' if v else 'No'}" for k,v in conversation_state['symptoms'].items()])}

Medical Knowledge:
{knowledge}
"""

    partial = ""
    for response in llm.stream(prompt):
        partial += response.content
        yield partial


# Gradio UI setup
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(placeholder="Describe your symptoms...", container=False, scale=7),
    title="NeoDoc - Your Smart AI Doctor 🧠💊"
)

if __name__ == "__main__":
    chatbot.launch()
