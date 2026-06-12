import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from src.helpers import load_embedding, repo_ingestion


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
REPO_DIR = BASE_DIR / "repo"
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = load_embedding()
qa = None


def build_qa_chain():
    if not DB_DIR.exists():
        return None

    vectordb = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    llm = ChatOpenAI()
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8},
        ),
        memory=memory,
    )


qa = build_qa_chain()


@app.route("/", methods=["GET", "POST"])
def index():
    if (TEMPLATE_DIR / "index.html").exists():
        return render_template("index.html")

    return "Realtime Source Code Analyser is running. Add templates/index.html for the UI."


@app.route("/chatbot", methods=["GET", "POST"])
def git_repo():
    global qa

    if request.method != "POST":
        return jsonify(
            {"response": "Send a POST request with a GitHub repo URL in the 'question' field."}
        )

    repo_url = request.form.get("question")
    if not repo_url:
        return jsonify({"error": "Missing required form field: question"}), 400

    shutil.rmtree(REPO_DIR, ignore_errors=True)
    shutil.rmtree(DB_DIR, ignore_errors=True)

    repo_ingestion(repo_url)
    subprocess.run([sys.executable, "store_index.py"], cwd=BASE_DIR, check=True)
    qa = build_qa_chain()

    return jsonify({"response": f"Repository indexed: {repo_url}"})


@app.route("/get", methods=["GET", "POST"])
def chat():
    global qa

    question = request.values.get("msg")
    if not question:
        return "Missing required parameter: msg", 400

    print(question)

    if question.lower() == "clear":
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        shutil.rmtree(DB_DIR, ignore_errors=True)
        qa = None
        return "Repository and vector database cleared."

    if qa is None:
        return "Vector database is not ready. Submit a repository URL first.", 400

    result = qa.invoke({"question": question})
    print(result["answer"])
    return str(result["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
