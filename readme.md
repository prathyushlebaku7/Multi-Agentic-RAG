# 🤖 Multi-Agentic RAG System

An advanced Retrieval-Augmented Generation (RAG) system using multiple intelligent agents to analyze and interact with document-based data.

## 📦 Requirements

- Python >= 3.0

## 🧱 Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📚 Creating Embeddings

1. Place your PDF files into the `Asset_Management-1` folder.
2. Run the embedding script:

```bash
python embedding.py
```

## 🌐 Launch Web Interface

To start the Streamlit interface:

```bash
streamlit run app.py
```

## 🎨 UI Enhancements

The interface includes refined and interactive logos for a more engaging user experience.

## 🛠 Project Structure

```
multi-agentic-rag/
├── Asset_Management-1/
│   └── your-pdfs-here.pdf
├── embedding.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 📫 Contact

For issues or contributions, feel free to open an issue or submit a pull request.

---

Want to add badges, license, or contribution guidelines? Let me know!
