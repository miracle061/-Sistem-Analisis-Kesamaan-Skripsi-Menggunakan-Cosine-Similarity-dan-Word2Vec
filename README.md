# Plagiarism Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A plagiarism detection system that analyzes PDF documents using a combination of **Word2Vec**, **BERT**, and **TF-IDF**.  
Supports **document-level** and **sentence-level** similarity with confidence scoring.

---

## Features

- Upload PDF documents for analysis
- Preprocessing: tokenization, stemming, stopword removal
- Vectorization: Word2Vec + BERT + TF-IDF weighted
- Document similarity ranking using cosine similarity
- Detailed sentence-level comparison with Jaccard similarity and confidence score
- REST API backend built with Flask
- Supports previewing uploaded PDFs
- Filters out references, acknowledgements, and non-content sections

---

## System Flow

The workflow of the system:

```mermaid
graph TB
    Start([User]) --> Upload[Upload PDF]
    Upload --> Store[(Store in uploads/)]
    
    Store --> Extract[Extract & Filter Text]
    Extract --> PreProc[Preprocess Text<br/>- Tokenize<br/>- Stem<br/>- Remove Stopwords]
    
    PreProc --> Vector[Generate Combined Vector<br/>Word2Vec + BERT]
    
    Vector --> Compare{Compare with<br/>Database}
    
    DB[(Vector Database<br/>115 Documents)] -.-> Compare
    
    Compare --> Rank[Rank by<br/>Cosine Similarity]
    Rank --> Filter[Filter > 75% threshold]
    Filter --> TopDocs[Return Top Matches]
    
    TopDocs --> Detail{Request<br/>Details?}
    
    Detail -->|Yes| SentExtract[Extract Sentences<br/>Per Page]
    SentExtract --> SentVec[Compute Sentence<br/>Vectors]
    SentVec --> SentComp[Compare Sentences<br/>Cosine + Jaccard]
    SentComp --> ConfScore[Calculate<br/>Confidence Score]
    ConfScore --> MatchList[Return Matched<br/>Sentences with Pages]
    
    Detail -->|No| End([Display Results])
    MatchList --> End
    
    Models[Models Layer<br/>- Word2Vec<br/>- BERT MiniLM<br/>- TF-IDF Weights] -.-> Vector
    Models -.-> SentVec
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style DB fill:#fff4e1
    style Models fill:#f0e1ff
    style Compare fill:#ffe1e1
    style Detail fill:#ffe1e1
