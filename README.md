# Data Processing Pipeline

This document describes the steps involved in our data processing pipeline. The following sections detail each stage of the process from data cleaning to global clustering.

---

## 1. Vocabulary Filtering

- **Objective:** Remove noise and focus on frequently occurring words.
- **Process:**  
  - For each dataset, filter out words that appear less than 3 times.
  - This step helps in reducing noise and ensuring that only the most common words are considered for further analysis.

---

## 2. Document Filtering

- **Objective:** Ensure documents are meaningful for analysis.
- **Process:**  
  - Remove any document that contains fewer than 2 words.
  - Documents that are too short do not provide sufficient context or meaning for analysis.

---

## 3. Global Clustering

### 3.1. Embedding Generation

- **Pretrained Model:**  
  - We utilize the `all-MiniLM-L6-v2` model to convert documents into semantic representations (embeddings).
- **Purpose:**  
  - These embeddings capture the underlying semantic meaning of the documents, allowing us to perform clustering based on content similarity.

### 3.2. Clustering Algorithms

- **Algorithms Used:**  
  - **K-Means:** Groups similar documents into clusters based on their embeddings.
  - **UMAP:** Reduces the dimensionality of embeddings for visualization and further analysis.
- **Outcome:**  
  - Documents are grouped into clusters, facilitating a global overview of similar content across the dataset.

---

## Summary

This pipeline ensures that our data is preprocessed effectively by:
- Filtering out infrequent words and extremely short documents,
- Converting documents into robust semantic embeddings using a state-of-the-art pretrained model,
- Clustering the documents to identify meaningful groups using K-Means and UMAP.

Each step is designed to enhance the quality of the data analysis and improve the interpretability of the results.
