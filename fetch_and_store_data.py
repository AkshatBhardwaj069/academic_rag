import requests
import xml.etree.ElementTree as ET
import sqlite3
from datetime import datetime

def fetch_arxiv_data(query, max_results=100):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    response = requests.get(base_url, params=params)
    return response.content

def parse_arxiv_data(data):
    root = ET.fromstring(data)
    namespace = {"ns": "http://www.w3.org/2005/Atom"}
    articles = []
    
    for entry in root.findall("ns:entry", namespace):
        arxiv_id = entry.find("ns:id", namespace).text.split('/')[-1]
        title = entry.find("ns:title", namespace).text.strip()
        summary = entry.find("ns:summary", namespace).text.strip()
        published = entry.find("ns:published", namespace).text.strip()
        
        articles.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "published": published
        })
    return articles

def save_to_database(db_name, articles):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            published DATE
        )
    ''')
    for article in articles:
        cursor.execute('''
            INSERT OR REPLACE INTO papers (arxiv_id, title, summary, published)
            VALUES (:arxiv_id, :title, :summary, :published)
        ''', article)
    conn.commit()
    conn.close()

def update_arxiv_data(query, db_name="arxiv_papers.db"):
    data = fetch_arxiv_data(query)
    articles = parse_arxiv_data(data)
    save_to_database(db_name, articles)

if __name__ == "__main__":
    ai_categories = [
        "cs.AI",     
        "cs.LG",     
        "stat.ML",   
        "cs.CL",     
        "cs.CV",     
        "cs.RO",     
        "cs.HC",     
    ]

    for category in ai_categories:
        update_arxiv_data(category)
