import sqlite3
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

class ArxivRAGChatbot:
    def __init__(self, db_name):
        self.db_name = db_name
        self.papers = self.load_papers()

    def load_papers(self):
        sql_query = "SELECT arxiv_id, title, summary, published FROM papers"
        return self.query_database(sql_query)

    def query_database(self, query):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def generate_response(self, user_input):
        context = " ".join([f"{paper[1]}: {paper[2]}" for paper in self.papers])
        response = qa_model(question=user_input, context=context)
        return response['answer']

    def handle_query(self, user_input):
        if "latest papers" in user_input.lower():
            sql_query = "SELECT * FROM papers ORDER BY published DESC LIMIT 5"
            results = self.query_database(sql_query)
            response = "Here are the latest papers:\n"
            for result in results:
                response += f"- {result[1]} (Published: {result[3]})\n"
            return response
        elif "search" in user_input.lower():
            search_term = user_input.split("search")[-1].strip()
            
            titles_summaries = [f"{paper[1]} {paper[2]}" for paper in self.papers]
            
            vectorizer = TfidfVectorizer().fit(titles_summaries)
            
            tfidf_matrix = vectorizer.transform(titles_summaries)
            query_vec = vectorizer.transform([search_term])
            
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            sorted_indices = np.argsort(-similarities)
            
            relevant_papers = [self.papers[i] for i in sorted_indices if similarities[i] > 0]
            
            if relevant_papers:
                response = f"Papers matching '{search_term}':\n"
                for paper in relevant_papers:
                    response += f"- {paper[1]} (Published: {paper[3]})\n"
                return response
            else:
                return f"No papers found matching '{search_term}'"
        else:
            return self.generate_response(user_input)

if __name__ == "__main__":
    chatbot = ArxivRAGChatbot("arxiv_papers.db")

    user_input = "Can you show me the latest papers?"
    print(chatbot.handle_query(user_input))

    user_input = "search quantum computing"
    print(chatbot.handle_query(user_input))
