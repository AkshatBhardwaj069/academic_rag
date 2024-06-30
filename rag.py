import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from pdfminer.high_level import extract_text  
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import anthropic  

scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

PDF_PATH = r'path to pdf'
text = extract_text(PDF_PATH)  

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

text_chunks = chunk_text(text)

def get_embeddings(text):
    inputs = scibert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = scibert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

dimension = get_embeddings(text_chunks[0]).shape[1]
index = faiss.IndexFlatL2(dimension)
text_embeddings = np.vstack([get_embeddings(chunk) for chunk in text_chunks])
index.add(text_embeddings)

def multi_stage_retrieval(query, k=5):
    query_embedding = get_embeddings(query)
    D, I = index.search(query_embedding, k)
    initial_results = [text_chunks[i] for i in I[0]]
    
    subqueries = [summarize(doc) for doc in initial_results]
    refined_results = []
    for subquery in subqueries:
        subquery_embedding = get_embeddings(subquery)
        D_sub, I_sub = index.search(subquery_embedding, k)
        refined_results.extend([text_chunks[i] for i in I_sub[0]])
    
    return list(set(refined_results))[:k]

def contextual_query_understanding(query, context):
    input_text = f"question: {query} context: {context}"
    inputs = t5_tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = t5_model.generate(inputs.input_ids, max_length=64, num_beams=4, early_stopping=True)
    contextual_query = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return contextual_query

def contextual_retrieval(query, text, k=5):
    context = text[:512]
    contextual_query = contextual_query_understanding(query, context)
    return multi_stage_retrieval(contextual_query, k)

def summarize(document):
    inputs = bart_tokenizer(document, return_tensors='pt', max_length=1024, truncation=True)
    with torch.no_grad():
        summary_ids = bart_model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

client = anthropic.Client(api_key="claude_Api_key") 

def chatbot_response(user_query):
    results = contextual_retrieval(user_query, text)
    combined_results = ' '.join(results)
    response = client.completions.create(
        model="claude-2",
        prompt=f"Human: {user_query}\n\nAssistant: Based on the following information: {combined_results}",
        max_tokens_to_sample=150
    )
    return response.completion

if __name__ == '__main__':
    user_query = "What is the main finding of the paper?"
    response = chatbot_response(user_query)
    print(response)