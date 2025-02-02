import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize the models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
response_model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(response_model_name)
response_model = T5ForConditionalGeneration.from_pretrained(response_model_name)

def read_file_chunks(file_path):
    """
    Reads a text file and splits it into chunks by newline.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        chunks = file.read().strip().split("\n")
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def generate_embeddings(chunks):
    """
    Generates embeddings for the text chunks.
    """
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return dict(zip(chunks, embeddings))

def find_similar_chunks(query, embeddings_dict, top_n=3):
    """
    Finds the top N similar text chunks based on cosine similarity.
    """
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu()  # Move to CPU

    similarities = {}
    for text_chunk, embedding in embeddings_dict.items():
        sim_score = cosine_similarity(
            query_embedding.cpu().reshape(1, -1),
            embedding.cpu().reshape(1, -1)
        )[0][0]
        similarities[text_chunk] = sim_score

    sorted_chunks = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return [chunk for chunk, score in sorted_chunks[:top_n]]

def generate_response(query, relevant_chunks):
    """
    Generates a response using the query and relevant text chunks.
    """
    combined_prompt = f"Query: {query}\nRelevant Information: {' '.join(relevant_chunks)}\nResponse: "

    # Tokenize and generate response
    input_ids = tokenizer.encode(combined_prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = response_model.generate(input_ids, max_length=200, num_return_sequences=1) #, early_stopping=True

    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response_text

if __name__ == "__main__":
    file_path = "The_Social_and_Cultural_Order_of_Ancient.txt"
    #input("Enter the path to the text file: ")
    text_chunks = read_file_chunks(file_path)

    # Generate embeddings for the text chunks
    embeddings_dict = generate_embeddings(text_chunks)

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Find similar chunks and generate a response
        similar_chunks = find_similar_chunks(query, embeddings_dict)
        response = generate_response(query, similar_chunks)

        print("\nGenerated Response:\n", response, "\n")
