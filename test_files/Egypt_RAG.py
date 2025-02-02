from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

def split_text_by_sentences(file_path: str) -> list:
    # Open the file and read its content
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()

        # Define a pattern to split text at the end of sentences, assuming sentences end with '.', '?', or '!'
        sentence_end_pattern = re.compile(r'([.!?])\s+')

        # Split the text into sentences
        sentences = sentence_end_pattern.split(text)
        
        # Rebuild the sentences with punctuation marks included
        sentences = [sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)]

        # Now split the text into chunks by double newlines
        chunks = text.split("\n")

        # Optionally, print chunks or return them for further processing
        return chunks

    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Function to embed the query and find the most similar chunks
def get_top_similar_chunks(query, embedding_dict, model, top_n=3):

    query_embedding = model.encode(query)
    
    # Calculate cosine similarities
    similarities = {
        text: cosine_similarity([query_embedding], [embedding])[0][0]
        for text, embedding in embedding_dict.items()
    }
    
    # Sort and select the top-N similar chunks
    top_chunks = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    
    return top_chunks


# Function to generate a response based on the combined chunks
def generate_response(query):
    # Initialize the Hugging Face T5 model and tokenizer
    model_name = "google/flan-t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Retrieve the top 3 most relevant text chunks based on the query
    top_chunks = get_top_similar_chunks(query)
    
    # Combine the chunks into a single prompt
    combined_prompt = " ".join(top_chunks)
    combined_prompt = f"Answer the following question based on the information: {combined_prompt} Question: {query}"

    # Tokenize the input prompt
    inputs = tokenizer(combined_prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate the response from the model
    output = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)

    # Decode the generated output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":

    # Query input
    query = input("Enter your query: ")

    # Example text chunks and their pre-computed embeddings (dictionary format)
    text_chunks = split_text_by_sentences("Selected_Document.txt")
    
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Precompute embeddings for text chunks
    embeddings = model.encode(text_chunks)
    embedding_dict = dict(zip(text_chunks, embeddings))  # Fix embedding_dict construction

    sorted_chunks = get_top_similar_chunks(query, embedding_dict, model)
    
    # Display the top 3 most similar text chunks
    print("\nTop 3 most similar text chunks:")
    for i, (chunk, similarity) in enumerate(sorted_chunks, start=1):
        print(f"{i}. {text_chunks[chunk]} (Similarity: {similarity:.4f})")



    # Generate and print the response
    #response = generate_response(query)
    #print("\nGenerated Response:")
    #print(response)
