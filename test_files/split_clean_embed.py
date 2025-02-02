import re
from sentence_transformers import SentenceTransformer
import json

def read_and_clean_document(file_path: str) -> str:
    """
    Reads and cleans the text from a given document.

    Args:
    file_path (str): The path to the text file.

    Returns:
    str: The cleaned text from the document.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()

        # Remove extra spaces, line breaks, and non-alphanumeric characters (excluding spaces)
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Collapse spaces and strip

        # Save the cleaned text to "Selected_Document.txt"
        with open("Selected_Document.txt", 'w', encoding='utf-8') as output_file:
            output_file.write(cleaned_text)

        return cleaned_text

    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
def read_and_split_document(file_path: str) -> list:
    """
    Reads the content of a file and splits it into chunks separated by double newline characters.

    Args:
    file_path (str): The path to the text file.

    Returns:
    list: A list of text chunks split by double newlines.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()

        # Split the text by double newline characters
        chunks = text.split('\n')
        return chunks

    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
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

# Load the pre-trained model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

def generate_embeddings(text_chunks: list) -> dict:
    """
    Generates embeddings for a list of text chunks using SentenceTransformers.

    Args:
    text_chunks (list): A list of text strings.

    Returns:
    dict: A dictionary mapping text chunks to their embeddings.
    """
    # Generate embeddings
    embeddings = model.encode(text_chunks, show_progress_bar=True)

    # Create a dictionary mapping text to its embedding
    embedding_dict = {text: embedding.tolist() for text, embedding in zip(text_chunks, embeddings)}

    return embedding_dict

# Example usage
if __name__ == "__main__":
    #cleaned_text = read_and_clean_document("The_Social_and_Cultural_Order_of_Ancient.txt")
    split_text = split_text_by_sentences("The_Social_and_Cultural_Order_of_Ancient.txt")

    # Generate embeddings and store the result
    embeddings = generate_embeddings(split_text)

    # Save the embeddings to a JSON file
    with open("text_embeddings.json", "w", encoding="utf-8") as file:
        json.dump(embeddings, file, indent=4)

    print("Embeddings saved to text_embeddings.json.")
    ###for chunk in split_text:
    ###    print(f"Chunk:\n{chunk}\n{'-'*50}")


