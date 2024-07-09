from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Embed and store vectors in memory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

import pdfplumber
import spacy

torch.cuda.empty_cache()

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file and maintains page number and text position."""
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = []
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            for word in words:
                word['page_number'] = page_number
            pages_text.extend(words)
    return pages_text

def preprocess_text(words):
    """Preprocess the text by removing unnecessary spaces, new lines, and special characters like bullets."""
    text = ' '.join([word['text'] for word in words])
    # Remove common bullet characters and other special items
    cleaned_text = text.replace('•', '').replace('◦', '').replace('‣', '').replace('-', ' ').replace('*', '')
    # Normalize spaces
    return ' '.join(cleaned_text.split())




def chunk_text(words, max_chunk_size=500, overlap=50):
    """Chunks the text based on semantic structure using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    cleaned_text = preprocess_text(words)
    doc = nlp(cleaned_text)
    chunks = []
    current_chunk = ""
    current_chunk_words = []
    buffer = []  # To store words for possible overlap

    word_iter = iter(words)  # Create an iterator over words
    for sent in doc.sents:
        sent_words = []
        sent_text = sent.text.split()
        for _ in range(len(sent_text)):
            try:
                sent_words.append(next(word_iter))
            except StopIteration:
                break
        buffer.extend(sent_words)
        if len(current_chunk) + len(sent.text) > max_chunk_size:
            # When max size is exceeded, finalize the current chunk
            chunks.append((current_chunk.strip(), current_chunk_words))
            # Start new chunk with overlap from the buffer
            overlap_words = buffer[-overlap:] if len(buffer) > overlap else buffer
            current_chunk = ' '.join([word['text'] for word in overlap_words])
            current_chunk_words = overlap_words
            buffer = []  # Clear buffer or start new for next round of overlap
        else:
            current_chunk += sent.text + ' '
            current_chunk_words.extend(sent_words)

    if current_chunk:  # Append the last chunk if not empty
        chunks.append((current_chunk.strip(), current_chunk_words))

    return chunks



def create_document_chunks(pdf_path):
    """Creates document chunks with metadata."""
    words = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(words)

    documents = []
    for content, metadata_words in chunks:
        metadata = {
            'source': pdf_path,
            'page_numbers': list(set([word['page_number'] for word in metadata_words])),
            #'positions': [(word['x0'], word['top'], word['x1'], word['bottom']) for word in metadata_words]
        }
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    return documents


# Example Usage
pdf_path = './example.pdf'
documents = create_document_chunks(pdf_path)

# Initialize the embedding model
embed_model = HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1")

# Create an in-memory FAISS vector database
vector_db = FAISS.from_documents(documents, embed_model)

# Load inference model and setup
model_id = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

messages = []

def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation

    return formatted_input

terminators = tokenizer.eos_token_id

while True:
    user_query = input("Enter your question (or 'q' to quit): ")
    if user_query.lower() == 'q':
        break

    messages.append({"role": "user", "content": user_query})
    results, scores, _ = vector_db.similarity_search(user_query, k=3)
    large_results=vector_db.similarity_search(user_query, k=10)
    print("next")

    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]

    # Extract the text content from the document
    # Aggregate content from the top three documents
    print(results)
    text_content = " ".join(doc.page_content for doc in results[:3])
    print(text_content)

    formatted_text = f"""{large_results}"""

       # Tokenize and count tokens
    tokenized_large_results = tokenizer(formatted_text)
    token_count = len(tokenized_large_results['input_ids'])
    print("Number of tokens in stringified large_results:", token_count)

        # Optionally collect metadata for display
    metadata = "\n".join(f"Source {i+1}: {doc.metadata}" for i, doc in enumerate(results[:3]))


    formatted_input = get_formatted_input(messages, formatted_text)
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=tokenized_prompt.input_ids,
        attention_mask=tokenized_prompt.attention_mask,
        do_sample=True,
        max_new_tokens=1024,
        temperature=1.2,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        num_beams=5,
        eos_token_id=terminators,
        num_return_sequences=1,
        length_penalty=1.5
    )
    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    assistant_response = tokenizer.decode(response, skip_special_tokens=True)
    full_response = f"Assistant: {assistant_response}\n\n{metadata}"
    print("Assistant:", full_response)

    messages.append({"role": "assistant", "content": assistant_response})

print("Goodbye!")