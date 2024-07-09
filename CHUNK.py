import pdfplumber
import spacy

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
            'positions': [(word['x0'], word['top'], word['x1'], word['bottom']) for word in metadata_words]
        }
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    return documents

# Usage
pdf_path = '/path/to/your/pdf_file.pdf'
documents = create_document_chunks(pdf_path)
