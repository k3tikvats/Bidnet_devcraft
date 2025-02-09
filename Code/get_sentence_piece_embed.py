from sentence_transformers import SentenceTransformer

def encode_text(sentences):
    """
    Encode text using a pretrained Sentence-BERT model.
    :param model_name: Name of the SBERT model to use.
    :param sentences: List of sentences to encode.
    :return: List of sentence embeddings.
    """
    model_name="sentence-t5-xl"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings

if __name__ == "__main__":
    # model_name = "sentence-t5-xl"  # Choose an appropriate SBERT model
    sentences = [
        "Hello, how are you?",
        "Sentence embeddings are useful for NLP tasks.",
        "This is an example of Sentence-BERT encoding."
    ]
    
    embeddings = encode_text( sentences)
    
    for i, embedding in enumerate(embeddings):
        print(embedding.shape)
        # print(f"Sentence {i+1} Embedding: {embedding[:5]}... (truncated)")
