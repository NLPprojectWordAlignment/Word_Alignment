from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Example sentences
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A fast brown fox leaps over a lazy canine."

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode sentences
tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)
tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
token_ids = tokenizer.convert_tokens_to_ids(tokens)
segments_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)

# Convert to tensor
input_ids = torch.tensor([token_ids])
segments_ids = torch.tensor([segments_ids])

# Obtain BERT embeddings
with torch.no_grad():
    outputs = model(input_ids, token_type_ids=segments_ids)
    embeddings = outputs.last_hidden_state

# Extract embeddings for the first token ([CLS]) of each sentence
embedding1 = embeddings[0, 0].numpy().reshape(1, -1)
embedding2 = embeddings[0, len(tokens1) + 1].numpy().reshape(1, -1)

# Calculate cosine similarity
similarity = cosine_similarity(embedding1, embedding2)[0][0]

print("Similarity between the two sentences:", similarity)
