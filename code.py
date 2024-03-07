import difflib

# Example sentences
sentence1 = "The Boy is playing in the ground "
sentence2 = "The man is playing at the football stadium"

# Tokenize sentences
tokens1 = sentence1.split()
tokens2 = sentence2.split()

# Get word alignment
matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
matches = matcher.get_matching_blocks()

# Align words
aligned_words = []
for match in matches:
    aligned_words.extend(tokens1[match.a:match.a+match.size])

# Print aligned words
print(aligned_words)
