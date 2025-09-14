# Step 1: Import the tokenizer class from HuggingFace Transformers
from transformers import AutoTokenizer

# Step 2: Load a pretrained tokenizer (e.g., BERT base uncased)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.add_tokens(["U.K."])  # Add abbreviation as a new token
# Step 3: Define your input sentence
sentence = "Apple is looking at buying U.K. startup for $1 billion."

# Step 4: Tokenize the sentence and convert to token IDs
# This includes adding special tokens like [CLS] and [SEP]
encoded = tokenizer(sentence, return_tensors="pt")

# Step 5a: Extract token IDs and tokens
token_ids = encoded["input_ids"][0]         # Tensor of token IDs
tokens = tokenizer.convert_ids_to_tokens(token_ids)  # Human-readable tokens

# Step 6a: Print results for clarity
print("Tokens:", tokens)
print("Token IDs:", token_ids.tolist())

# Step 5b: Reverse mapping — decode token IDs back to text
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

# Step 6b: Print results
print("Token IDs:", token_ids.tolist())
print("Decoded Text:", decoded_text)

# output result of prints
"""
Tokens: ['[CLS]', 'Apple', 'is', 'looking', 'at', 'buying', 'U.K.', 'start', '##up', 'for', '$', '1', 'billion', '.', '[SEP]']
Token IDs: [101, 7302, 1110, 1702, 1120, 9241, 28996, 1838, 4455, 1111, 109, 122, 3775, 119, 102]
Token IDs: [101, 7302, 1110, 1702, 1120, 9241, 28996, 1838, 4455, 1111, 109, 122, 3775, 119, 102]
Decoded Text: Apple is looking at buying U.K. startup for $ 1 billion.
"""