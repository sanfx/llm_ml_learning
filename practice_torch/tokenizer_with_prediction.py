import torch
# Step 1: Import the tokenizer class from HuggingFace Transformers
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoModelForQuestionAnswering

# Step 2: Load a pretrained tokenizer (e.g., BERT base uncased)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenizer.add_tokens(["U.K."])  # Add abbreviation as a new token
# Step 3: Define your input sentence
sentence = "Apple is looking at buying U.K. startup for $1 billion."

# Step 4: Tokenize the sentence and convert to token IDs
# This includes adding special tokens like [CLS] and [SEP]
encoded = tokenizer(sentence, return_tensors="pt")
indexed_tokens = encoded['input_ids'][0].tolist()
# Step 5a: Extract token IDs and tokens
token_ids = encoded["input_ids"][0]         # Tensor of token IDs
tokens = tokenizer.convert_ids_to_tokens(token_ids)  # Human-readable tokens

# Step 6a: Print results for clarity
print("Tokens:", tokens)
print("Token IDs:", token_ids.tolist())

# Step 5b: Reverse mapping — decode token IDs back to text
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

# Step 6b: Print results
print("Decoded Text:", decoded_text)

cls_token = 101
sep_token = 102

def get_segmented_ids(indexed_tokens):
    segmented_ids = []
    segment_id = 0

    for token in indexed_tokens:
        if token == sep_token:
            segment_id += 1
        segmented_ids.append(segment_id)
    segmented_ids[-1] -= 1 # ignore the last [SEP]
    return torch.tensor([segmented_ids]), torch.tensor([indexed_tokens])

# output result of prints
"""
Tokens: ['[CLS]', 'Apple', 'is', 'looking', 'at', 'buying', 'U.K.', 'start', '##up', 'for', '$', '1', 'billion', '.', '[SEP]']
Token IDs: [101, 7302, 1110, 1702, 1120, 9241, 28996, 1838, 4455, 1111, 109, 122, 3775, 119, 102]
Token IDs: [101, 7302, 1110, 1702, 1120, 9241, 28996, 1838, 4455, 1111, 109, 122, 3775, 119, 102]
Decoded Text: Apple is looking at buying U.K. startup for $ 1 billion.
"""

segments_tensors, tokens_tensor = get_segmented_ids(indexed_tokens)
masked_index = 3
indexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([indexed_tokens])
print(f"Masked sentence {tokenizer.decode(indexed_tokens)}")

masked_lm_model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
masked_lm_model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

if predictions:
    print(predictions[0].shape)
    predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
    print(f"Predicted index: {predicted_index}")
    word = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(f"Word is {word}")
    # question_answering_model = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    question_answering_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    question_answering_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    indexed_tokens = question_answering_tokenizer.encode(sentence, add_special_tokens=True)
    segments_tensors, tokens_tensor = get_segmented_ids(indexed_tokens)

    with torch.no_grad():
        out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)
        answer_sequence = indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1]
        print(f"answer sequence: {answer_sequence}")
        masked_word_list = question_answering_tokenizer.convert_ids_to_tokens(answer_sequence)
        result = question_answering_tokenizer.decode(answer_sequence)
        print(f"Result: {result}")