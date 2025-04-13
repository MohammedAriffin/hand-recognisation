from transformers import AutoTokenizer, BartForConditionalGeneration
import torch

# --- Load Model and Tokenizer ---
print("Loading BART tokenizer and model (facebook/bart-large-cnn)...")
# This model is pre-trained for SUMMARIZATION, not space insertion.
# We use BartForConditionalGeneration because it's designed for text generation (Seq2Seq).
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
print("Model and tokenizer loaded.")
print("-" * 30)

# --- Function to test BART generation ---
def test_bart_generation(text_input, max_output_length=50):
    print(f"Input Text: '{text_input}'")

    # Tokenize the input for the Encoder
    # Note: BART expects the input wrapped in a list for batching
    inputs = tokenizer([text_input], max_length=1024, return_tensors="pt", truncation=True)

    print("Tokenized Input IDs:", inputs['input_ids'])
    # Convert IDs back to tokens to see how the text was split
    input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print("Input Tokens:", input_tokens)

    # Generate text using the Decoder
    print("Generating output...")
    with torch.no_grad(): # No need to calculate gradients for inference
        # Using generate method for sequence-to-sequence models
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=4, # Use beam search for potentially better results
            min_length=5, # Minimum length of the output
            max_length=max_output_length, # Maximum length of the output
            early_stopping=True # Stop when end-of-sequence token is generated
        )

    # --- Decode and Examine the Output ---
    # Decode the generated token IDs back into a string
    generated_text = tokenizer.batch_decode(
        summary_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True # Tries to clean up spaces around tokens
    )[0] # Get the first result from the batch

    print("Generated Output Text:", generated_text)
    print("-" * 30)
    return generated_text

# --- Test Case 1: Normal Input ---
# Expectation: Might try to summarize or just reproduce, depending on length.
test_bart_generation("Hello, my dog is cute")

# --- Test Case 2: Unspaced Input (Simulating Buffer) ---
# Expectation: Will likely NOT insert spaces correctly. Might generate nonsense,
# repeat parts, or try to "summarize" the character stream poorly.
unspaced_long = "thequickbrownfoxjumpsoverthelazydog"
test_bart_generation(unspaced_long, max_output_length=len(unspaced_long) + 10) # Allow for spaces

# --- Test Case 3: Short Unspaced Input ---
unspaced_short = "thisisatestsequence"
test_bart_generation(unspaced_short, max_output_length=len(unspaced_short) + 10)

# --- Test Case 4: Unspaced Input with a "Prompt" ---
# Expectation: BART (especially this summarization one) doesn't treat the start
# of the input as an instruction like GPT. It encodes the whole thing.
# Output is highly unpredictable but unlikely to follow the "instruction".
prompt = "Restructure this sequence: "
unspaced_text = "addspaceshereplease"
test_bart_generation(prompt + unspaced_text, max_output_length=50)

