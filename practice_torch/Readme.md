preprocessing step (map IDs to vectors).
===

<code>self.embed = nn.Embedding(vocab_size, hidden_size)</code>

Embedding (TinyLM):

    → Converts discrete token IDs → dense vectors (learned representations).
    → Example: word "cat" (ID=5) → vector [0.2, -0.7, 1.1, …].
    → Purpose: gives the model semantic information instead of just raw IDs.

Final prediction layer of a language model
===
<h4>line of code below is an important part of how a language model generates words/tokens</h4>
<code> self.fc = nn.Linear(hidden_size, vocab_size)</code>

1. <b>in_features = hidden_size</b>
→ the number of neurons in the hidden representation (the size of the vector output by your model’s encoder/transformer/hidden state).

2. <b>out_features = vocab_size</b>
→ the number of possible tokens in your vocabulary (all the words, characters, or subwords your model can predict).

<p>So this layer <b>maps from the model’s hidden state → to a probability distribution over the vocabulary.</b></p>

--------------------------

Why is this needed in a Language Model?
=====

1. Inside the network (RNN, Transformer, etc.), you work with hidden states (e.g. vectors of size 128, 256, etc.).
These don’t directly correspond to words.

2. At the end, you need to predict the next token from your vocabulary.<br><b>Example:</b>

    1. Hidden vector size = 128
    2. Vocabulary size = 10,000 words
    3. You need to map 128 → 10,000

3. The nn.Linear(hidden_size, vocab_size) does exactly that mapping.
4. After this, you usually apply a softmax:
    1. logits → raw scores for each token in vocab
    2. softmax(logits) → converts them into probabilities

But, no softmax in this rnn model! why?

----
1. Training efficiency

    PyTorch has a special loss function:
<code>nn.CrossEntropyLoss()</code>
It expects raw logits (not softmaxed probabilities).

2. Internally, it applies <b>log-softmax + NLLLoss</b> in one step → more stable numerically.

If we applied softmax in the model, we’d be doing extra unnecessary work.

✅ <b>Summary</b>

1. TinyLM ends with nn.Linear(hidden_size, vocab_size).

2. No softmax inside the class → because nn.CrossEntropyLoss already expects logits.

3. <b><font color=red>You apply torch.softmax (or argmax) outside when you need probabilities or predictions</font></b>.

<img src="./image/flow_of_prediction.png" />

---
internal RNN step (combine current input with memory from the past).
====
Example api: `torch.cat` used <a href="https://github.com/patrickloeber/pytorch-examples/blob/master/rnn-name-classification/rnn.py#L20">here</a>

Used when dataset (character classification on names) already represents characters as one-hot vectors (like `[0,0,1,0,...]`).

<code>combined = torch.cat((input_tensor, hidden_tensor), 1)</code>

    torch.cat (RNN):
    → Concatenates input features + previous hidden state into one long vector.
    → Example: [input_t, hidden_t] → [0.4, 0.9, -0.3, …, 0.8, 0.5].
    → Purpose: manually builds the “combined” vector that the custom RNN cell will process.

<img src="./image/407735c8-8ae2-44e1-a940-e3af8058ee78.png" />

<h3>🔍 Where tokenization Fits in the entire machine learning pipeline</h3>
<p>in case of <a href="tokenizer.py"><code>tokenzier.py</code></a> it refers to <b>NLP modeling pipeline</b></p>
<table>
  <thead>
    <tr>
      <th style="text-align:left;">Stage</th>
      <th style="text-align:left;">Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pre-classification</td>
      <td>Text → Token IDs</td>
    </tr>
    <tr>
      <td>Classification</td>
      <td>Token IDs → Embeddings → Model Output</td>
    </tr>
    <tr>
      <td>Post-classification</td>
      <td>Output IDs → Decoded Text</td>
    </tr>
  </tbody>
</table>
