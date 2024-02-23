# Transformer

## Word Embedding - nn.Embedding

Embedding Dim
: The dimensionality of the vector representation for each word/token.
  This determines how much information can be encoded in that single vector.

Hidden Dim
: The dimensionality of the hidden states within the transformer's attention
  layers. This represents the model's internal working memory and capacity to
  process relationships between words in the sequence.

Padding Idx
: During text processing, sequences often need to be padded to a uniform
  length. The padding_idx identifies which index within your word/token
  vocabulary is the "padding" symbol. The nn.Embedding layer will produce
  a zero vector (all values are 0) for any word/token whose index matches
  the padding_idx. This effectively ignores padded elements, preventing
  them from contributing to your model's calculations.

Embeddings start with relatively lower dimensionality and are projected into a
higher-dimensional space for computation within the attention mechanisms.

## Positional Encoding - Custom nn.Module

Model dim
: The dimensionality of the word embeddings for a transformer's task.

Dropout
: Transformer models initially lack an inherent understanding of word order.
  Positional encodings are the sole providers of sequential information. This
  creates the risk of the model overly relying on  exact positional
  representations, potentially hindering generalization.

## Positional Embedding - Word embedding + Postional Encoding

1. embedding = nn.Embedding(vocab_size, d_model)
2. pos_encoder = PositionalEncoding(d_model)
3. x = embedding(input_tokens)
4. x = pos_encoder(x)
5. input -> embedding layer -> pos_encoding layer -> dropout layer -> output

## Self-Attention

The model is attending to different positions within the same input sequence.
Query, key, and value vectors for each word/token are derived from the initial
word/token embeddings within the same sequence. When using this same sequence
as input for nn.MultiheadAttention, the model handles self-attention with
multiple heads.

## Multi-Head Attention - nn.MultiheadAttention

num_heads
: Number of parallel attention heads. Note that embed_dim will be split
  across num_heads.

Each head operates on smaller, projected representations of the input.
These multiple parallel heads enable the model to focus on different aspects
or 'subspaces' of the input simultaneously, enriching the representations it
learns.

## Masked Multi-Head Attention(Decoder)

Autoregressive Generation
: The decoder generates text (or other sequences) one token at a time.
  During generation, it must avoid attending to future tokens it hasn't
  yet predicted – this would "leak" information and break the logic of
  the model.

Causal Generation
: Masked multi-head attention enforces a causal language modeling structure,
  ensuring the decoder's prediction at each step relies only on previous tokens.

## Encoder-Decoder Attention(Decoder) Cross-Attention

Encoder Outputs
: The final output of the encoder contains rich contextual representations
  for each token in the input sequence. These outputs serve as the keys (K)
  and values (V) for the decoder's multi-head attention.

Decoder Self-Attention Outputs
: The output of the decoder's first masked multi-head attention layer. This
  carries information about what the decoder has generated so far. It serve
  as the Query (Q) for decoder's second multi-head attention.

## Residual Connection

The output of the multihead attention will be added back to this original input
before proceeding to the next layer. This helps stabilize training and allows
information to flow more easily through gradients.

## Layer Normalization

Focus on a Single Sample
: Layer Normalization operates independently on each data point
  (e.g., each image or each sentence) within a batch.

Element-Wise Standardization
: Element is feature or dimension. By normalizing across features within each
  data point, LN ensures that no single feature dominates the subsequent
  calculations in the layer. This creates better stability during training.

Batch Independence
: Since each sample is normalized independently, the input sequence length or
  varying batch sizes won't destabilize Layer Normalization.

Caculate mean, caculate variance and Normalization.

## Positionwise Feed-Forward Network

Position-Wise
: Processes each token's embedding vector independently.

Non-Linear Transformation
: While self-attention excels at capturing relationships within a sequence,
  it's inherently linear. The FFN adds non-linearity to the model, allowing
  it to learn more complex transformations on the representations produced
  by self-attention.

Feature Processing
: The FFN acts independently upon each token's representation from the
  preceding self-attention layer. This means it can refine features within
  each position of the sequence.

Feed-Forward Network
: The FFN in a Transformer operates independently on the representation of
  each token in the sequence. You can think of each token having its own
  tiny, private feed-forward network.

Sequential Linear Transformations
: The core of the FFN comprises two linear (fully-connected) layers. Data
  for each token undergoes these transformations sequentially.

## Targets Shifted Right

During training, the decoder needs to know the correct word to predict and
the words that came before it. Shifting the target sequence right by one
position and inserting a "start" token achieves this.
