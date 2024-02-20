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

## Positional Embedding

Word embedding + Postional Encoding

1. embedding = nn.Embedding(vocab_size, d_model)
2. pos_encoder = PositionalEncoding(d_model)
3. x = embedding(input_tokens)
4. x = pos_encoder(x)

## Multi-Head Attention

## Masked Multi-Head Attention

## Residual Connection

## Layer Normalization

## Positionwise Feed-Forward Network
