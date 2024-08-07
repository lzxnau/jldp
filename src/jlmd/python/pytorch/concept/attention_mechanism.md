# Attention Mechanism

## Attention Pooling

Definition
: 1. Dataset: {math}`m` tuples of keys and values
     :::{math}
     \mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1),
	 \ldots (\mathbf{k}_m, \mathbf{v}_m)\}
     :::
  2. Attention Pooling: query {math}`q` that operate on ({math}`k`, {math}`v`)
     pairs
     :::{math}
	 \textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=}
	 \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,
	 :::

::::{card}
Attention Pooling by Similarity
^^^
:::{image} am_img1.svg
:align: center
:::
+++
Kernel Regression

1. Non-parametric Model: Captures complex, non-linear relationships.
2. Focus on Similarity: Kernel shape and bandwidth control how it adapts
   to local data structure.
3. Kernel Function: Less sensitive to outliers than parametric models.
::::

::::{card}
Attention Scoring Function
^^^
:::{image} am_img2.svg
:align: center
:::
::::

::::{card}
Sequence-to-sequence Model
^^^
:::{image} am_img3.svg
:align: center
:::
::::

::::{card}
The Bahdanau Attention Mechanism
^^^
:::{image} am_img4.svg
:align: center
:::
::::

::::{card}
Multi-Head Attention
^^^
:::{image} am_img5.svg
:align: center
:::
::::

::::{card}
CNNs, RNNs, and Self-Attention
^^^
:::{image} am_img6.svg
:align: center
:::
::::

::::{card}
The Transformer Architecture
^^^
:::{image} am_img7.svg
:align: center
:::
::::

::::{card}
The Vision Transformer Architecture
^^^
:::{image} am_img8.svg
:align: center
:::
::::

## Large-Scale Pretraining with Transformers

1. Encoder-Only
   ::::{card}
   Pretraining BERT
   ^^^
   :::{image} am_img9.svg
   :align: center
   :::
   ::::

   ::::{card}
   Fine-Tuning BERT
   ^^^
   :::{image} am_img10.svg
   :align: center
   :::
   ::::

2. Encoder–Decoder

   ::::{card}
   Pretraining T5
   ^^^
   :::{image} am_img11.svg
   :align: center
   :::
   ::::

   ::::{card}
   Fine-Tuning T5
   ^^^
   :::{image} am_img12.svg
   :align: center
   :::
   ::::

3. Decoder-Only

   ::::{card}
   GPT-2
   ^^^
   :::{image} am_img13.svg
   :align: center
   :::
   ::::

   ::::{card}
   GPT-3
   ^^^
   :::{image} am_img14.svg
   :align: center
   :::
   ::::
