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

Figure
: :::{image} am_img1.svg
  :align: center
  :::
  :::{image} am_img2.svg
  :align: center
  :::
  :::{image} am_img3.svg
  :align: center
  :::

# Kernel Regression

1. Non-parametric Model: Captures complex, non-linear relationships.
2. Focus on Similarity: Kernel shape and bandwidth control how it adapts
   to local data structure.
3. Kernel Function: Less sensitive to outliers than parametric models.
