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
