# Tensor conventions

The components use a batch-first layout. `position` is a token's index in a
sequence. `pos_q` and `pos_k` distinguish query and key positions during attention.

| Tensor | Shape |
| --- | --- |
| Token IDs | `[batch, position]` |
| Token embeddings | `[batch, position, d_model]` |
| Position embeddings | `[position, d_model]` (broadcast across the batch) |
| Layer norm input/output | `[batch, position, d_model]` |
| Q, K, V | `[batch, position, n_heads, d_head]` |
| Attention scores and pattern | `[batch, n_heads, pos_q, pos_k]` |
| Z (weighted values) | `[batch, pos_q, n_heads, d_head]` |
| Attention output | `[batch, pos_q, d_model]` |

Each head has its own projection matrices. Putting the token axis before the
head axis does not change that independence.

Q and K are contracted over `d_head` to produce scores. Scores are scaled by
`sqrt(d_head)`, masked to exclude future key positions, then normalized with a
softmax over `pos_k`. A query can attend to itself and earlier positions.

The pattern weights V over `pos_k` to produce Z. The output projection contracts
both `n_heads` and `d_head` to return to `d_model`.

The PyTorch comparison test uses `d_model = n_heads * d_head`, as required by
`nn.MultiheadAttention`. The default configuration follows this relationship.
