
# Vocabulary 
# What each special token is for:
# - [PAD]: fills short sequences in a batch so they all have the same length. 
# - [UNK]: a fallback for unknown input. 
# - [BOS] / [EOS]: mark the start and end of a sequence. 
# - No [MASK]: Llama-style model predicts the next token, no need [MASK] to fill hidden tokens.

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
BASES = list("ACGTN")
VOCAB = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + BASES)}

