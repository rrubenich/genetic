from tokenizers import Tokenizer, models
from .vocabulary import VOCAB

tok = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))