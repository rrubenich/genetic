from tokenizers import Regex, Tokenizer, models, normalizers
from .vocabulary import VOCAB


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))

    # Normalizer: the FASTA is soft-masked (repeats in lowercase), and user input
    # may contain IUPAC codes (R, Y, ...). Uppercase first, then map anything that
    # is not ACGTN to N, so the vocab stays closed. There is no Uppercase normalizer.
    tok.normalizer = normalizers.Sequence([
        *(normalizers.Replace(c, c.upper()) for c in "acgtn"),
        normalizers.Replace(Regex("[^ACGTN]"), "N"),
    ])

    return tok
