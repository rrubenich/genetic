from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from .vocabulary import SPECIAL_TOKENS, VOCAB


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    # Mark them as special (they keep their vocab ids), so decode() can skip them
    # and the normalizer never touches them.
    tok.add_special_tokens(SPECIAL_TOKENS)

    # Normalizer: the FASTA is soft-masked (repeats in lowercase), and user input
    # may contain IUPAC codes (R, Y, ...). Uppercase first, then map anything that
    # is not ACGTN to N, so the vocab stays closed. There is no Uppercase normalizer.
    tok.normalizer = normalizers.Sequence([
        *(normalizers.Replace(c, c.upper()) for c in "acgtn"),
        normalizers.Replace(Regex("[^ACGTN]"), "N"),
    ])

    # Pre-tokenizer: WordLevel looks up whole "words", so split every character
    # into its own piece. One base = one token, so a SNV changes exactly one token.
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")

    # Post-processor: wrap every sequence as [BOS] ... [EOS] automatically.
    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", VOCAB["[BOS]"]), ("[EOS]", VOCAB["[EOS]"])],
    )

    # Decoder: join tokens with no separator, so ids decode back to "ACGT".
    tok.decoder = decoders.Fuse()

    return tok
