import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from .config import MODEL_DIR
from .vocabulary import SPECIAL_TOKENS, VOCAB

# model/runs/tokenizer, no matter where this is run from
DEFAULT_OUT_DIR = MODEL_DIR / "runs" / "tokenizer"


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


def build_hf_tokenizer() -> PreTrainedTokenizerFast:
    # The transformers wrapper adds padding, batching, save_pretrained and
    # AutoTokenizer support. It needs to be told which token plays which role.
    return PreTrainedTokenizerFast(
        tokenizer_object=build_tokenizer(),
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


if __name__ == "__main__":
    # Usage: python -m genetic_model.tokenizer [out_dir]   (default: runs/tokenizer)
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    build_hf_tokenizer().save_pretrained(out_dir)
    print(f"Saved tokenizer to {out_dir}")
