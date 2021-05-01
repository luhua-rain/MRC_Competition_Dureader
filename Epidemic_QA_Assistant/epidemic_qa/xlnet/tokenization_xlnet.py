""" Tokenization classes for XLNet model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
from shutil import copyfile
import collections

import unicodedata
import six

# from .tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'spiece.model'}

SPIECE_UNDERLINE = u'▁'

# Segments (not really needed)
SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class XLNetTokenizer():  # PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES

    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file='spiece.model', max_len=512,
                 do_lower_case=False, remove_space=True, keep_accents=False,
                 bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="<sep>",
                 pad_token="<pad>", cls_token="<cls>", mask_token="<mask>",
                 additional_special_tokens=["<eop>", "<eod>"]):
        super(XLNetTokenizer, self).__init__()
        import sentencepiece as spm

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = './data/spiece.model'

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load('./data/spiece.model')

        self.vocab = collections.OrderedDict()
        for n, i in enumerate(range(32000)):
            self.vocab[self._convert_id_to_token(i)] = n
            # print(.a._convert_id_to_token(i))

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if six.PY2 and isinstance(outputs, str):
            outputs = outputs.decode('utf-8')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, return_unicode=True, sample=False):
        """ Tokenize a string.
            return_unicode is used only for py2
        """
        text = self.preprocess_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        new_pieces = [piece.replace(SPIECE_UNDERLINE, '') for piece in new_pieces if
                      len(piece.replace(SPIECE_UNDERLINE, '')) > 0]
        return new_pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index, return_unicode=True):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        if six.PY2 and return_unicode and isinstance(token, str):
            token = token.decode('utf-8')
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def add_special_tokens_single_sentence(self, token_ids):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        An XLNet sequence pair has the following format: A [SEP] B [SEP][CLS]
        """
        sep = [self._convert_token_to_id(self.sep_token)]
        cls = [self._convert_token_to_id(self.cls_token)]
        return token_ids + sep + cls

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence for sequence classification tasks.
        An XLNet sequence has the following format: X [SEP][CLS]
        """
        sep = [self._convert_token_to_id(self.sep_token)]
        cls = [self._convert_token_to_id(self.cls_token)]
        return token_ids_0 + sep + token_ids_1 + sep + cls

    def save_vocabulary(self):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        out_vocab_file = 'xlnet_vocab.txt'
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    def load_xlnet_vocab(self):
        self.vocab = collections.OrderedDict()
        for n, i in enumerate(range(32000)):
            self.vocab[self._convert_id_to_token(i)] = n
            # print(.a._convert_id_to_token(i))
        return self.vocab

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token == '¿':
                token = '<mask>'

            token = token.lower()
            ids.append(self.vocab.get(token, 0))

        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self._convert_id_to_token(i))
        return tokens


if __name__ == "__main__":
    a = XLNetTokenizer()
    vocab = a.load_xlnet_vocab()
    print(a._convert_token_to_id('<sep>'))
    print(vocab)
