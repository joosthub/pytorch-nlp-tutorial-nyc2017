from collections import Counter

import numpy as np
from torch.utils.data import Dataset
import six


class Vocabulary(object):
    """
    An implementation that manages the interface between a token dataset and the
        machine learning algorithm.
    """

    def __init__(self, use_unks=False, unk_token="<UNK>",
                 use_mask=False, mask_token="<MASK>", use_start_end=False,
                 start_token="<START>", end_token="<END>"):
        """
        Args:
            use_unks (bool): The vocabulary will output UNK tokens for out of
                vocabulary items.
                [default=False]
            unk_token (str): The token used for unknown tokens.
                If `use_unks` is True, this will be added to the vocabulary.
                [default='<UNK>']
            use_mask (bool): The vocabulary will reserve the 0th index for a mask token.
                This is used to handle variable lengths in sequence models.
                [default=False]
            mask_token (str): The token used for the mask.
                Note: mostly a placeholder; it's unlikely the token will be seen.
                [default='<MASK>']
            use_start_end (bool): The vocabulary will reserve indices for two tokens
                that represent the start and end of a sequence.
                [default=False]
            start_token: The token used to indicate the start of a sequence.
                If `use_start_end` is True, this will be added to the vocabulary.
                [default='<START>']
            end_token: The token used to indicate the end of a sequence
                 If `use_start_end` is True, this will be added to the vocabulary.
                 [default='<END>']
        """

        self._mapping = {}  # str -> int
        self._flip = {}  # int -> str;
        self._counts = Counter()  # int -> int; count occurrences
        self._forced_unks = set()  # force tokens to unk (e.g. if < 5 occurrences)
        self._i = 0
        self._frozen = False
        self._frequency_threshold = -1

        # mask token for use in masked recurrent networks
        # usually need to be the 0th index
        self.use_mask = use_mask
        self.mask_token = mask_token
        if self.use_mask:
            self.add(self.mask_token)

        # unk token for out of vocabulary tokens
        self.use_unks = use_unks
        self.unk_token = unk_token
        if self.use_unks:
            self.add(self.unk_token)

        # start token for sequence models
        self.use_start_end = use_start_end
        self.start_token = start_token
        self.end_token = end_token
        if self.use_start_end:
            self.add(self.start_token)
            self.add(self.end_token)

    def iterkeys(self):
        for k in self._mapping.keys():
            if k == self.unk_token or k == self.mask_token:
                continue
            else:
                yield k

    def keys(self):
        return list(self.iterkeys())

    def iteritems(self):
        for key, value in self._mapping.items():
            if key == self.unk_token or key == self.mask_token:
                continue
            yield key, value

    def items(self):
        return list(self.iteritems())

    def values(self):
        return [value for _, value in self.iteritems()]

    def __getitem__(self, k):
        if self._frozen:
            if k in self._mapping:
                out_index = self._mapping[k]
            elif self.use_unks:
                out_index = self.unk_index
            else:  # case: frozen, don't want unks, raise exception
                raise VocabularyException("Vocabulary is frozen. " +
                                          "Key '{}' not found.".format(k))
            if out_index in self._forced_unks:
                out_index = self.unk_index
        elif k in self._mapping:  # case: normal
            out_index = self._mapping[k]
            self._counts[out_index] += 1
        else:
            out_index = self._mapping[k] = self._i
            self._i += 1
            self._flip[out_index] = k
            self._counts[out_index] = 1

        return out_index

    def add(self, k):
        return self.__getitem__(k)

    def add_many(self, x):
        return [self.add(k) for k in x]

    def lookup(self, i):
        try:
            return self._flip[i]
        except KeyError:
            raise VocabularyException("Key {} not in Vocabulary".format(i))

    def lookup_many(self, x):
        for k in x:
            yield self.lookup(k)

    def map(self, sequence, include_start_end=False):
        if include_start_end:
            yield self.start_index

        for item in sequence:
            yield self[item]

        if include_start_end:
            yield self.end_index

    def freeze(self, use_unks=False, frequency_cutoff=-1):
        self.use_unks = use_unks
        self._frequency_cutoff = frequency_cutoff

        if use_unks and self.unk_token not in self:
            self.add(self.unk_token)

        if self._frequency_cutoff > 0:
            for token, count in self._counts.items():
                if count < self._frequency_cutoff:
                    self._forced_unks.add(token)

        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def get_counts(self):
        return {self._flip[i]: count for i, count in self._counts.items()}

    def get_count(self, token=None, index=None):
        if token is None and index is None:
            return None
        elif token is not None and index is not None:
            print("Cannot do two things at once; choose one")
        elif token is not None:
            return self._counts[self[token]]
        elif index is not None:
            return self._counts[index]
        else:
            raise Exception("impossible condition")

    @property
    def unk_index(self):
        if self.unk_token not in self:
            return None
        return self._mapping[self.unk_token]

    @property
    def mask_index(self):
        if self.mask_token not in self:
            return None
        return self._mapping[self.mask_token]

    @property
    def start_index(self):
        if self.start_token not in self:
            return None
        return self._mapping[self.start_token]

    @property
    def end_index(self):
        if self.end_token not in self:
            return None
        return self._mapping[self.end_token]

    def __contains__(self, k):
        return k in self._mapping

    def __len__(self):
        return len(self._mapping)

    def __repr__(self):
        return "<Vocabulary(size={},frozen={})>".format(len(self), self._frozen)


    def get_serializable_contents(self):
        """
        Creats a dict containing the necessary information to recreate this instance
        """
        config = {"_mapping": self._mapping,
                  "_flip": self._flip,
                  "_frozen": self._frozen,
                  "_i": self._i,
                  "_counts": list(self._counts.items()),
                  "_frequency_threshold": self._frequency_threshold,
                  "use_unks": self.use_unks,
                  "unk_token": self.unk_token,
                  "use_mask": self.use_mask,
                  "mask_token": self.mask_token,
                  "use_start_end": self.use_start_end,
                  "start_token": self.start_token,
                  "end_token": self.end_token}
        return config

    @classmethod
    def deserialize_from_contents(cls, content):
        """
        Recreate a Vocabulary instance; expect same dict as output in `serialize`
        """
        try:
            _mapping = content.pop("_mapping")
            _flip = content.pop("_flip")
            _i = content.pop("_i")
            _frozen = content.pop("_frozen")
            _counts = content.pop("_counts")
            _frequency_threshold = content.pop("_frequency_threshold")
        except KeyError:
            raise Exception("unable to deserialize vocabulary")
        if isinstance(list(_flip.keys())[0], six.string_types):
            _flip = {int(k): v for k, v in _flip.items()}
        out = cls(**content)
        out._mapping = _mapping
        out._flip = _flip
        out._i = _i
        out._counts = Counter(dict(_counts))
        out._frequency_threshold = _frequency_threshold

        if _frozen:
            out.freeze(out.use_unks)

        return out

