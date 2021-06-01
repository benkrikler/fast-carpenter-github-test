import types

import pandas as pd
import numpy as np

from .tree_wrapper import WrappedTree
from .dataspace import create_aliases, recursive_index


class MaskedUprootTree(object):
    def __init__(self, tree, event_ranger, mask=None):
        if isinstance(tree, MaskedUprootTree):
            self.tree = tree.tree
            self._mask = tree._mask
            return

        self.tree = WrappedTree(tree, event_ranger)
        self.event_ranger = event_ranger

        if mask is None:
            self._mask = None
            return

        self._mask = _normalise_mask(mask, len(self.tree))
        self.pandas = types.SimpleNamespace(df=self._df)

    def _df(self, *args, **kwargs):
        df = self.tree.pandas.df(*args, **kwargs)
        if self._mask is None:
            return df
        masked = mask_df(df, self._mask, self.event_ranger.start_entry)
        return masked

    def unmasked_array(self, *args, **kwargs):
        return self.tree.array(*args, **kwargs)

    def unmasked_arrays(self, *args, **kwargs):
        return self.tree.arrays(*args, **kwargs)

    def __getitem__(self, key):
        return self.tree[key]

    def array(self, *args, **kwargs):
        array = self.tree.array(*args, **kwargs)
        if self._mask is None:
            return array
        return array[self._mask]

    def arrays(self, *args, **kwargs):
        arrays = self.tree.arrays(*args, **kwargs)
        if self._mask is None:
            return arrays
        if isinstance(arrays, dict):
            return {k: v[self._mask] for k, v in arrays.items()}
        if isinstance(arrays, tuple):
            return tuple([v[self._mask] for v in arrays])
        if isinstance(arrays, list):
            return [v[self._mask] for v in arrays]
        if isinstance(arrays, pd.DataFrame):
            return mask_df(arrays, self._mask, self.event_ranger.start_entry)
        if isinstance(arrays, np.ndarray):
            if arrays.ndim == 1:
                return arrays[self._mask]
            if arrays.ndim == 2:
                if arrays.shape[1] == len(self.tree):
                    return arrays[:, self._mask]
            msg = "Unexpected numpy array for mask, shape:%s, mask length: %s"
            raise NotImplementedError(msg % (arrays.shape, len(self)))
        return arrays[self._mask]

    @property
    def mask(self):
        return self._mask

    def apply_mask(self, new_mask):
        if self._mask is None:
            self._mask = _normalise_mask(new_mask, len(self.tree))
        else:
            self._mask = self._mask[new_mask]

    def __len__(self):
        if self._mask is None:
            return len(self.tree)
        return len(self._mask)

    def __contains__(self, element):
        return self.tree.__contains__(element)

    def __getattr__(self, attr):
        return getattr(self.tree, attr)

    def reset_mask(self):
        self._mask = None


def _normalise_mask(mask, tree_length):
    if isinstance(mask, (tuple, list)):
        mask = np.array(mask)
    elif not isinstance(mask, np.ndarray):
        raise RuntimeError("mask is not a numpy array, a list, or a tuple")

    if np.issubdtype(mask.dtype, np.integer):
        return mask
    elif mask.dtype.kind == "b":
        if len(mask) != tree_length:
            raise RuntimeError(
                "boolean mask has a different length to the input tree")
        return np.where(mask)[0]


def mask_df(df, mask, start_event):
    mask = mask + start_event

    # Either of these methods could work on a general df (multiindex, or not) but they
    # have opposite performances, so check if multi-index and choose accordingly
    if isinstance(df.index, pd.MultiIndex):
        broadcast_map = np.isin(df.index.get_level_values("entry"), mask)
        masked = df.iloc[broadcast_map]
    else:
        masked = df.loc[mask]
    return masked


class MaskedTrees(object):
    nonOverloadedFunctions = [
        'pandas', 'array', 'arrays', 'mask', 'unmasked_array',
        'unmasked_arrays', 'apply_mask', 'trees', '_mask', 'event_ranger',
        '_index', '__getitem__', 'reset_mask', 'new_variable', 'keys',
    ]

    def __init__(self, trees, event_ranger, mask=None):
        self.pandas = types.SimpleNamespace(df=self._df)
        self.trees = {}
        self._mask = mask
        # TODO: need to make sure event_ranger act very low level to prevent reads.
        # currently, that is broken and only works for DFs
        # MAskedTrees is probably too late to deal with event ranges
        self.event_ranger = event_ranger
        self._index = {}
        for name, tree in trees.items():
            self.trees[name] = MaskedUprootTree(tree, event_ranger, mask)
            provenance = name.split('/')
            self._index.update(recursive_index(self._index, provenance, self.trees[name]))
        self._index = create_aliases(self._index)

    def __getitem__(self, key):
        return self._index[key]

    def _df(self, *args, **kwargs):
        return self.arrays(*args, **kwargs, outputtype=pd.DataFrame)

    def array(self, *args, **kwargs):
        array = None

        value = self._index[args[0]]
        array = value.array(*args[1:], **kwargs)

        if self._mask is None:
            return array
        return array[self._mask]

    def arrays(self, *args, **kwargs):
        outputtype = kwargs.pop('outputtype', dict)

        arrays = dict()
        names = []
        if args:
            if isinstance(args[0], (list, tuple)):
                names = args[0]
            if isinstance(args[0], (str, bytes)):
                names = [args[0]]
        else:
            names = self.keys()

        for name in names:

            arrays[name] = self.array(name, **kwargs)

        if outputtype is dict:
            return arrays
        if outputtype is list:
            return list(arrays.values())
        if outputtype is tuple:
            return tuple(arrays.values())
        if outputtype is pd.DataFrame:
            return pd.DataFrame.from_dict(arrays)
        if isinstance(outputtype, (types.FunctionType, types.LambdaType)):
            # probe output type
            t = outputtype(1)
            outputtype = type(t)
        if outputtype is np.ndarray:
            return np.array(list(arrays.values()))

        return arrays

    @property
    def mask(self):
        return self._mask

    def apply_mask(self, new_mask):
        for tree in self.trees.values():
            tree.apply_mask(new_mask)
            self._mask = tree._mask

    def __len__(self):
        # if self._mask is None:
        return max([len(e) for e in self.trees.values()])
        # return len(self._mask)

    def __contains__(self, element):
        # TODO: return a masked wrapper instead
        return element in self._index

    def __getattr__(self, attr):
        if attr in MaskedTrees.nonOverloadedFunctions:
            return self.__getattribute__(attr)
        for tree in self.trees.values():
            if tree.__contains__(attr):
                return getattr(tree, attr)
        raise AttributeError('Cannot find attribute "{}"'.format(attr))

    def reset_mask(self):
        self._mask = None
        for tree in self.trees.values():
            tree.reset_mask()

    def reset_cache(self):
        for tree in self.trees.values():
            tree.reset_cache()

    def new_variable(self, name, value):
        if name in self:
            msg = "Trying to overwrite existing variable: '%s'"
            raise ValueError(msg % name)
        if len(value) != len(self):
            msg = "New array %s does not have the right length: %d not %d"
            raise ValueError(msg % (name, len(value), len(self)))

        outputtype = WrappedTree.FakeBranch

        self._index[name] = outputtype(name, value, self.event_ranger)

    def keys(self):
        return self._index.keys()
