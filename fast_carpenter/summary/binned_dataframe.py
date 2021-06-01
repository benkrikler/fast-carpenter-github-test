"""
Summarize the data by producing binned and possibly weighted counts of the data.
"""
import os
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from . import binning_config as cfg


class Collector():
    valid_ext = {'xlsx': 'excel', 'h5': 'hdf', 'msg': 'msgpack', 'dta': 'stata', 'pkl': 'pickle', 'p': 'pickle'}

    def __init__(self, filename, dataset_col, binnings, file_format):
        self.filename = filename
        self.dataset_col = dataset_col
        self.binnings = binnings
        self.file_format = file_format

    def collect(self, dataset_readers_list, doReturn=True, writeFiles=True):
        if len(dataset_readers_list) == 0:
            if doReturn:
                return pd.DataFrame()
            else:
                return None

        output = self._prepare_output(dataset_readers_list)

        if writeFiles:
            for file_dict in self.file_format:
                file_ext = file_dict.pop('extension', None)
                save_func = file_ext.split('.')[1]
                if save_func in Collector.valid_ext:
                    save_func = Collector.valid_ext[save_func]
                try:
                    getattr(output, "to_%s" % save_func)(self.filename + file_ext, **file_dict)
                except AttributeError as err:
                    print("Incorrect file format: %s" % err)
                except TypeError as err:
                    print("Incorrect args: %s" % err)

        if doReturn:
            return output

    def _prepare_output(self, dataset_readers_list):
        return combined_dataframes(dataset_readers_list,
                                   self.dataset_col,
                                   binnings=self.binnings)


def combined_dataframes(dataset_readers_list, dataset_col, binnings=None):
    dataset_readers_list = [(d, [r.contents for r in readers])
                            for d, readers in dataset_readers_list if readers]
    if not dataset_readers_list:
        return None

    if dataset_col:
        output = _merge_dataframes(dataset_readers_list)
    else:
        output = _sum_dataframes(dataset_readers_list)
    if binnings:
        output = densify_dataframe(output, binnings)
    return output


def _merge_dataframes(dataset_readers_list):
    all_dfs = []
    keys = []
    for dataset, readers in dataset_readers_list:
        dataset_df = readers[0]
        for df in readers[1:]:
            if df is None or df.empty:
                continue
            dataset_df = dataset_df.add(df, fill_value=0.)
        if dataset_df is None or dataset_df.empty:
            continue
        all_dfs.append(dataset_df)
        keys.append(dataset)
    if all_dfs:
        final_df = pd.concat(all_dfs, keys=keys, names=['dataset'], sort=True)
    else:
        final_df = pd.DataFrame()

    return final_df


def _sum_dataframes(dataset_readers_list):
    final_df = None
    for _, readers in dataset_readers_list:
        for df in readers:
            if df is None:
                continue
            if final_df is None:
                final_df = df
                continue
            final_df = final_df.add(df, fill_value=0.)
    return final_df


def densify_dataframe(in_df, binnings):
    in_index = in_df.index
    index_values = []
    for dim in in_index.names:
        bins = binnings.get(dim, None)
        if bins is None:
            index_values.append(in_index.unique(dim))
            continue
        index_values.append(bins)
    out_index = pd.MultiIndex.from_product(index_values, names=in_index.names)
    out_df = in_df.reindex(index=out_index, copy=False)
    return out_df


class BinnedDataframe():
    """Produces a binned dataframe (a multi-dimensional histogram).

    def __init__(self, name, out_dir, binning, weights=None, dataset_col=False):

    Parameters:
      binning (list[dict]): A list of dictionaries describing the variables to
        bin on, and how they should be binned.  Each of these dictionaries can
        contain the following:

        +-----------+----------------+--------------------------------------------------------------+
        | Parameter | Default        | Description                                                  |
        +===========+================+==============================================================+
        | ``in``    |                | The name of the attribute on the event to use.               |
        +-----------+----------------+--------------------------------------------------------------+
        | ``out``   | same as ``in`` | The name of the column to be filled in the output dataframe. |
        +-----------+----------------+--------------------------------------------------------------+
        | ``bins``  | ``None``       | | Must be either ``None`` or a dictionary.  If a dictionary, |
        |           |                |   it must contain one of the follow sets of                  |
        |           |                | | key-value pairs:                                           |
        |           |                | |   1. ``nbins``, ``low``, ``high``: which are used to       |
        |           |                |     produce a list of bin edges equivalent to:               |
        |           |                | |      ``numpy.linspace(low, high, nbins + 1)``              |
        |           |                | |   2. ``edges``: which is treated as the list of bin        |
        |           |                |     edges directly.                                          |
        |           |                | | If set to ``None``, then the input variable is assumed     |
        |           |                |   to already be categorical (ie. binned or discrete)         |
        +-----------+----------------+--------------------------------------------------------------+

      weights (str or list[str], dict[str, str]): How to weight events in the
        output table.  Must be either a single variable, a list of
        variables, or a dictionary where the values are variables in the data and
        keys are the column names that these weights should be called in the
        output tables.
      file_format (str or list[str], dict[str, str]): determines the file format to
        use to save the binned dataframe to disk.  Should be either a) a string with
        the file format, b) a dict containing the keyword `extension` to give the file
        format and then all other keyword-argument pairs are passed on to the
        corresponding pandas function, or c) a list of values matching a) or b).
      dataset_col (bool): adds an extra binning column with the name for each dataset.
      pad_missing (bool): If ``False``, any bins that don't contain data are
        excluded from the stored dataframe.  Leaving this ``False`` can save
        some disk-space and improve processing time, particularly if the bins are
        only very sparsely filled.
      observed (bool): If ``False`` bins in the dataframe will only be filled
        if their are datapoints contained within them.  Otherwise, depending on
        the binning specification for each dimension, all bins for that
        dimension will be present.  Use `pad_missing: true` to force all bins
        to be present.

    Other Parameters:
      name (str):  The name of this stage (handled automatically by fast-flow)
      out_dir (str):  Where to put the summary table (handled automatically by
          fast-flow)

    Raises:
      BadBinnedDataframeConfig: If there is an issue with the binning description.

    """

    def __init__(self, name, out_dir, binning, weights=None, dataset_col=True,
                 pad_missing=False, file_format=None, observed=False, weight_data=False):
        self.name = name
        self.out_dir = out_dir
        ins, outs, binnings = cfg.create_binning_list(self.name, binning)
        self._bin_dims = ins
        self.potential_inputs = set(sum((re.findall(r"\w+", dim) for dim in self._bin_dims), []))
        self._out_bin_dims = outs
        self._binnings = binnings
        self._dataset_col = dataset_col
        self._weights = cfg.create_weights(self.name, weights)
        self._pad_missing = pad_missing
        self._file_format = cfg.create_file_format(self.name, file_format)
        self._observed = observed
        self.contents = None
        self.weight_data = weight_data

    def collector(self):
        outfilename = "tbl_"
        if self._dataset_col:
            outfilename += "dataset."
        outfilename += ".".join(self._out_bin_dims)
        outfilename += "--" + self.name
        outfilename = os.path.join(self.out_dir, outfilename)
        binnings = None
        if self._pad_missing:
            binnings = dict(zip(self._out_bin_dims, self._binnings))
        return Collector(outfilename, self._dataset_col, binnings=binnings, file_format=self._file_format)

    def event(self, chunk):
        all_keys = list(chunk.tree.keys())
        if isinstance(all_keys[0], bytes):
            all_keys = [key.decode() for key in all_keys]

        all_inputs = [key for key in all_keys if key in self.potential_inputs]
        if chunk.config.dataset.eventtype == "mc" or self.weight_data:
            weights = list(self._weights.values())
            all_inputs += weights
        else:
            weights = None

        data = chunk.tree.pandas.df(all_inputs, flatten=False)
        data = explode(data)
        if data is None or data.empty:
            return True

        binned_values = _bin_values(data, dimensions=self._bin_dims,
                                    binnings=self._binnings,
                                    weights=weights,
                                    out_weights=self._weights.keys(),
                                    out_dimensions=self._out_bin_dims,
                                    observed=self._observed)
        if self.contents is None:
            self.contents = binned_values
        else:
            self.contents = self.contents.add(binned_values, fill_value=0)

        return True

    def merge(self, rhs):
        if rhs.contents is None or len(rhs.contents) == 0:
            return
        if self.contents is None:
            self.contents = rhs.contents
            return
        self.contents = self.contents.add(rhs.contents, fill_value=0)


count_label = "n"
weight_labels = ("sumw", "sumw2")


def _make_column_labels(weights):
    labels = [weight + ":" + label for label in weight_labels for weight in weights]
    return [count_label] + labels


def _bin_values(data, dimensions, binnings, weights, out_dimensions=None, out_weights=None, observed=True):
    if not out_dimensions:
        out_dimensions = dimensions
    if not out_weights:
        out_weights = weights

    final_bin_dims = []
    for dimension, binning in zip(dimensions, binnings):
        if binning is None:
            final_bin_dims.append(dimension)
            continue
        out_dimension = dimension + "_bins"
        data[out_dimension] = pd.cut(data.eval(dimension, engine='numexpr'), binning, right=False)
        final_bin_dims.append(out_dimension)

    if weights:
        weight_sq_dims = [w + "_squared" for w in weights]
        data[weight_sq_dims] = data[weights] ** 2

    bins = data.groupby(final_bin_dims, observed=observed)
    counts = bins[data.columns[0]].count()

    if weights:
        sums = bins[weights].sum()
        sum_sqs = bins[weight_sq_dims].sum()
        histogram = pd.concat([counts, sums, sum_sqs], axis="columns")
        histogram.columns = _make_column_labels(out_weights)
    else:
        histogram = counts.to_frame(count_label)

    histogram.index.set_names(out_dimensions, inplace=True)
    return histogram


_explodable_types = (tuple, list, np.ndarray)


def explode(df):
    """
    Based on this answer:
    https://stackoverflow.com/questions/12680754/split-explode-pandas\
    -dataframe-string-entry-to-separate-rows/40449726#40449726
    """
    if df is None or df.empty:
        return df

    # get the list columns
    lst_cols = [col for col, dtype in df.dtypes.items() if is_object_dtype(dtype)]
    # remove empty columns
    lst_cols = [col for col in lst_cols if not df[col].empty]
    # Be more specific about which objects are ok
    lst_cols = [col for col in lst_cols if isinstance(df[col].iloc[0], _explodable_types)]
    if not lst_cols:
        return df

    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # check all lists have same length
    lens = pd.DataFrame({col: df[col].str.len() for col in lst_cols})
    different_length = (lens.nunique(axis=1) > 1).any()
    if different_length:
        raise ValueError("Cannot bin multiple arrays with different jaggedness")
    lens = lens[lst_cols[0]]

    # create "exploded" DF
    flattened = {col: df.loc[lens > 0, col].values for col in lst_cols}
    flattened = {col: sum(map(list, vals), []) for col, vals in flattened.items()}
    res = pd.DataFrame({col: np.repeat(df[col].values, lens) for col in idx_cols})
    res = res.assign(**flattened)

    # Check that rows are fully "exploded"
    return explode(res)
