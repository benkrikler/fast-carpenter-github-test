import pandas as pd
import copy
import numpy as np
import pytest
import fast_carpenter.summary.binned_dataframe as bdf
from . import dummy_binning_descriptions as binning
from ..conftest import FakeBEEvent


@pytest.fixture
def config_1():
    return dict(binning=[binning.bins_met_px,
                         binning.bins_py],
                weights=binning.weight_list)


@pytest.fixture
def config_2():
    return dict(binning=[binning.bins_py,
                         binning.bins_met_px,
                         binning.bins_nmuon],
                weights=binning.weight_dict)


@pytest.fixture
def config_3():
    return dict(binning=[binning.bins_electron_pT],
                weights=binning.weight_dict)


@pytest.fixture
def binned_df_1(config_1):
    return make_binned_df_1(config_1)


def make_binned_df_1(config_1):
    return bdf.BinnedDataframe("binned_df_1", out_dir="somewhere", **config_1)


def test_BinnedDataframe(binned_df_1, tmpdir):
    assert binned_df_1.name == "binned_df_1"
    assert len(binned_df_1._binnings) == 2
    assert binned_df_1._bin_dims[0] == "MET_px"
    # length of bin lists = nbins plus 2 for +/-inf
    assert len(binned_df_1._binnings[0]) == 29 + 2
    # length of bin lists = len(edges) - 1 plus 2 for +/-inf
    assert len(binned_df_1._binnings[1]) == 3 - 1 + 2
    assert len(binned_df_1._weights) == 1


@pytest.fixture
def binned_df_2(tmpdir, config_2):
    return bdf.BinnedDataframe("binned_df_2", out_dir="somewhere", **config_2)


@pytest.fixture
def binned_df_2_weightData(tmpdir, config_2):
    return bdf.BinnedDataframe("binned_df_2_weightData", out_dir="somewhere",
                               weight_data=True, **config_2)


def test_BinnedDataframe_run_mc(binned_df_1, tmpdir, infile):
    chunk = FakeBEEvent(infile, "mc")
    collector = binned_df_1.collector()

    binned_df_1.event(chunk)
    dataset_readers_list = (("test_dataset", (binned_df_1,)),)
    results = collector._prepare_output(dataset_readers_list)

    totals = results.sum()
    # Based on: events->Draw("Jet_Py", "", "goff")
    assert totals["n"] == 4616

    # Based on:
    # events->Draw("EventWeight * (Jet_Py/Jet_Py)>>htemp", "", "goff")
    # htemp->GetMean() * htemp->GetEntries()
    assert totals["EventWeight:sumw"] == pytest.approx(231.91339)

    coll_results = collector.collect(dataset_readers_list, writeFiles=False)

    totals = coll_results.sum()
    # Based on: events->Draw("Jet_Py", "", "goff")
    assert totals["n"] == 4616

    # Based on:
    # events->Draw("EventWeight * (Jet_Py/Jet_Py)>>htemp", "", "goff")
    # htemp->GetMean() * htemp->GetEntries()
    assert totals["EventWeight:sumw"] == pytest.approx(231.91339)


def test_BinnedDataframe_run_data(binned_df_2, tmpdir, infile):
    chunk = FakeBEEvent(infile, "data")
    binned_df_2.event(chunk)

    collector = binned_df_2.collector()
    dataset_readers_list = (("test_dataset", (binned_df_2,)),)
    results = collector._prepare_output(dataset_readers_list)

    totals = results.sum()
    # Based on: events->Draw("Jet_Py", "", "goff")
    assert totals["n"] == 4616


def test_BinnedDataframe_run_dataWeighted(binned_df_2_weightData, tmpdir, infile):
    chunk = FakeBEEvent(infile, "data")
    binned_df_2_weightData.event(chunk)

    collector = binned_df_2_weightData.collector()
    dataset_readers_list = (("test_dataset", (binned_df_2_weightData,)),)
    results = collector._prepare_output(dataset_readers_list)

    totals = results.sum()
    # Based on: events->Draw("Jet_Py", "", "goff")
    assert totals["n"] == 4616
    assert totals["weighted:sumw"] == pytest.approx(231.91339)


def test_BinnedDataframe_run_twice(binned_df_1, tmpdir, infile):
    chunk = FakeBEEvent(infile, "mc")
    collector = binned_df_1.collector()

    binned_df_1.event(chunk)
    binned_df_1.event(chunk)

    dataset_readers_list = (("test_dataset", (binned_df_1,)),)
    results = collector._prepare_output(dataset_readers_list)

    totals = results.sum()
    # Based on: events->Draw("Jet_Py", "", "goff")
    assert totals["n"] == 4616 * 2

    # Based on:
    # events->Draw("EventWeight * (Jet_Py/Jet_Py)>>htemp", "", "goff")
    # htemp->GetMean() * htemp->GetEntries()
    assert totals["EventWeight:sumw"] == pytest.approx(231.91339 * 2)


@pytest.fixture
def run_twice_data_mc(config_1, infile, observed):
    chunk_mc = FakeBEEvent(infile, "mc")
    chunk_data = FakeBEEvent(infile, "data")
    config_1["observed"] = observed

    binned_dfs = [make_binned_df_1(config_1) for _ in range(4)]
    binned_dfs[0].event(chunk_mc)
    binned_dfs[1].event(chunk_mc)
    binned_dfs[2].event(chunk_data)
    binned_dfs[3].event(chunk_data)

    return binned_dfs[0], (("test_mc", (binned_dfs[0], binned_dfs[1])),
                           ("test_data", (binned_dfs[2], binned_dfs[3])))


@pytest.mark.skipif(int(pd.__version__.split(".")[0]) < 1, reason="requires Pandas 1.0 or higher")
@pytest.mark.parametrize("dataset_col", [True, False])
@pytest.mark.parametrize("pad_missing", [True, False])
@pytest.mark.parametrize("observed", [True, False])
def test_binneddataframe_run_twice_data_mc(run_twice_data_mc, dataset_col, pad_missing, observed):
    binned_df_1, dataset_readers_list = run_twice_data_mc
    binned_df_1._pad_missing = pad_missing
    binned_df_1._dataset_col = dataset_col
    collector = binned_df_1.collector()
    results = collector._prepare_output(dataset_readers_list)

    assert results.index.nlevels == 2 + int(dataset_col)
    if pad_missing or not observed:
        length = (4 * 31)
    elif observed:
        length = 111

    length *= 1 + int(dataset_col)
    assert len(results) == length

    totals = results.sum()
    # Based on: events->Draw("Jet_Py", "", "goff")
    assert totals["n"] == 4616 * 4

    # Based on:
    # events->Draw("EventWeight * (Jet_Py/Jet_Py)>>htemp", "", "goff")
    # htemp->GetMean() * htemp->GetEntries()
    assert totals["EventWeight:sumw"] == pytest.approx(231.91339 * 2)


@pytest.fixture
def binned_df_3(tmpdir, config_3):
    return bdf.BinnedDataframe("binned_df_3", out_dir="somewhere", **config_3)


def test_BinnedDataframe_numexpr(binned_df_3, tmpdir):
    assert binned_df_3.name == "binned_df_3"
    assert len(binned_df_3._binnings) == 1
    assert binned_df_3._bin_dims[0] == "sqrt(Electron_Px**2 + Electron_Py**2)"
    assert binned_df_3._binnings[0][1].left == 0.0
    assert binned_df_3._binnings[0][-1].left == 200
    # length of bin lists = nbins plus 2 for +/-inf
    assert len(binned_df_3._binnings[0]) == 2 * 10 + 2
    assert len(binned_df_3._weights) == 1


def test_BinnedDataframe_user_var_run(config_3, tmpdir, full_wrapped_tree):
    config_4 = copy.deepcopy(config_3)
    config_4["binning"][0]["in"] = "Electron_Pt"
    binned_df_4 = bdf.BinnedDataframe("binned_df_4", out_dir="somewhere", **config_4)

    chunk = FakeBEEvent(full_wrapped_tree, "mc")
    px, py = chunk.tree.arrays(["Electron_Px", "Electron_Py"], outputtype=tuple)
    pt = np.hypot(px, py)
    chunk.tree.new_variable("Electron_Pt", pt)
    collector = binned_df_4.collector()

    binned_df_4.event(chunk)
    dataset_readers_list = (("test_dataset", (binned_df_4,)),)
    results = collector._prepare_output(dataset_readers_list)

    bin_centers = pd.IntervalIndex(results.index.get_level_values('electron_pT')).mid
    mean = np.sum((bin_centers[1:-1] * results['n'][1:-1]) / results['n'][1:-1].sum())
    assert mean == pytest.approx(44.32584)


def test_BinnedDataframe_numexpr_run_mc(binned_df_3, tmpdir, infile):
    chunk = FakeBEEvent(infile, "mc")
    collector = binned_df_3.collector()

    binned_df_3.event(chunk)
    dataset_readers_list = (("test_dataset", (binned_df_3,)),)
    results = collector._prepare_output(dataset_readers_list)

    bin_centers = pd.IntervalIndex(results.index.get_level_values('electron_pT')).mid
    mean = np.sum((bin_centers[1:-1] * results['n'][1:-1]) / results['n'][1:-1].sum())
    assert mean == pytest.approx(44.32584)


def test_BinnedDataframe_numexpr_similar_branch(binned_df_3, tmpdir, full_wrapped_tree):
    chunk = FakeBEEvent(full_wrapped_tree, "mc")
    new_var = 2 * chunk.tree.array("Electron_Px")
    chunk.tree.new_variable("tron_Px", new_var)
    collector = binned_df_3.collector()

    binned_df_3.event(chunk)
    dataset_readers_list = (("test_dataset", (binned_df_3,)),)
    results = collector._prepare_output(dataset_readers_list)

    bin_centers = pd.IntervalIndex(results.index.get_level_values('electron_pT')).mid
    mean = np.sum((bin_centers[1:-1] * results['n'][1:-1]) / results['n'][1:-1].sum())
    assert mean == pytest.approx(44.32584)


def test_explode():
    df = pd.DataFrame({'A': [[1, 2, 3], [9], [], [3, 4]], 'B': 1})
    exploded = bdf.explode(df)
    assert len(exploded) == 6
    assert all(df.B == 1)
    assert not df.isnull().any().any()

    df["C"] = df.A.copy()
    exploded = bdf.explode(df)
    assert len(exploded) == 6
    assert all(df.B == 1)
    assert all(df.A == df.C)

    df["D"] = [[1], [3], [4, 5], []]
    with pytest.raises(ValueError) as e:
        exploded = bdf.explode(df)
    assert "different jaggedness" in str(e)

    df2 = pd.DataFrame({'A': [[np.arange(i + 1) + j for i in range((j % 2) + 1)] for j in range(4)],
                        'B': np.arange(4)[::-1],
                        })
    exploded = bdf.explode(df2)
    assert len(exploded) == 8

    df = pd.DataFrame({'number': [1, 8, 3], 'string': ['one', 'eight', 'three']})
    exploded = bdf.explode(df)
    assert len(exploded) == 3

    df["list"] = [list(range(i)) for i in df.number]
    exploded = bdf.explode(df)
    assert len(exploded) == 1 + 8 + 3
    assert np.array_equal(exploded.list, [0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2])

    exploded = bdf.explode(pd.DataFrame(columns=["one", "two", "3"]))
    assert exploded.empty is True

    df.index = np.arange(len(df)) + 100
    exploded = bdf.explode(df)
    assert len(exploded) == 1 + 8 + 3
    assert np.array_equal(exploded.list, [0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2])


def test_explode_empty():
    df = pd.DataFrame({'A': pd.Series([], dtype=object)})
    exploded = bdf.explode(df)
    assert len(exploded) == 0


def test_densify_dataframe_integers():
    index = [("one", 1), ("one", 3), ("two", 2), ("three", 1), ("three", 2)]
    index = pd.MultiIndex.from_tuples(index, names=["foo", "bar"])
    df = pd.DataFrame({'A': np.arange(5, 0, -1), 'B': list("abcde")}, index=index)
    out_df = bdf.densify_dataframe(df, {"bar": list(range(1, 4))})

    assert len(out_df) == 9
    assert out_df.loc[("one", 2)].isna().all()
    assert out_df.loc[("two", 1)].isna().all()
    assert out_df.loc[("two", 3)].isna().all()
    assert out_df.loc[("three", 3)].isna().all()


def test_densify_dataframe_intervals():
    index = [("one", 1), ("one", 3), ("two", 2), ("three", 1), ("three", 2)]
    index = [(a, pd.Interval(b, b + 1)) for a, b in index]
    index = pd.MultiIndex.from_tuples(index, names=["foo", "bar"])
    df = pd.DataFrame({'A': np.arange(5, 0, -1), 'B': list("abcde")}, index=index)
    out_df = bdf.densify_dataframe(df, {"bar": pd.IntervalIndex.from_breaks(range(1, 5))})
    print(df)
    print(out_df)

    assert len(out_df) == 9
    assert out_df.loc[("one", pd.Interval(2, 3))].isna().all()
    assert out_df.loc[("two", pd.Interval(1, 2))].isna().all()
    assert out_df.loc[("two", pd.Interval(3, 4))].isna().all()
    assert out_df.loc[("three", pd.Interval(3, 4))].isna().all()
