from __future__ import division
import pytest
import numpy as np
import pandas as pd
import fast_carpenter.masked_tree as m_tree


@pytest.fixture
def tree_no_mask(single_tree_input, full_event_range):
    return m_tree.MaskedTrees(single_tree_input, event_ranger=full_event_range)


@pytest.fixture
def tree_w_mask_bool(single_tree_input, full_event_range):
    mask = np.ones(full_event_range.entries_in_block, dtype=bool)
    mask[::2] = False
    return m_tree.MaskedTrees(single_tree_input, event_ranger=full_event_range, mask=mask)


@pytest.fixture
def tree_w_mask_int(single_tree_input, event_range):
    mask = np.ones(event_range.entries_in_block, dtype=bool)
    mask[::2] = False
    mask = np.where(mask)[0]
    return m_tree.MaskedTrees(single_tree_input, event_ranger=event_range, mask=mask)


@pytest.fixture
def multitree_no_mask(multiple_trees_input, full_event_range_multitree):
    return m_tree.MaskedTrees(multiple_trees_input, event_ranger=full_event_range_multitree)


@pytest.fixture
def multitree_w_mask_bool(multiple_trees_input, full_event_range_multitree):
    mask = np.ones(full_event_range_multitree.entries_in_block, dtype=bool)
    mask[::2] = False
    return m_tree.MaskedTrees(multiple_trees_input, event_ranger=full_event_range_multitree, mask=mask)


@pytest.fixture
def multitree_w_mask_int(multiple_trees_input, full_event_range_multitree):
    mask = np.ones(full_event_range_multitree.entries_in_block, dtype=bool)
    mask[::2] = False
    mask = np.where(mask)[0]
    return m_tree.MaskedTrees(multiple_trees_input, event_ranger=full_event_range_multitree, mask=mask)

# ####################################
# ######## Single tree tests #########
# ####################################


def test_no_mask(tree_no_mask, infile):
    assert len(tree_no_mask) == len(infile)
    assert tree_no_mask.mask is None
    compare_df = infile.pandas.df("EventWeight")
    compare_df.columns = ["events." + str(col) for col in compare_df.columns]
    assert np.all(tree_no_mask.pandas.df("events.EventWeight") == compare_df)


def test_w_mask_bool(tree_w_mask_bool, infile):
    initial_length = tree_w_mask_bool.event_ranger.entries_in_block
    assert len(tree_w_mask_bool) == int(initial_length / 2)

    df = tree_w_mask_bool.pandas.df("events.NMuon")
    assert len(df) == int(initial_length / 2)

    new_mask = np.ones(len(tree_w_mask_bool), dtype=bool)
    new_mask[::2] = False
    tree_w_mask_bool.apply_mask(new_mask)
    assert len(tree_w_mask_bool) == int(initial_length / 4)


def test_w_mask_int(tree_w_mask_int, infile):
    initial_length = tree_w_mask_int.event_ranger.entries_in_block
    assert len(tree_w_mask_int) == int(initial_length / 2)

    tree_w_mask_int.apply_mask(np.arange(0, len(tree_w_mask_int), 2))
    assert len(tree_w_mask_int) == int(initial_length / 4)

    df = tree_w_mask_int.pandas.df("events.Muon_Px")
    assert len(df.index.unique(0)) == int(initial_length / 4)


def test_array(tree_w_mask_int, infile):
    initial_length = tree_w_mask_int.event_ranger.entries_in_block
    assert len(tree_w_mask_int) == int(initial_length / 2)

    tree_w_mask_int.apply_mask(np.arange(0, len(tree_w_mask_int), 2))
    assert len(tree_w_mask_int) == int(initial_length / 4)

    array = tree_w_mask_int.array("events.Muon_Px")
    assert len(array) == int(initial_length / 4)


def test_arrays(tree_w_mask_int, infile):
    initial_length = tree_w_mask_int.event_ranger.entries_in_block
    assert len(tree_w_mask_int) == int(initial_length / 2)
    tree_w_mask_int.apply_mask(np.arange(0, len(tree_w_mask_int), 2))
    assert len(tree_w_mask_int) == int(initial_length / 4)

    arrays = tree_w_mask_int.arrays(["events.Muon_Px", "events.Muon_Py"], outputtype=dict)
    assert isinstance(arrays, dict)
    assert len(arrays) == 2
    assert [len(v) for v in arrays.values()] == [int(initial_length / 4), int(initial_length / 4)]

    for outtype in [list, tuple]:
        arrays = tree_w_mask_int.arrays(
            ["events.Muon_Px", "events.Muon_Py"], outputtype=outtype)
        assert isinstance(arrays, outtype)
        assert len(arrays) == 2
        assert len(arrays[0]) == int(initial_length / 4)
        assert len(arrays[1]) == int(initial_length / 4)


def test_arrays_as_numpy(tree_w_mask_int, infile):
    tree_w_mask_int.apply_mask(np.arange(0, len(tree_w_mask_int), 2))
    arrays = tree_w_mask_int.arrays(["events.Muon_Px", "events.Muon_Py"],
                                    outputtype=lambda *args: np.array(args))
    assert isinstance(arrays, np.ndarray)
    assert arrays.shape == (2, 25)


def test_array_as_numpy(tree_w_mask_int, infile):
    tree_w_mask_int.apply_mask(np.arange(0, len(tree_w_mask_int), 2))
    arrays = tree_w_mask_int.arrays(["events.Muon_Px"],
                                    outputtype=lambda *args: np.array(args))
    assert isinstance(arrays, np.ndarray)
    assert arrays.shape == (1, 25)


def test_arrays_as_pandas(tree_w_mask_int, infile):
    tree_w_mask_int.apply_mask(np.arange(0, len(tree_w_mask_int), 2))
    arrays = tree_w_mask_int.arrays(
        ["events.Muon_Px", "events.Muon_Py"], outputtype=pd.DataFrame)
    assert isinstance(arrays, pd.DataFrame)
    assert len(arrays) == 25
    assert len(arrays.columns) == 2

# ####################################
# ######### Multi tree tests #########
# ####################################


def test_multitree_no_mask(multitree_no_mask, multiple_trees_input):
    expected = max([len(tree) for tree in multiple_trees_input.values()])
    assert len(multitree_no_mask) == expected
    assert multitree_no_mask.mask is None


def test_multitree_w_mask_bool(multitree_w_mask_bool, infile):
    initial_length = multitree_w_mask_bool.event_ranger.entries_in_block
    assert len(multitree_w_mask_bool) == int(initial_length / 2)
    new_mask = np.ones(len(multitree_w_mask_bool), dtype=bool)
    new_mask[::2] = False
    multitree_w_mask_bool.apply_mask(new_mask)
    assert len(multitree_w_mask_bool) == int(initial_length / 4)


def test_multitree_w_mask_int(multitree_w_mask_int, infile):
    initial_length = multitree_w_mask_int.event_ranger.entries_in_block
    assert len(multitree_w_mask_int) == int(initial_length / 2)
    multitree_w_mask_int.apply_mask(np.arange(0, len(multitree_w_mask_int), 2))
    assert len(multitree_w_mask_int) == int(initial_length / 4)


def test_contains_with_multiple_trees_from_file(multitree_no_mask):
    trees = multitree_no_mask.trees
    assert 'L1CaloTower' in trees['l1CaloTowerTree/L1CaloTowerTree']

    # assert 'l1CaloTowerTree/L1CaloTowerTree' in multitree_no_mask
    assert 'l1CaloTowerTree.L1CaloTowerTree' in multitree_no_mask

    # assert 'l1CaloTowerTree/L1CaloTowerTree.L1CaloTower' in multitree_no_mask
    assert 'l1CaloTowerTree.L1CaloTowerTree.L1CaloTower' in multitree_no_mask

    # assert 'l1CaloTowerTree/L1CaloTowerTree.L1CaloTower.et' in multitree_no_mask
    assert 'l1CaloTowerTree.L1CaloTowerTree.L1CaloTower.et' in multitree_no_mask
    assert 'l1CaloTowerEmuTree.L1CaloTowerTree.L1CaloTower.et' in multitree_no_mask

    assert 'DoesNotExist' not in multitree_no_mask


def test_multitree_array(multitree_w_mask_int, infile):
    initial_length = multitree_w_mask_int.event_ranger.entries_in_block
    assert len(multitree_w_mask_int) == int(initial_length / 2)
    multitree_w_mask_int.apply_mask(np.arange(0, len(multitree_w_mask_int), 2))
    assert len(multitree_w_mask_int) == int(initial_length / 4)
    assert 'l1CaloTowerEmuTree.L1CaloTowerTree.L1CaloTower.iet' in multitree_w_mask_int
    array = multitree_w_mask_int.array("l1CaloTowerEmuTree.L1CaloTowerTree.L1CaloTower.iet")
    assert len(array) == int(initial_length / 4)


def test_multitree_add_variable(multitree_no_mask):
    assert 'XYZ' not in multitree_no_mask
    new_var = multitree_no_mask.array("l1CaloTowerEmuTree.L1CaloTowerTree.L1CaloTower.iet")
    multitree_no_mask.new_variable("XYZ", new_var)
    assert 'XYZ' in multitree_no_mask

    retrieve_xyz = multitree_no_mask.array("XYZ")
    assert (retrieve_xyz == new_var).flatten().all()


def test_multitree_add_variable_and_full_query(multitree_no_mask):
    assert 'XYZ' not in multitree_no_mask
    new_var = multitree_no_mask.array("l1CaloTowerEmuTree.L1CaloTowerTree.L1CaloTower.iet")
    multitree_no_mask.new_variable("XYZ", new_var)
    assert 'XYZ' in multitree_no_mask

    retrieve_xyz = multitree_no_mask.arrays("XYZ")["XYZ"]
    assert (retrieve_xyz == new_var).flatten().all()


def test_multitree_aliases(multitree_no_mask):
    test_for = 'l1CaloTowerTree.L1CaloTowerTree.L1CaloTower.et'
    assert test_for in multitree_no_mask
    alias = test_for.replace('.', '__DOT__')
    assert alias in multitree_no_mask

    assert (multitree_no_mask[test_for] == multitree_no_mask[alias])
