import os

import pytest
import uproot
from collections import namedtuple
from six.moves.urllib.request import urlretrieve as download


@pytest.fixture
def infile():
    filename = "tests/data/CMS_HEP_tutorial_ww.root"
    return uproot.open(filename)["events"]


@pytest.fixture
def single_tree_input(infile):
    return {'events': infile}


@pytest.fixture
def multiple_trees_input():
    input_file = '/tmp/CMS_L1T_study.root'
    url = 'http://fast-hep-data.web.cern.ch/fast-hep-data/cms/L1T/CMS_L1T_study.root'
    if not os.path.exists(input_file):
        download(url, input_file)
    trees = ['l1CaloTowerEmuTree/L1CaloTowerTree',
             'l1CaloTowerTree/L1CaloTowerTree']
    f = uproot.open(input_file)
    trees = {tree: f[tree] for tree in trees}
    return trees


FakeEventRange = namedtuple("FakeEventRange", "start_entry stop_entry entries_in_block")


@pytest.fixture
def event_range():
    return FakeEventRange(100, 200, 100)


@pytest.fixture
def full_event_range():
    return FakeEventRange(0, 4580, 4580)


@pytest.fixture
def full_event_range_multitree():
    return FakeEventRange(0, 1853, 1853)


@pytest.fixture
def wrapped_tree(infile, event_range):
    import fast_carpenter.tree_wrapper as tree_w
    tree = tree_w.WrappedTree(infile, event_range)
    return tree


@pytest.fixture
def full_wrapped_tree(infile, full_event_range):
    import fast_carpenter.tree_wrapper as tree_w
    tree = tree_w.WrappedTree(infile, full_event_range)
    return tree


class Namespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeBEEvent(object):
    def __init__(self, tree, eventtype):
        self.tree = tree
        self.config = Namespace(dataset=Namespace(eventtype=eventtype))
