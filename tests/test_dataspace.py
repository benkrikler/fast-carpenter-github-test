
import pytest
import fast_carpenter.dataspace as ds


@pytest.mark.parametrize(
    'path',
    [
        ('tree1/s'),
        ('tree1/folder/something'),
        (b'tree1/folder/something'),
        (b'tree1/s'),
    ]
)
def test_normalize_internal_path(path):
    normPath = ds.normalize_internal_path(str(path))
    if type(path) is str:
        assert '/' in str(path)
    if type(path) is bytes:
        assert b'/' in path
    assert '/' not in str(normPath)


@pytest.mark.parametrize(
    'index, provenance, name, value, expected',
    [
        ({}, ['path', 'tree'], 'var', 3, {'path.tree.var': 3}),
        ({}, ['tree'], 'var', 42, {'tree.var': 42}),
        ({'var': 42}, ['tree'], 'var', 42, {'tree.var': 42, 'var': 42}),
        ({'tree.var': 42}, ['tree'], 'var', 42, {'tree.var': 42}),
    ]
)
def test_add_to_index(index, provenance, name, value, expected):
    if provenance:
        name = '.'.join(provenance + [name])
    index = ds.add_to_index(index, name, value)
    assert index == expected


@pytest.mark.parametrize(
    'index, provenance, dict_like_object, expected',
    [
        ({}, [], {'var': 3}, {'var': 3}),
        ({}, ['path', 'tree'], {'var': 3, 'obj': {'et': 42}}, {
         'path.tree.var': 3, 'path.tree.obj.et': 42,
         'path.tree.obj': {'et': 42},
         'path.tree': {'var': 3, 'obj': {'et': 42}},
         }),
    ]
)
def test_recursive_index(index, provenance, dict_like_object, expected):
    index = ds.recursive_index(index, provenance, dict_like_object)
    assert index == expected
