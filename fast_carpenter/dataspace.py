
def normalize_internal_path(path):
    if type(path) is str:
        return path.replace('/', '.')
    if type(path) is bytes:
        return path.replace(b'/', b'.').decode('utf-8')
    return path


def add_to_index(index, provenance, name, value):
    name = normalize_internal_path(name)
    full_path = '.'.join(provenance + [name])
    if name not in index and full_path not in index:
        name = name.decode('utf-8') if type(name) is bytes else name
        index[name] = value
        index[full_path] = value
    return index
    # TODO: what do we want to happen if name is included but full_path is not,
    # i.e. when multiple trees have the same variable name


def alias(key):
    if '.' in key:
        return key.replace('.', '__DOT__')
    return key


def create_aliases(index):
    aliases = {}
    for key in index.keys():
        a = alias(key)
        if a != key:
            aliases[a] = index[key]
    index.update(aliases)
    return index


def recursive_index(index, provenance, dict_like_object):
    if provenance:
        index = add_to_index(index, [], '.'.join(provenance), dict_like_object)
    for n in dict_like_object.keys():
        v = dict_like_object[n]
        n = n.decode('utf-8') if isinstance(n, bytes) else n
        if hasattr(v, 'keys'):
            index = recursive_index(index, provenance + [n], v)
        else:
            path = normalize_internal_path('.'.join(provenance + [n]))
            index = add_to_index(index, [], path, v)

    return index