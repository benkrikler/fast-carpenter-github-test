import uproot
from atuproot.BEvents import BEvents
from .masked_tree import MaskedUprootTree, MaskedTrees


class EventRanger():
    def __init__(self):
        self._owner = None

    def set_owner(self, owner):
        self._owner = owner

    @property
    def start_entry(self):
        return (self._owner.start_block + self._owner.iblock) * self._owner.nevents_per_block

    @property
    def stop_entry(self):
        i_block = min(self._owner.iblock + 1, self._owner.nblocks)
        stop_entry = (self._owner.start_block + i_block) * self._owner.nevents_per_block
        return min(self._owner.nevents_in_tree, stop_entry)

    @property
    def entries_in_block(self):
        if self._owner and self._owner.iblock > -1:
            return self.stop_entry - self.start_entry
        return None


class BEventsWrapped(BEvents):
    def __init__(self, tree, *args, **kwargs):
        ranges = EventRanger()
        if isinstance(tree, dict):
            tree = MaskedTrees(tree, ranges)
        else:
            tree = MaskedUprootTree(tree, ranges)
        super(BEventsWrapped, self).__init__(tree, *args, **kwargs)
        ranges.set_owner(self)

    def _block_changed(self):
        self.tree.reset_mask()
        self.tree.reset_cache()

    def __getitem__(self, i):
        result = super(BEventsWrapped, self).__getitem__(self, i)
        self._block_changed()
        return result

    def __iter__(self):
        for value in super(BEventsWrapped, self).__iter__():
            self._block_changed()
            yield value
        self._block_changed()


class EventBuilder(object):
    def __init__(self, config):
        self.config = config

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__,
            self.config,
        )

    def __call__(self):
        if len(self.config.inputPaths) != 1:
            # TODO - support multiple inputPaths
            raise AttributeError("Multiple inputPaths not yet supported")
        # Try to open the tree - some machines have configured limitations
        # which prevent memmaps from begin created. Use a fallback - the
        # localsource option
        try:
            # TODO: we probably need iterate here or something else to enforce
            # nevents_per_block, start_block, stop_block
            rootfile = uproot.open(self.config.inputPaths[0])
            if isinstance(self.config.treeName, (list, tuple)):
                tree = {name: rootfile[name] for name in self.config.treeName}
            else:
                tree = rootfile[self.config.treeName]
        except MemoryError:
            rootfile = uproot.open(self.config.inputPaths[0],
                                   localsource=uproot.FileSource.defaults)
            if isinstance(self.config.treeName, (list, tuple)):
                tree = {name: rootfile[name] for name in self.config.treeName}
            else:
                tree = rootfile[self.config.treeName]

        events = BEventsWrapped(tree,
                                self.config.nevents_per_block,
                                self.config.start_block,
                                self.config.stop_block)
        events.config = self.config
        return events
