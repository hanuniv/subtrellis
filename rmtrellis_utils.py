import numpy as np
import operator
import json
import pickle
import inspect
from glob import glob

def retrieve_name(var):
    """ get name string, c.f. https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string"""
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def save_pickle(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)


def load_obj(name):
    with open(name + '.json', 'r') as f:
        return json.load(f)

def indpkls(dataname):
    """yield integer part after dataname*.pkl, ignore non-integer files"""
    for s in glob(dataname+'*.pkl'):
        stri = s[len(dataname):-4]
        if not stri.isdigit():
            continue
        i = int(stri)
        yield i

def compress_pickle(suffix='rm14', dim=2, inds = None):
    """
    Reduces storage usage by collecting psucc, jlist, alist to an id.
    """
    def indexadd(l, item, comp=operator.__eq__):
        """
        returns the index of the item in the list l,
        if item does not exist in l (determined with comp),
        add it to the list
        """
        for i, x in enumerate(l):
            if comp(item, x):
                return i
        l.append(item)
        return len(l) - 1
    dataname = 'data/' + suffix + '_{0}dim_'.format(dim)
    if inds is None:
        inds = indpkls(dataname)
    Gs = []
    states = []
    Ps = []
    pss = []
    psuccs = []
    alists = []
    jlists = []
    for i in inds:
        T, SB, ps, psucc, jlist, alist = load_pickle(dataname + str(i))
        igs = indexadd(Gs, T.G, comp=np.allclose)
        istates = indexadd(states, T.state_space)
        iPs = indexadd(Ps, T.P)
        ips = indexadd(pss, ps, comp=np.allclose)  # ps is np.ndarray
        ipsucc = indexadd(psuccs, psucc, comp=np.allclose)
        ias = [indexadd(alists, a) for a in alist]
        ijs = [indexadd(jlists, j) for j in jlist]
        save_pickle((igs, SB, istates, iPs, ips, ipsucc, ias, ijs), dataname + str(i) + '_l')
    for name in (Gs, pss, psuccs):
        for s in retrieve_name(name):
            if s.endswith('s'): # otherwise save to "name"
                namestring = s
                break
        print("Saving: {} with length {}".format(namestring, len(name)))
        save_pickle(name, dataname + namestring)
    save_pickle([alists, jlists, Ps, states], dataname+'ajps') # combine big items into one to save storage

def test_compress(suffix='rm14', dim=2, inds = None):
    """ testing specific retrieval, inds can be [0, 2] or None"""
    dataname = 'data/' + suffix + '_{0}dim_'.format(dim)
    if inds is None:
        inds = indpkls(dataname)
    for ind in inds:
        T, SB, ps, psucc, jlist, alist = load_pickle(dataname + str(ind))
        igs, SB, istates, iPs, ips, ipsucc, ias, ijs = load_pickle(dataname + str(ind) + '_l')
        Gs = load_pickle(dataname+'Gs')
        pss = load_pickle(dataname+'pss')
        psuccs = load_pickle(dataname+'psuccs')
        # alists = load_pickle(dataname+'alists')
        # jlists = load_pickle(dataname+'jlists')
        # Ps = load_pickle(dataname+'Ps')
        # states = load_pickle(dataname+'states')
        alists, jlists, Ps, states = load_pickle(dataname+'ajps')
        assert np.allclose(Gs[igs], T.G)
        assert states[istates] == T.state_space
        assert Ps[iPs] == T.P
        assert np.allclose(pss[ips], ps)
        assert np.allclose(psuccs[ipsucc], psucc)
        for j, ij in zip(jlist, ijs):
            assert j == jlists[ij]
        for a, ia in zip(alist, ias):
            assert a == alists[ia]

def combine_pickles(files, newfile):
    items = []
    for f in files:
        items.append(load_pickle(f))
    save_pickle(items, newfile)


def main():
    compress_pickle(suffix='rm14', dim=4)
    test_compress(suffix='rm14', dim=4, inds = None)


if __name__ == '__main__':
    main()
