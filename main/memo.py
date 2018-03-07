from scipy.sparse import save_npz, load_npz
import numpy as np
import pickle as pkl

MEMO_ROOT = 'static/'

def load_pkl(path):
    file = open(path + '.pkl', 'rb')
    return pkl.load(file)

def save_pkl(path, obj):
    file = open(path + '.pkl', 'wb')
    pkl.dump(obj, file)

def load_data(root=MEMO_ROOT):
    train_X = load_npz(root + 'train_X.npz')
    train_Y = np.load(root + 'train_Y.npy')
    test_X = load_npz(root + 'test_X.npz')
    test_Y = np.load(root + 'test_Y.npy')
    summary_cv = load_pkl(root + 'summary_cv')
    review_cv = load_pkl(root + 'review_cv')
    return train_X, train_Y, test_X, test_Y, summary_cv, review_cv

def save_data(train_X, train_Y, test_X, test_Y, summary_cv, review_cv, root=MEMO_ROOT):
    save_npz(root + 'train_X', train_X)
    np.save(root + 'train_Y', train_Y)
    save_npz(root + 'test_X', test_X)
    np.save(root + 'test_Y', train_Y)
    save_pkl(root + 'summary_cv', summary_cv)
    save_pkl(root + 'review_cv', review_cv)

def get_data(load_saved, overwrite_saved, categories, percent, cutoff):
    if load_saved:
        return load_data()
    import loader, parser
    train_X, train_Y, test_X, test_Y = loader.load_categories(categories, percent, cutoff)
    train_X, test_X, summary_cv, review_cv = parser.fit_transform(train_X, test_X)
    if overwrite_saved:
        save_data(train_X, train_Y, test_X, test_Y, summary_cv, review_cv)
    return train_X, train_Y, test_X, test_Y, summary_cv, review_cv

def get_classifier(load_saved, overwrite_saved, classifier, train_X, train_Y, name, root=MEMO_ROOT):
    if load_saved:
        return load_pkl(root + name)
    clf = classifier(train_X, train_Y)
    if overwrite_saved:
        save_pkl(root + name, classifier)
    return clf
