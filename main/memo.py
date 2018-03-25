from scipy.sparse import save_npz, load_npz
import numpy as np
import pickle as pkl

MEMO_ROOT = 'static/'

def load_txt(path): # unused
    '''Loads text from a given text file (excluding the extension).'''
    file = open(path + '.txt', 'r')
    return file

def save_txt(path, content):
    '''Saves text to a given text file (excluding the extension).'''
    file = open(path + '.txt', 'w')
    file.write(content)
    file.close()

def load_pkl(path):
    '''Loads an object from a given pickle file (excluding the extension).'''
    file = open(path + '.pkl', 'rb')
    return pkl.load(file)

def save_pkl(path, obj):
    '''Saves an object to a given pickle file (excluding the extension).'''
    file = open(path + '.pkl', 'wb')
    pkl.dump(obj, file)

def load_data(root=MEMO_ROOT):
    '''Loads train data, test data, summary vectorizers, and review vectorizers.'''
    train_X = load_npz(root + 'train_X.npz')
    train_Y = np.load(root + 'train_Y.npy')
    test_X = load_npz(root + 'test_X.npz')
    test_Y = np.load(root + 'test_Y.npy')
    summary_cv = load_pkl(root + 'summary_cv')
    review_cv = load_pkl(root + 'review_cv')
    return train_X, train_Y, test_X, test_Y, summary_cv, review_cv

def save_data(train_X, train_Y, test_X, test_Y, summary_cv, review_cv, root=MEMO_ROOT):
    '''Saves train data, test data, and vectorizers.'''
    save_npz(root + 'train_X', train_X)
    np.save(root + 'train_Y', train_Y)
    save_npz(root + 'test_X', test_X)
    np.save(root + 'test_Y', test_Y)
    save_pkl(root + 'summary_cv', summary_cv)
    save_pkl(root + 'review_cv', review_cv)

def get_data(load_saved, overwrite_saved, categories, percent, cutoff, root=MEMO_ROOT):
    '''
    Loads and/or saves train data, test data, and vectorizers.

    load_saved:      whether or not to load saved files
    overwrite_saved: whether or not to save data to a file
    categories:      a list of category file basenames
    percent:         the percent of data to use as train data (the remainder is test data)
    cutoff:          the number of reviews to use from the category file (skips unhelpful reviews)
    root:            the root directory from which to save/load
    '''
    if load_saved:
        return load_data(root)
    import loader, parser
    train_X, train_Y, test_X, test_Y = loader.load_categories(categories, percent, cutoff)
    train_X, test_X, summary_cv, review_cv = parser.fit_transform(train_X, test_X)
    if overwrite_saved:
        save_data(train_X, train_Y, test_X, test_Y, summary_cv, review_cv)
    return train_X, train_Y, test_X, test_Y, summary_cv, review_cv

def get_classifier(load_saved, overwrite_saved, classifier, train_X, train_Y, name, root=MEMO_ROOT):
    '''
    Loads and/or saves classifiers with the specified name.

    load_saved:      whether or not to load saved files
    overwrite_saved: whether or not to save data to a file
    classifier:      the classifier object to save (parameter ignored if load_saved=true)
    train_X:         the train data to train and save the classifiers with (parameter ignored if load_saved=true)
    train_Y:         the train labels to train and save the classifiers with (parameter ignored if load_saved=true)
    name:            the base name (excluding root directory and extension)
    root:            the root directory from which to save/load
    '''
    if load_saved:
        return load_pkl(root + name)
    clf = classifier(train_X, train_Y)
    if overwrite_saved:
        save_pkl(root + name, clf)
    return clf
