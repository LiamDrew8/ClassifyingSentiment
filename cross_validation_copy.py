import numpy as np

from performance_metrics import calc_root_mean_squared_error

def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    '''
    train_error_per_fold = np.zeros(n_folds, dtype=np.float32)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float32)
    
    

    # TODO define the folds here by calling your function
    # e.g. ... = make_train_and_test_row_ids_for_n_fold_cv(...)

    # TODO loop over folds and compute the train and test error
    # for the provided estimator
    train_ids_per_fold, test_ids_per_fold = make_train_and_test_row_ids_for_n_fold_cv(x_NF.shape[0], n_folds, random_state)
    #print(train_ids_per_fold[0])
    
    for i in range(0, n_folds):
        #fit and predict for that fold
        estimator.fit(x_NF[train_ids_per_fold[i]], y_N[train_ids_per_fold[i]])
        yhat_M = estimator.predict(x_NF[train_ids_per_fold[i]])
        train_error_per_fold[i] = calc_root_mean_squared_error(y_N[train_ids_per_fold[i]], yhat_M)
        
#         print("train?")
#         print(x_NF[train_ids_per_fold[i]].shape)
#         print("test?")
#         print(x_NF[test_ids_per_fold[i]].shape)
        test_yhat_M = estimator.predict(x_NF[test_ids_per_fold[i]])
        test_error_per_fold[i] = calc_root_mean_squared_error(y_N[test_ids_per_fold[i]], test_yhat_M)
    
    

    return train_error_per_fold, test_error_per_fold

def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N

    '''
    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    # TODO obtain a shuffled order of the n_examples
    
    shuffled_examples = random_state.permutation(np.arange(n_examples))
    #now we have all the shuffled examples of 
    
#     print(shuffled_examples)
#     print(shuffled_examples[:3])
#     print(shuffled_examples[3:6])
    
    
    sorted_count = 0
    
    
    
        #ex_per_fold = math.ceil(n_examples / n_folds)
    
    #train_ids_per_fold = np.split(shuffled_examples, ex_per_fold)
    train_ids_per_fold = list()
    test_ids_per_fold = list()
    
    
    
    remaining_examples = n_examples
    previous_examples_per_fold = 0
    cumulative_examples = 0
    
    for fold_num in range(0, n_folds):
        remaining_folds = n_folds - fold_num
        
        ex_per_f = ((n_examples - cumulative_examples) // remaining_folds)
        current_examples_per_fold =  ex_per_f + cumulative_examples
#         print("current examples per fold")
#         print(current_examples_per_fold)
        
        #print("Test data")
        #print(previous_examples_per_fold)
        #print(current_examples_per_fold)
        #print(shuffled_examples[previous_examples_per_fold:current_examples_per_fold])
        test_ids_per_fold.append(shuffled_examples[previous_examples_per_fold:current_examples_per_fold])
                                
        #print("Train data")
        #print(np.concatenate([shuffled_examples[0:previous_examples_per_fold], shuffled_examples[current_examples_per_fold:]]))
        train_ids_per_fold.append(np.concatenate
                                  ([shuffled_examples[0:previous_examples_per_fold], shuffled_examples[current_examples_per_fold:]])
                                 )
        
        previous_examples_per_fold = current_examples_per_fold
        cumulative_examples += ex_per_f
        
        
    
    # TODO establish the row ids that belong to each fold's
    # train subset and test subset

    return train_ids_per_fold, test_ids_per_fold