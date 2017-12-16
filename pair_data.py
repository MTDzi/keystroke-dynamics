import numpy as np
import pandas as pd


def extract_pairs(DF_registration, feature_names, only_feature_diffs):
    '''This function produces a single frame which contains information
    about pairs of password typing patterns.

    Args:
        DF_registration: a pd.DataFrame that should contain only training
            (i.e. registration) data.

        feature_names: a list of column names present in DF_registration,
            that will be used to form the final frame (the rest of the columns
            will be discarded).

        only_feature_diffs: a flag determining whether the function should
            calculate only absulate differences between features, or rather
            keep both feature vectors.



    Returns:
        A pd.DataFrame containing information about pairs of password typing patterns.
            Each row has a label:
                1 -- if that row contains information about two instances of the SAME user;
                0 -- otherwise
    '''
    sequence_in_feature_names = ('sequence' in feature_names)
    if only_feature_diffs and sequence_in_feature_names:
        msg = (
            'If you are calling `extract_pairs` with the flag'
            ' `only_feature_diffs` set to True, "sequence" should not be'
            ' present in `feature_names`'
        )
        raise ValueError(msg)

    user_names = pd.unique(DF_registration['user_name'])
    num_users = len(user_names)

    # Extract negative examples (pairs of rows from two different users)
    all_pair_DFs = []
    for i in range(num_users - 1):
        for j in range(i + 1, num_users):
            user_name_A, user_name_B = user_names[[i, j]]
            selector_A = (DF_registration['user_name'] == user_name_A)
            selector_B = (DF_registration['user_name'] == user_name_B)
            DF_features_A = DF_registration.loc[selector_A, feature_names]
            DF_features_B = DF_registration.loc[selector_B, feature_names]
            # Waaaaay slower, as it turns out, but cleaner:
            # DF_features_A = DF_registration.query('user_name == @user_name_A')[feature_names]
            # DF_features_B = DF_registration.query('user_name == @user_name_B')[feature_names]
            DF_pair_data = get_pair_data(
                only_feature_diffs,
                DF_features_A,
                DF_features_B
            )
            all_pair_DFs.append(DF_pair_data)

    DF_negative = pd.concat(all_pair_DFs)
    DF_negative['label'] = 0

    # Now, extract positive examples (pairs of rows from the same user)
    all_pair_DFs = []
    for user_name in user_names:
        this_user_rows = (DF_registration['user_name'] == user_name)
        DF_features = DF_registration.loc[this_user_rows, feature_names]
        DF_pair_data = get_pair_data(
            only_feature_diffs,
            DF_features
        )
        all_pair_DFs.append(DF_pair_data)

    DF_positive = pd.concat(all_pair_DFs)
    DF_positive['label'] = 1

    return pd.concat([DF_negative, DF_positive])


def get_pair_data(only_feature_diffs, DF_data_A, DF_data_B=None):
    '''Function used to create a frame in which each row contains
    information about two instances of typing in a password on keyboard
    (keystroke dynamics).

    This information can be represented in the form of an absolute difference
    (L1 distance) between features describing those two instances, or it may contain
    all feature values for the two instances. In the first case, the resulting
    frame will contain the same number of columns as the input frames,
    whereas in the latter -- the frame will have twice the number of columns.

    Args:
        only_feature_diffs: a flag determining whether the function should
            calculate only absulate differences between features, or rather
            keep both feature vectors.

        DF_data_A: the first pd.DataFrame ...

        DF_data_B: ... and the second one, whose all row pairs will be created
            and added to the resulting frame. If the second frame is None,
            calculations will be carried out on all pairs of rows within DF_data_A.

    Returns:
        A pd.DataFrame with information about pairs of instances, comming from,
            DF_data_A and DF_data_B. If DF_data_B is None, (unique) pairs will be
            created from rows within DF_data_A.
    '''
    within_A = False
    if DF_data_B is None:
        within_A = True
        DF_data_B = DF_data_A

    if only_feature_diffs:
        # The resulting frame will contain L1 distances between features
        # of two keystroke dynamics only (as opposed to complete feature vectors
        # comming from the two instances)
        DF_pair_data = abs_diff_between_all_row_pairs(DF_data_A, DF_data_B, within_A)
    else:
        # The resulting frame will contain feature vectors from both instances
        DF_pair_data = cartesian_product(DF_data_A, DF_data_B, within_A)

    return DF_pair_data


def abs_diff_between_all_row_pairs(DF_data_A, DF_data_B, within_A):
    '''Function for calculating L1 distances between all pairs of
    rows of frames DF_data_A, and DF_data_B.

    To compute differences between all pairs of rows, we use numpy's
    broadcasting functionality:
    https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html

    NOTE: tensorflow also supports this functionality.
    Here's a question on SO exemplifying evaluation of all pairs of
    rows in tensorflow (which is very similar to how it's done in numpy):
    https://stackoverflow.com/questions/43534057/evaluate-all-pair-combinations-of-rows-of-two-tensors-in-tensorflow
    '''

    # Broadcasting
    pairs_diffs = np.abs(
        np.expand_dims(DF_data_A, 0) - np.expand_dims(DF_data_B, 1)
    )

    num_examples, num_features = DF_data_A.shape
    if within_A:
        # Keep unique pairs within DF_data_A
        pairs_within = [pairs_diffs[i, i+1:] for i in range(num_examples-1)]
        pair_data = np.concatenate(pairs_within)
    else:
        # Take all pairs of rows from DF_data_A and DF_data_B
        pair_data = np.reshape(pairs_diffs, [-1, num_features])

    return pd.DataFrame(pair_data, columns=DF_data_A.columns)


def cartesian_product(DF_data_A, DF_data_B, within_A, suffixes=('_A', '_B')):
    '''Function for producing a Cartesian (or: cross) product of the two input
    frames.

    Args:
        DF_data_A: the first pd.DataFrame ...

        DF_data_B: ... and the second pd.DataFrame, whose Cartesian product
            will be calculated.

        suffixes: a two-element tuple, or list, with suffixes added to colnames
            in the first and second frames, after the Cartesian join.

    Returns:
        A pd.DataFrame that that contains all pairs of rows from frames,
            DF_data_A and DF_data_B, and columns comming from with both
            frames (colnames have added suffixes).
    '''
    tmp_cart_prod_colname = '_tmp_cart_prod'
    DF_data_A[tmp_cart_prod_colname] = 0
    DF_data_B[tmp_cart_prod_colname] = 0

    if within_A:
        tmp_row_number_colname = '_tmp_row_number'
        DF_data_A[tmp_row_number_colname] = range(DF_data_A.shape[0])
        DF_data_B[tmp_row_number_colname] = range(DF_data_B.shape[0])

    DF_all_pairs = pd.merge(DF_data_A, DF_data_B,
                            on=tmp_cart_prod_colname, suffixes=suffixes)
    DF_all_pairs = DF_all_pairs.drop(tmp_cart_prod_colname, axis=1)

    if within_A:
        row_number_colname_A = tmp_row_number_colname + suffixes[0]
        row_number_colname_B = tmp_row_number_colname + suffixes[1]
        nonredundant_rows = (
            DF_all_pairs[row_number_colname_A] > DF_all_pairs[row_number_colname_B]
        )
        DF_all_pairs = DF_all_pairs[nonredundant_rows]
        DF_all_pairs = DF_all_pairs.drop(row_number_colname_A, axis=1)
        DF_all_pairs = DF_all_pairs.drop(row_number_colname_B, axis=1)

    return DF_all_pairs
