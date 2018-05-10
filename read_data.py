import os
import glob
import numpy as np
import pandas as pd


class UserData:
    def __init__(self, user_name, dataset):
        self.user_name = user_name
        self.dataset = dataset
        self.reg_data = None
        self.DF_sequences = pd.DataFrame()

    def add_file(self, filename):
        file_empty = (os.stat(filename).st_size == 0)
        if file_empty:
            return

        with open(filename) as inp:
            one_file_data = []
            for line in inp:
                timestamps = line.split(',')
                if len(timestamps) == 1:
                    continue
                else:
                    one_file_data.append([int(t) for t in timestamps])

        # If the filename contains "[]", it means that this file contains
        # sequences provided during registration
        if '[]' in filename:
            for seq in one_file_data:
                self._append_one_row_to_DF_sequences(1, 0, 'y', seq)
            return

        # If this is not a registration file, it should contain only one
        # sequence
        assert len(one_file_data) == 1

        # The file has a name of the form "BLABLABLA_1_n.txt", where
        # `1` denotes whether this sequence comes from an imposter, and
        # `n` denotes whether the password was correct (NOTE: I'm assuming
        # that's what it means -- it's not 100% clear from the article
        # describing the dataset)
        components = filename.split('_')
        imposter = int(components[1])
        success = components[2].split('.')[0]
        self._append_one_row_to_DF_sequences(
            0, imposter, success, one_file_data[0]
	)

    def _append_one_row_to_DF_sequences(self, reg, imp, succ, seq):
        seq, false_start, odd = self._seq_cleanup(seq)
        succ = {'y': 1, 'n': 0}[succ]
        self.DF_sequences = self.DF_sequences.append({
            'user_name': self.user_name,
            'registration': reg,
            'imposter': imp,
            'success': succ,
            'sequence': seq,
            'dataset': self.dataset,
            'false_start': false_start,
            'odd': odd,
        }, ignore_index=True)

    def _seq_cleanup(self, seq):
        seq = np.array(seq)
        false_start = 0
        odd = 0
        if seq[0] != 0:
            seq -= seq[0]
            false_start = 1
        if len(seq) % 2 == 1:
            seq = seq[:-1]
            odd = 1
        return seq, false_start, odd


def produce_whole_DF(path, datasets=['A'], presentation=True):
    all_user_data = {}
    for dataset in datasets:
        dirnames = sorted(glob.glob(os.path.join(path, 'Dataset' + dataset, '*')))
        for dirname in dirnames:
            user_name = os.path.split(dirname)[-1]
            user_data = UserData(user_name, dataset)
            user_filenames = glob.glob(os.path.join(dirname, '*'))
            for filename in user_filenames:
                user_data.add_file(filename)

            all_user_data[user_name] = user_data

    DF_whole = pd.concat(
        [user_data.DF_sequences for user_data in all_user_data.values()],
        ignore_index=True
    )

    if presentation:
        redundant_colnames = ['dataset', 'odd', 'false_start', 'success']
        DF_whole = DF_whole.drop(redundant_colnames, axis=1)

    return DF_whole
