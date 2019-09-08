from os import listdir
from os.path import isfile, join
import random
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def save_list_to_file(file_path=None, list_to_save=None):
    f = open(file_path, 'w')
    f.writelines([line + "\n" for line in list_to_save])
    f.close()


def list_files_in_folder(folder_path):
    """
    Return: A list of the file names in the folder
    """

    list = listdir(folder_path)
    onlyfiles = [f for f in list if isfile(join(folder_path, f))]
    return onlyfiles


def load_list_from_file(file_path):
    """
    Return: A list saved in a file
    """

    f = open(file_path, 'r')
    listlines = [line.rstrip() for line in f.readlines()]
    f.close()
    return listlines


def generate_random_idx(dataset=None, nb_graphs=None, save_folder=None):

    for shuffle_idx in range(1,11):
        idxs = list(range(nb_graphs))
        random.shuffle(idxs)
        save_list_to_file(save_folder + "/" + dataset + "_" + str(shuffle_idx), [str(idx) for idx in idxs])


def create_geometric_cv_dataset(raw_cv_data_path, geometric_cv_data_path, dataset_name):
    return 0


def create_geometric_holdout_dataset(raw_holdout_train_data_path, geometric_holdout_train_data_path, dataset_name):
    return 0