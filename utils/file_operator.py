import os
import pickle
import json
import cv2


def create_folder(full_path):
    """ Make directory if specified patch does not exist """
    dir_list = full_path.split("/")
    for i in range(len(dir_list)):
        chk_path = "/".join(dir_list[0:i+1])
        if not os.path.exists(chk_path):
            os.mkdir(chk_path)


def save_target_as_pickle(target, save_root_path, save_key):
    create_folder(save_root_path)
    with open(f"{save_root_path}/{save_key}.pickle", "wb") as f:
        pickle.dump(target, f)


def save_as_jpg(img_array, save_key, save_root_path, file_name):
    """ img_array should be [0, 255] """
    target_dir = f"{save_root_path}/{save_key}"
    create_folder(target_dir)
    cv2.imwrite(target_dir+f"/{file_name}.jpg", img_array)


def load_pickle(load_root_path, load_key):
    with open(f"{load_root_path}/{load_key}.pickle", "rb") as f:
        return pickle.load(f)


def load_json(load_root_path, load_key):
    json_open = open(f"{load_root_path}/{load_key}.json", "r")
    parameters = json.load(json_open)
    json_open.close()
    return parameters
