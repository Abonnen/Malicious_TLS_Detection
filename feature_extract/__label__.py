# -*- coding: utf-8 -*-

import label_log
import config_manager
if __name__ == "__main__":
    dataset_path = config_manager.read_config()
    if dataset_path == -1:
        raise ValueError
    malicious_path = dataset_path + "\\Malicious"
    malicious_folder_path = config_manager.get_folders_name(malicious_path)
    for dir_name in malicious_folder_path:
        path_to_single = malicious_path + "\\" + dir_name
        label_log.label_conn_log(path_to_single)
    normal_path = dataset_path + "\\Normal"
    normal_folder_path = config_manager.get_folders_name(normal_path)
    # process normal dataset
    for dir_name in normal_folder_path:
        path_to_single = normal_path + "\\" + dir_name
        label_log.label_conn_log(path_to_single)