# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
import re


# This method returns path with name of the binetflow file.
def find_name_of_binetflow(path_to_folder):
    binetflow_file = tf.gfile.Glob(path_to_folder + "/*.binetflow")
    if len(binetflow_file) > 1 or len(binetflow_file) == 0:
        return -1
    return binetflow_file[0]


def check_conn_label(path_to_dataset, infected_ips_list, normal_ips_list):
    print("--------- Checking conn file -------------\n")

    malicious_label = 0
    normal_label = 0

    flow_array = []
    file_name = path_to_dataset + '\\bro\\conn.log'

    # src address in both infected_ips_list and normal_ips_list
    dual_src_add = set()

    with open(file_name) as f:
        for line in f:
            newline = line
            if not ('#' == line[0]):
                split = line.split('\t')
                src_address = split[2]

                flag = "Background"
                if src_address in infected_ips_list:
                    flag = "Malicious"

                if src_address in normal_ips_list:
                    if flag == "Malicious":
                        dual_src_add.add(src_address)
                    else:
                        flag = "Normal"

                if flag == "Background":
                    newline = line.rstrip() + "\t" + "Background" + "\n"
                elif flag == "Normal":
                    newline = line.rstrip() + "\t" + "Normal" + "\n"
                    normal_label += 1
                else:
                    newline = line.rstrip() + "\t" + "Malicious" + "\n"
                    malicious_label += 1
            else:
                if 'fields' in line:
                    newline = line.rstrip() + "\t" + "label" + "\n"
                elif 'types' in line:
                    newline = line.rstrip() + "\t" + "string" + "\n"

            flow_array.append(newline)
    print("malicious:", malicious_label)
    print("normal:", normal_label)
    if dual_src_add:
        print(
            "Note: following srcAddress is in both infected_ips_list and normal_ips_list."
        )
        print(dual_src_add)
    return flow_array


def process_binetflow(entire_path_to_binetflow):
    print("<<< Reading binetflow:")
    print("     <<<\n", entire_path_to_binetflow)

    infected_ips_list = set()
    normal_ips_list = set()

    with open(entire_path_to_binetflow) as f:
        for line in f:
            if 'StartTime' in line:
                term = line.split(',')
                label_i = term.index('Label\n')
                srcadd_i = term.index('SrcAddr')
                continue

            data = line.split(',')
            label = data[label_i]
            src_address = data[srcadd_i]

            if ('Malicious' in label) or ('Botnet' in label) or (
                    'Malware' in label):
                infected_ips_list.add(src_address)

            elif 'Normal' in label:
                normal_ips_list.add(src_address)

    return infected_ips_list, normal_ips_list


def write_conn_label(path, flow_array):
    print("<< Writing conn_label.log --------------\n")
    index = 0
    with open(path + '\\bro\\conn_label.log', 'w+') as f:
        for i in range(len(flow_array)):
            f.write(flow_array[i])
            index += 1

    print("     << Number of lines:", index)
    print("<< New file conn_label.log was succesfly created.")


def check_binetflow_contain_label(path_to_binet):
    with open(path_to_binet) as f:
        for line in f:
            if 'StartTime' in line:
                continue

            data = line.split(',')
            if data[-1] and data[-1] != '\n':  # label is not empty
                return True
            else:
                return False
    return False


def process_given_ip(path):
    normal_ips_list = set()
    infected_ips_list = set()

    with open(path + "\\IPadr.txt") as f:
        for line in f:
            if 'Normal' in line:
                label = 'Normal'
                continue
            if 'Malicious' in line:
                label = 'Malicious'
                continue

            if label == 'Normal':
                normal_ips_list.add(line)
            elif label == 'Malicious':
                infected_ips_list.add(line)
    return infected_ips_list, normal_ips_list


"""
Take labels from binetflows file and then label conn.log
"""


def label_conn_log(path):
    print(">>>------------------------------------------------------------<<<")
    print(path)
    path_to_binet = find_name_of_binetflow(path)
    if path_to_binet != -1 and check_binetflow_contain_label(path_to_binet):
        infected_ips_list, normal_ips_list = process_binetflow(path_to_binet)
    else:
        infected_ips_list, normal_ips_list = process_given_ip(path)

    if infected_ips_list:
        print("Infected ip list: ", infected_ips_list)
    else:
        print("Infected ip list is empty")
    if normal_ips_list:
        print("Normal ip list: ", normal_ips_list)
    else:
        print("Normal ip list is empty")

    flow_array = check_conn_label(path, infected_ips_list, normal_ips_list)
    write_conn_label(path, flow_array)
    print('\n\n')
