import os
import pickle
from random import shuffle

import numpy as np
from scapy.all import rdpcap

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

FLOW_SIZE = 300
N_NEG_PER_POS = 9
MOD = 1 + N_NEG_PER_POS
RATIO_TRAIN = 0.6


def process_pcap_file(filename, flow_size=FLOW_SIZE):
    pcap_here_up = rdpcap(RAW_PATH + filename + '-A-upstream.pcapng')
    pcap_here_down = rdpcap(RAW_PATH + filename + '-A-downstream.pcapng')

    pcap_there_up = rdpcap(RAW_PATH + filename + '-B-upstream.pcapng')
    pcap_there_down = rdpcap(RAW_PATH + filename + '-B-downstream.pcapng')

    time_here_up = []
    time_here_down = []
    size_here_up = []
    size_here_down = []

    time_there_up = []
    time_there_down = []
    size_there_up = []
    size_there_down = []

    dataset = []
    min_size = min(len(pcap_here_up), len(pcap_here_down), len(pcap_there_up), len(pcap_there_down))
    print('Generating dataset', filename, '...')
    for k in range(min_size - 1):
        if not k == 0 and k % flow_size == 0:
            dataset.append(
                {
                    'here': (
                        {'->': time_here_up, '<-': time_here_down}, {'->': size_here_up, '<-': size_here_down}
                    ),
                    'there': (
                        {'->': time_there_up, '<-': time_there_down}, {'->': size_there_up, '<-': size_there_down}
                    )
                }
            )

            time_here_up = []
            time_here_down = []
            size_here_up = []
            size_here_down = []

            time_there_up = []
            time_there_down = []
            size_there_up = []
            size_there_down = []

        time_here_up.append(float(pcap_here_up[k + 1].time - pcap_here_up[k].time))
        time_here_down.append(float(pcap_here_down[k + 1].time - pcap_here_down[k].time))

        size_here_up.append(len(pcap_here_up[k]))
        size_here_down.append(len(pcap_here_down[k]))

        time_there_up.append(float(pcap_there_up[k + 1].time - pcap_there_up[k].time))
        time_there_down.append(float(pcap_there_down[k + 1].time - pcap_there_down[k].time))

        size_there_up.append(len(pcap_there_up[k]))
        size_there_down.append(len(pcap_there_down[k]))

    return dataset


def generate_datasets(topology, flow_size=FLOW_SIZE):
    TOTAL_FILES = 260
    client_data = []
    data = []

    for f in range(TOTAL_FILES):
        file_name = f'{topology}/client/cap{f}'
        if not os.path.exists(RAW_PATH + file_name + '-A-upstream.pcapng'):
            continue
        flow = process_pcap_file(file_name, flow_size=flow_size)
        client_data += flow
        data += flow
    print(f'Total client flows {len(client_data)}')

    pickle.dump(client_data, open(f'{PROCESSED_PATH}{topology}/client_data{flow_size}.pkl', 'wb'))

    TOTAL_FILES = 100
    peers_data = []

    for f in range(TOTAL_FILES):
        file_name = f'{topology}/peers/cap{f}'
        if not os.path.exists(RAW_PATH + file_name + '-A-upstream.pcapng'):
            continue
        flow = process_pcap_file(file_name, flow_size=flow_size)
        peers_data += flow
        data += flow

    print(f'Total peers flows {len(peers_data)}')

    pickle.dump(peers_data, open(f'{PROCESSED_PATH}{topology}/peers_data{flow_size}.pkl', 'wb'))
    pickle.dump(data, open(f'{PROCESSED_PATH}{topology}/joined_data{flow_size}.pkl', 'wb'))


def get_data_labels(dataset, flow_size=FLOW_SIZE, client=1):
    """
    :param flow_size:
    :param dataset: list
    :param client: 1 - for client dataset; 0 - for peers dataset
    :return: x: ndarray, for each dataset sample there are 1 positive(paired) flow + some negative(unpaired) flows.
                        each input data is in shape of [8 * flow_size].
            y: ndarray
    """
    n_pos = len(dataset)
    n_flows = n_pos * MOD
    flows = np.zeros((n_flows, 8, flow_size))
    labels = np.zeros(n_flows)
    print('Generating data and labels...')
    for i in range(n_pos):
        index = MOD * i
        flows[index, 0, :] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
        flows[index, 1, :] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
        flows[index, 2, :] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
        flows[index, 3, :] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0

        flows[index, 4, :] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
        flows[index, 5, :] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
        flows[index, 6, :] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
        flows[index, 7, :] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0

        labels[index] = client
        indices = list(range(n_pos))
        unpaired = indices[:i] + indices[i + 1:]
        shuffle(unpaired)
        for j in range(N_NEG_PER_POS):
            index = MOD * i + j + 1
            if j == len(unpaired):
                break

            flows[index, 0, :] = np.array(dataset[unpaired[j]]['here'][0]['<-'][:flow_size]) * 1000.0
            flows[index, 1, :] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
            flows[index, 2, :] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
            flows[index, 3, :] = np.array(dataset[unpaired[j]]['here'][0]['->'][:flow_size]) * 1000.0

            flows[index, 4, :] = np.array(dataset[unpaired[j]]['here'][1]['<-'][:flow_size]) / 1000.0
            flows[index, 5, :] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
            flows[index, 6, :] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
            flows[index, 7, :] = np.array(dataset[unpaired[j]]['here'][1]['->'][:flow_size]) / 1000.0
            labels[index] = 0
    return flows, labels


def get_indices(dataset_size):
    """
    :param dataset_size: size of the dataset
    :return: indices for training and testing in the x, y matrix.
    """
    indices = list(range(dataset_size))
    n_train = int(dataset_size * RATIO_TRAIN)

    shuffle(indices)
    train_indices = []
    for i in indices[:n_train]:
        train_indices += list(range(i * MOD, i * MOD + N_NEG_PER_POS + 1))
    test_indices = []
    for i in indices[n_train:]:
        test_indices += list(range(i * MOD, i * MOD + N_NEG_PER_POS + 1))
    return train_indices, test_indices


def load_data(traffic_type, flow_size=FLOW_SIZE):
    if not os.path.exists(f'{PROCESSED_PATH}{traffic_type}/client_data{flow_size}.pkl'):
        raise Exception("Client data was not found! Need to generate the dataset before loading the data!")
    if not os.path.exists(f'{PROCESSED_PATH}{traffic_type}/peers_data{flow_size}.pkl'):
        raise Exception("Peers data was not found! Need to generate the dataset before loading the data!")

    dataset_client = pickle.load(open(f'{PROCESSED_PATH}{traffic_type}/client_data{flow_size}.pkl', 'rb'))
    dataset_peers = pickle.load(open(f'{PROCESSED_PATH}{traffic_type}/peers_data{flow_size}.pkl', 'rb'))

    train_index, test_index = get_indices(len(dataset_client) + len(dataset_peers))

    data_client, labels_client = get_data_labels(dataset_client, flow_size, client=1)
    data_peers, labels_peers = get_data_labels(dataset_peers, flow_size, client=0)

    shuffler = np.random.permutation(len(data_client) + len(data_peers))
    data = np.concatenate((data_client, data_peers))[shuffler]
    labels = np.concatenate((labels_client, labels_peers))[shuffler]

    train_x = [data[index] for index in train_index]
    train_y = [labels[index] for index in train_index]

    test_x = [data[index] for index in test_index]
    test_y = [labels[index] for index in test_index]

    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


def load_data_client(traffic_type, flow_size=FLOW_SIZE):
    if not os.path.exists(f'{PROCESSED_PATH}{traffic_type}/client_data{flow_size}.pkl'):
        raise Exception("Client data was not found! Need to generate the dataset before loading the data!")

    dataset_client = pickle.load(open(f'{PROCESSED_PATH}{traffic_type}/client_data{flow_size}.pkl', 'rb'))

    data_client, labels_client = get_data_labels(dataset_client, flow_size, client=1)

    shuffler = np.random.permutation(len(data_client))
    data = data_client[shuffler]
    labels = labels_client[shuffler]

    train_index, test_index = get_indices(len(dataset_client))

    train_x = [data[index] for index in train_index]
    train_y = [labels[index] for index in train_index]

    test_x = [data[index] for index in test_index]
    test_y = [labels[index] for index in test_index]

    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


if __name__ == '__main__':
    # trainX, trainY, testX, testY = load_data(traffic_type='tir')
    for i in range(4):
        TOPO = f'{i}tir'
        for s in [100, 200, 300]:
            print(i, s)
            generate_datasets(TOPO, flow_size=s)
