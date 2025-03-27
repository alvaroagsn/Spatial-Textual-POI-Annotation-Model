import argparse
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from model.GCNEncoder import GCNClassification
from torch_geometric.data import Data
from tqdm import trange
import math
import os
import torch
import networkx as nx
import numpy as np
import pickle as pkl
import pandas as pd


def parse_args():
    """ parsing the arguments that are used in HGI """
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='illinois_cat', help='city name')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of output representation')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_norm', type=float, default=0.9)
    return parser.parse_args()


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    args = parse_args()
    """load the graph data of a study area"""
    # data = poi_graph(args.city).to(args.device)
    """load the Module"""
    model = GCNClassification(
        hidden_channels=args.dim,
    ).to(args.device)
    """load the optimizer, scheduler (including a warmup scheduler)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=50, gamma=args.gamma, verbose=False)

    def split_train_test(pois_ids, data):
        # kfold 5
        from sklearn.model_selection import StratifiedKFold

        y = data.y.cpu().numpy()

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        kf.get_n_splits(pois_ids, y)

        arr_data = []

        pois_ids_arr = np.array(pois_ids)

        for train_index, test_index in kf.split(pois_ids, y):
            edge_index = data.edge_index.T.cpu().numpy()
            weight = data.edge_weight.cpu().numpy()
            # print(edge_index)

            G = nx.from_edgelist(edge_index)

            for i, w in enumerate(weight):
                G[edge_index[i][0]][edge_index[i][1]]['weight'] = w

            train_ids = pois_ids_arr[train_index]
            test_ids = pois_ids_arr[test_index]

            for i in train_ids:
                if G.has_node(i):
                    G.remove_node(i)
            
            edges = nx.to_pandas_edgelist(G)

            edges_train = edges[["source", "target"]].T.values
            weights_train = edges["weight"].values
            y_train = data.y.cpu().numpy()[train_ids]
            uniq, count = np.unique(y_train, return_counts=True)
            print(count)

            edges_test = data.edge_index.cpu().numpy()
            weights_test = data.edge_weight.cpu().numpy()

            new_features = data.embedding_array.cpu().numpy()
            features_test = data.embedding_array_test.cpu().numpy()
            for i in test_ids:
                new_features[i] = features_test[i]

            features_test = new_features
            y_test = data.y.cpu().numpy()[test_ids]

            data_train = Data(
                x=torch.tensor(new_features, dtype=torch.float32),
                edge_index=torch.tensor(edges_train, dtype=torch.int64),
                edge_weight=torch.tensor(weights_train, dtype=torch.float32),
                y=torch.tensor(y_train, dtype=torch.int64),
                ids=torch.tensor(train_ids, dtype=torch.int64)
            )

            data_test = Data(
                x=torch.tensor(new_features, dtype=torch.float32),
                edge_index=torch.tensor(edges_test, dtype=torch.int64),
                edge_weight=torch.tensor(weights_test, dtype=torch.float32),
                y=torch.tensor(y_test, dtype=torch.int64),
                ids=torch.tensor(test_ids, dtype=torch.int64)
            )

            arr_data.append((data_train, data_test))
            
        return arr_data


    def train(data):
        model.train()
        optimizer.zero_grad()
        poi_emb = model(data)
        loss = model.loss(poi_emb, data.y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        scheduler.step()
        return loss.item()
    
    def test(data):
        model.eval()
        
        predicted = model(data)
        # predicted = predicted[data.ids]
        print(predicted)
        labels = data.y
        print(len(labels))
        ## calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

        predicted = predicted.argmax(dim=1)
        labels = labels.cpu().numpy()
        print(predicted)
        print(labels)
        # precision = precision_score(labels, predicted, average='macro', labels=np.unique(labels))
        # recall = recall_score(labels, predicted, average='macro', labels=np.unique(labels))
        accuracy = accuracy_score(labels, predicted)
        f1_avg = f1_score(labels, predicted, average='macro', labels=np.unique(labels))
        f1_weighted = f1_score(labels, predicted, average='weighted', labels=np.unique(labels))




        return precision_recall_fscore_support(labels, predicted, labels=np.unique(labels)), (accuracy, f1_avg, f1_weighted)



    print("Start training pois classification for the city of {}".format(args.city))
    t = trange(1, args.epoch + 1)
    lowest_loss = math.inf


    city = args.city

    city_dict_file = f'C:/Users/alvar/OneDrive/Documentos/GitHub/Spatial-Textual-POI-Annotation-Model/data/{city}_data.pkl'
    with open(city_dict_file, 'rb') as handle:
        city_dict = pkl.load(handle)
    data = Data(embedding_array=torch.tensor(city_dict['embedding_array'], dtype=torch.float32),
                embedding_array_test=torch.tensor(city_dict['embedding_array_test'], dtype=torch.float32),
                     edge_index=torch.tensor(city_dict['edge_index'], dtype=torch.int64),
                     edge_weight=torch.tensor(city_dict['edge_weight'], dtype=torch.float32),
                     number_pois=city_dict['number_pois'],
                     y=torch.tensor(city_dict['y'], dtype=torch.float64))
    print(data.number_pois)

    precision = []
    recall = []
    fscore = []
    support = [] 
    accuracy = []
    f1_avg_list = []
    f1_weighted_list = []

    import pickle as pkl

    with open(f'C:/Users/alvar/OneDrive/Documentos/GitHub/Spatial-Textual-POI-Annotation-Model/data/le_first_level_name_mapping_{city}.pkl', 'rb') as f:
        map = pkl.load(f)

    values = list(map.values())
    labels_unique = values*5
    for arr_data in split_train_test(list(range(data.number_pois)), data):
        data_train, data_test = arr_data
        for epoch in t:
            loss_train = train(data_train)

            t.set_postfix(train_loss='{:.4f}'.format(loss_train), refresh=True)
        report, general_metrics = test(data_test)
        pre, rec, fsc, sup = report
        acc, f1_avg, f1_weighted = general_metrics

        precision.extend(pre)
        recall.extend(rec)
        fscore.extend(fsc)
        support.extend(sup)

        accuracy.append(acc)
        f1_avg_list.append(f1_avg)
        f1_weighted_list.append(f1_weighted)
    df = pd.DataFrame(list(zip(precision, recall, fscore, ['STPA']*5*7, labels_unique, [city.capitalize()]*5*7)), columns=['precision', 'recall', 'fscore', 'Model', 'category', 'state'])
    df.to_csv(f'C:/Users/alvar/OneDrive/Documentos/GitHub/Spatial-Textual-POI-Annotation-Model/data/{city}_results.csv', index=False)

    df = pd.DataFrame(list(zip(accuracy, f1_avg_list, f1_weighted_list, ['STPA']*5, list(range(1,5+1)), [city.capitalize()]*5)), columns=['accuracy', 'macro_avg', 'weighted_avg', 'Model', 'fold', 'state'])
    df.to_csv(f'C:/Users/alvar/OneDrive/Documentos/GitHub/Spatial-Textual-POI-Annotation-Model/data/{city}_general_results.csv', index=False)


    # {'Community': 0, 'Entertainment': 1, 'Food': 2, 'Nightlife': 3, 'Outdoors': 4, 'Shopping': 5, 'Travel': 6}

    

    # torch.save(region_emb_to_save[0], f'./data/{args.save_name}.torch')









