import argparse
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from model.DGIModule import DGIModule  # Substituímos pelo novo modelo
from torch_geometric.data import Data
from tqdm import tqdm
import os
import torch
import numpy as np
import pickle as pkl
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='illinois_cat_placeid', help='city name')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of output representation')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_norm', type=float, default=0.9)
    return parser.parse_args()

def train(data, model, optimizer, scheduler, args):
    model.train()

    optimizer.zero_grad()
    print("\nGradientes zerados")
    
    # Forward pass
    pos_score, neg_score = model(data)
    print(f"pos_score shape: {pos_score.shape} | dtype: {pos_score.dtype}")
    print(f"neg_score shape: {neg_score.shape} | dtype: {neg_score.dtype}")
    
    # Cálculo da perda
    loss = model.loss(pos_score, neg_score)
    print("\nCálculo da Perda:")
    print(f"Loss total: {loss.item():.4f}")
    
    # Backpropagation
    loss.backward()
    
    # Clip de gradientes
    clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    print(f"\nGradientes após clipping (max_norm={args.max_norm}):")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
    
    # Atualização dos pesos
    optimizer.step()
    print("\nPesos atualizados pelo optimizer")
    
    # Atualização do learning rate
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    new_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {old_lr:.6f} → {new_lr:.6f}")
    
    return loss.item()

if __name__ == '__main__':
    args = parse_args()
  
    # Instanciando o modelo
    model = DGIModule(hidden_channels=args.dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma, verbose=False)

    # Carregando os dados
    city_dict_file = f'C:/Users/alvar/OneDrive/Documentos/GitHub/Spatial-Textual-POI-Annotation-Model/data/{args.city}_data.pkl'
    with open(city_dict_file, 'rb') as handle:
        city_dict = pkl.load(handle)

    data = Data(
        embedding_array=torch.tensor(city_dict['embedding_array'], dtype=torch.float32),
        x=torch.tensor(city_dict['embedding_array_test'], dtype=torch.float32),
        edge_index=torch.tensor(city_dict['edge_index'], dtype=torch.int64),
        edge_weight=torch.tensor(city_dict['edge_weight'], dtype=torch.float32),
        number_pois=city_dict['number_pois']
    )

    # Removendo a variável de rótulos y se necessário
    # data.y = None  # Não precisamos de rótulos para o aprendizado não supervisionado
    
    # Mover dados para o dispositivo correto
    place_ids = city_dict.get('place_id', [])
    category2 = city_dict.get('level_0', [])
    data = data.to(args.device)
    
    for epoch in tqdm(range(args.epoch), desc="Epochs de Treinamento"):
        loss_train = train(data, model, optimizer, scheduler, args)
        tqdm.write(f"Epoch {epoch}: Loss = {loss_train:.4f}")

    # Obter embeddings reais do encoder
    with torch.no_grad():
        model.eval()
        embeddings = model.poi_encoder(data.x, data.edge_index)
    
    # Salvando os embeddings
    output_path = f'C:/Users/alvar/OneDrive/Documentos/GitHub/Spatial-Textual-POI-Annotation-Model/data/{args.city}_embeddings.csv'
    embeddings_np = embeddings.cpu().numpy()
    
    df = pd.DataFrame(embeddings_np, columns=[f'embed_{i}' for i in range(embeddings_np.shape[1])])
    df.insert(0, 'place_id', place_ids) 
    df.insert(1, 'category', category2) 
    df.to_csv(output_path, index=False)
    print(f"Embeddings salvos em: {output_path}")
    print(f"Dimensões finais: {embeddings.shape} (deveria ser [num_pois, {args.dim * 4}])")
