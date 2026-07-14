import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import requests
import os
from tqdm import tqdm
from vae_modelo import VAEUpsampler # Certifique-se de que o arquivo se chama vae_modelo.py

# --- CONFIGURACOES E FUNCOES DE DADOS ---
CSV_PATH = "banco_de_dados_animes_limpo.csv"
TAMANHO_PEQUENO = (10, 10)
TAMANHO_GRANDE = (200, 200)
N_CANAIS = 3
LIMITE_IMAGENS = 300
BATCH_SIZE = 30
HEADERS = {"User-Agent": "Mozilla/5.0"}

def baixar_par(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        img_arr = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pequena = cv2.resize(img, TAMANHO_PEQUENO, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        grande = cv2.resize(img, TAMANHO_GRANDE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return pequena, grande
    except: return None

class ParesDataset(Dataset):
    def __init__(self, pequenas, grandes):
        self.pequenas = pequenas
        self.grandes = grandes
    def __len__(self): return len(self.pequenas)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.pequenas[idx].transpose(2, 0, 1)),
                torch.from_numpy(self.grandes[idx].transpose(2, 0, 1)))

# --- FUNCAO DE TREINO ---
def treinar():
    # Carregamento do dataset
    df = pd.read_csv(CSV_PATH)
    urls = df["Image_URL"].dropna().tolist()
    pequenas, grandes = [], []
    for url in tqdm(urls[:LIMITE_IMAGENS]):
        par = baixar_par(url)
        if par:
            pequenas.append(par[0])
            grandes.append(par[1])
    
    dataset = ParesDataset(np.array(pequenas), np.array(grandes))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = VAEUpsampler().to(device)
    otim = optim.Adam(modelo.parameters(), lr=1e-3)

    print("Iniciando treino...")
    for epoca in range(1, 31):
        for x_p, x_g in loader:
            x_p, x_g = x_p.to(device), x_g.to(device)
            otim.zero_grad()
            rec, mu, logvar = modelo(x_p)
            loss = torch.nn.functional.mse_loss(rec, x_g) - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss.backward()
            otim.step()
        print(f"Época {epoca} finalizada.")

    torch.save(modelo.state_dict(), "vae_upsampler.pt")
    print("Modelo salvo como 'vae_upsampler.pt'.")

if __name__ == "__main__":
    treinar()