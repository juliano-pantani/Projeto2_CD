import torch
from vae_upsampler_ver2 import VAEUpsampler, ParesDataset, montar_dataset, obter_vetor_latente, executar_visualizador, device, LATENT_DIM

# 1. Recarregar os dados necessários
print("--- Montando dataset ---")
pequenas, grandes = montar_dataset()
dataset = ParesDataset(pequenas, grandes)

# 2. Recriar a estrutura do modelo
print("--- Carregando o modelo treinado ---")
modelo = VAEUpsampler(latent_dim=LATENT_DIM).to(device)
modelo.load_state_dict(torch.load("vae_upsampler.pt", map_location=device))

# 3. Iniciar o visualizador
print("\n--- Iniciando o Visualizador Interativo ---")
img_teste = dataset[0][0] 
mu, logvar, z = obter_vetor_latente(modelo, img_teste)
executar_visualizador(modelo, z, porta=8080)