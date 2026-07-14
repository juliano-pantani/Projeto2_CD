# Projeto 2 - Ciência de Dados

**Anime VAE Upsampler: Geração de Detalhes em Imagens de Animes via Variational Autoencoder**

**Aluno:** Juliano Pantani  
**Disciplina:** Ciência de Dados  
**Professor:** Thiago Rodrigo Ramos  
**Data:** Julho/2026

---

## 📋 Resumo

Este projeto consiste no desenvolvimento de um **Variational Autoencoder (VAE)** capaz de realizar *super-resolução* de imagens de animes. O modelo recebe imagens extremamente reduzidas (10x10 pixels) e gera versões de 200x200 pixels, adicionando detalhes plausíveis através do aprendizado do espaço latente.

Além do modelo, foi implementada uma **interface web interativa** que permite buscar animes, visualizar o processo de upscaling e explorar o espaço latente por meio de sliders em tempo real.

---

## 🎯 Objetivos

- Implementar um VAE customizado para tarefa de super-resolução
- Utilizar um dataset real de capas de animes do MyAnimeList
- Criar uma aplicação interativa para demonstração e exploração do modelo
- Compreender na prática os conceitos de espaço latente, reparametrização e treinamento de VAEs

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3
- **Framework de Deep Learning:** PyTorch
- **Processamento de Imagens:** OpenCV e NumPy
- **Manipulação de Dados:** Pandas
- **Interface:** HTML5 + JavaScript (frontend) + Servidor HTTP Python (backend)
- **Gerenciamento de Pacotes:** uv / pyproject.toml

---

## 📁 Estrutura do Projeto

- `vae_modelo.py` — Definição da arquitetura do VAE (Encoder + Decoder)
- `vae_treinamento.py` — Script responsável pelo treinamento do modelo
- `vae_app.py` — Servidor web com rotas de busca, autocomplete e decodificação latente
- `interface.html` — Interface gráfica interativa
- `abrir_interface.py` — Script auxiliar para carregar o modelo e iniciar a aplicação
- `banco_de_dados_animes_limpo.csv` — Dataset com informações e URLs de imagens de animes
- `vae_upsampler.pt` — Pesos do modelo treinado

---

## 🔬 Metodologia

### Arquitetura do Modelo

O VAE desenvolvido possui:

- **Encoder**: Reduz a imagem 10×10 para um vetor latente de 128 dimensões (média e variância)
- **Decoder**: Gera um mapa de resíduos (detalhes) que é somado à versão interpolada bilinear da imagem pequena
- **Reparametrização**: Técnica essencial para permitir o backpropagation no espaço latente
- **Função de Perda**: Combinação de MSE (reconstrução) + termo KL Divergence

### Dataset

- Extraído do MyAnimeList
- Contém capas de animes e metadados
- Limitado a 300 imagens durante o treinamento para viabilidade computacional

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
```bash
uv sync
# ou
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas opencv-python requests numpy
