import os
import requests
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACOES
# ==========================================

CSV_PATH = "banco_de_dados_animes_limpo.csv"
TAMANHO_PEQUENO = (10, 10)     # entrada do encoder
TAMANHO_GRANDE = (200, 200)    # saida do decoder
N_CANAIS = 3                   # RGB
LATENT_DIM = 128
LIMITE_IMAGENS = 200          # quantas imagens baixar para treino (200 e MUITO pouco p/ essa tarefa)
BATCH_SIZE = 16
N_EPOCAS = 30
LR = 1e-3
BETA_KL_MAX = 1e-3             # peso MAXIMO do termo KL (regularizacao do espaco latente)
KL_WARMUP_EPOCAS = 20          # annealing: KL comeca em 0 e sobe ate BETA_KL_MAX nessas epocas
                                # (evita "colapso posterior": decoder ignorando o latente)
SEED = 0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# ==========================================
# DOWNLOAD: par (imagem pequena, imagem grande) da mesma URL
# ==========================================

def baixar_par(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        img_arr = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pequena = cv2.resize(img, TAMANHO_PEQUENO, interpolation=cv2.INTER_AREA)
        grande = cv2.resize(img, TAMANHO_GRANDE, interpolation=cv2.INTER_AREA)

        pequena = pequena.astype(np.float32) / 255.0
        grande = grande.astype(np.float32) / 255.0
        return pequena, grande
    except Exception:
        return None


def montar_dataset():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Nao encontrei '{CSV_PATH}'. Coloque o CSV na mesma pasta deste script "
            "(o mesmo usado no notebook de difusao)."
        )

    df = pd.read_csv(CSV_PATH)
    urls = df["Image_URL"].dropna().tolist()

    pequenas, grandes = [], []
    barra = tqdm(urls, total=min(LIMITE_IMAGENS, len(urls)), desc="Baixando pares", unit="img")
    for url in barra:
        par = baixar_par(url)
        if par is not None:
            p, g = par
            pequenas.append(p)
            grandes.append(g)
        if len(pequenas) >= LIMITE_IMAGENS:
            break
    barra.close()

    if len(pequenas) == 0:
        print("Falha ao baixar imagens do CSV. Usando dados sinteticos para nao travar o script.")
        pequenas = [np.random.rand(*TAMANHO_PEQUENO, N_CANAIS).astype(np.float32) for _ in range(8)]
        grandes = [np.random.rand(*TAMANHO_GRANDE, N_CANAIS).astype(np.float32) for _ in range(8)]

    print(f"Dataset pronto: {len(pequenas)} pares (10x10 -> 200x200)")
    return np.array(pequenas), np.array(grandes)


class ParesDataset(Dataset):
    def __init__(self, pequenas, grandes):
        self.pequenas = pequenas  # (N, 10, 10, 3)
        self.grandes = grandes    # (N, 200, 200, 3)

    def __len__(self):
        return len(self.pequenas)

    def __getitem__(self, idx):
        pequena = self.pequenas[idx].transpose(2, 0, 1)  # (3, 10, 10)
        grande = self.grandes[idx].transpose(2, 0, 1)    # (3, 200, 200)
        return (
            torch.from_numpy(pequena.astype(np.float32)),
            torch.from_numpy(grande.astype(np.float32)),
        )


# ==========================================
# ARQUITETURA DO VAE
# ==========================================

class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_CANAIS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 10 * 10, latent_dim)
        self.fc_logvar = nn.Linear(64 * 10 * 10, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 10 * 10)
        self.net = nn.Sequential(
            nn.Unflatten(1, (64, 10, 10)),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 20x20
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 40x40
            nn.Upsample(size=(50, 50), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 100x100
            nn.ConvTranspose2d(16, N_CANAIS, kernel_size=4, stride=2, padding=1),        # 200x200
            # SEM Sigmoid aqui: essa saida agora e um RESIDUO (delta), nao a imagem final.
            # O motivo: sem residuo, o decoder precisa reconstruir a imagem 200x200 INTEIRA
            # so a partir do vetor latente, e com poucos dados ele aprende a "chutar" a
            # media do dataset (o borrao marrom que aparece igual pra qualquer entrada).
            # Somando um upsample simples da entrada como base, a rede so precisa aprender
            # os detalhes/correcoes, o que e muito mais facil e MANTEM a saida ligada a entrada.
        )

    def forward(self, z):
        x = self.fc(z)
        return self.net(x)


class VAEUpsampler(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparametrizar(self, mu, logvar):
        # truque de reparametrizacao: z = mu + sigma * epsilon, epsilon ~ N(0, 1)
        # permite o gradiente passar pela amostragem (senao "amostrar" nao e diferenciavel)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x_pequena):
        mu, logvar = self.encoder(x_pequena)
        z = self.reparametrizar(mu, logvar)
        delta = self.decoder(z)

        # base: um upsample bilinear "burro" da entrada 10x10 -> 200x200.
        # Isso garante que, mesmo no inicio do treino, a saida ja se pareca em
        # cores/formas gerais com a entrada (em vez de ser um borrao independente dela).
        base = nn.functional.interpolate(
            x_pequena, size=delta.shape[-2:], mode="bilinear", align_corners=False
        )
        x_rec = torch.clamp(base + delta, 0.0, 1.0)
        return x_rec, mu, logvar


def beta_do_epoca(epoca, beta_max=BETA_KL_MAX, warmup=KL_WARMUP_EPOCAS):
    """Annealing linear do beta do KL: comeca em 0 e sobe ate beta_max.
    Evita 'colapso posterior' (decoder aprendendo a ignorar o latente),
    que e outra causa comum de saidas identicas para entradas diferentes."""
    return beta_max * min(1.0, epoca / max(1, warmup))


def perda_vae(x_rec, x_alvo, mu, logvar, beta):
    """Perda = erro de reconstrucao + beta * divergencia KL (regularizacao do latente)."""
    rec_loss = nn.functional.mse_loss(x_rec, x_alvo, reduction="mean")
    # KL entre N(mu, sigma^2) e N(0, 1), forma fechada
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec_loss + beta * kl_loss, rec_loss, kl_loss


# ==========================================
# TREINO
# ==========================================

def treinar(modelo, loader, n_epocas=N_EPOCAS, lr=LR):
    otim = optim.Adam(modelo.parameters(), lr=lr)
    historico = []

    modelo.train()
    for epoca in range(1, n_epocas + 1):
        beta_atual = beta_do_epoca(epoca)
        perda_total, rec_total, kl_total = 0.0, 0.0, 0.0
        barra = tqdm(loader, desc=f"Epoca {epoca}/{n_epocas}", unit="batch", leave=False)
        for x_pequena, x_grande in barra:
            x_pequena = x_pequena.to(device)
            x_grande = x_grande.to(device)

            otim.zero_grad()
            x_rec, mu, logvar = modelo(x_pequena)
            perda, rec_loss, kl_loss = perda_vae(x_rec, x_grande, mu, logvar, beta_atual)
            perda.backward()
            otim.step()

            perda_total += perda.item() * x_pequena.size(0)
            rec_total += rec_loss.item() * x_pequena.size(0)
            kl_total += kl_loss.item() * x_pequena.size(0)
            barra.set_postfix(perda=f"{perda.item():.4f}")

        n = len(loader.dataset)
        historico.append(perda_total / n)
        print(
            f"epoca {epoca}: perda={perda_total / n:.4f} "
            f"(reconstrucao={rec_total / n:.4f}, kl={kl_total / n:.4f}, beta={beta_atual:.6f})"
        )

    return historico


# ==========================================
# VISUALIZACAO
# ==========================================

def mostrar_resultados(modelo, dataset, n=6):
    modelo.eval()
    idxs = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    fig, axs = plt.subplots(3, n, figsize=(2 * n, 6))
    with torch.no_grad():
        for col, idx in enumerate(idxs):
            x_pequena, x_grande = dataset[idx]
            x_in = x_pequena.unsqueeze(0).to(device)
            x_rec, _, _ = modelo(x_in)
            rec = x_rec.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            axs[0, col].imshow(x_pequena.numpy().transpose(1, 2, 0))
            axs[0, col].axis("off")
            axs[1, col].imshow(x_grande.numpy().transpose(1, 2, 0))
            axs[1, col].axis("off")
            axs[2, col].imshow(np.clip(rec, 0, 1))
            axs[2, col].axis("off")

    axs[0, 0].set_title("entrada 10x10", loc="left", fontsize=9)
    axs[1, 0].set_title("alvo 200x200", loc="left", fontsize=9)
    axs[2, 0].set_title("reconstruida (VAE)", loc="left", fontsize=9)
    plt.tight_layout()
    plt.savefig("vae_resultados.png", dpi=120)
    print("Figura salva em 'vae_resultados.png'")
    plt.show()


def gerar_a_partir_de_imagem_pequena(modelo, img_pequena_10x10):
    modelo.eval()
    with torch.no_grad():
        x = torch.from_numpy(img_pequena_10x10.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        x_rec, _, _ = modelo(x)
        return x_rec.squeeze(0).cpu().numpy().transpose(1, 2, 0)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("--- Montando dataset (pares 10x10 / 200x200) ---")
    pequenas, grandes = montar_dataset()
    dataset = ParesDataset(pequenas, grandes)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n--- Criando o VAE ---")
    modelo = VAEUpsampler(latent_dim=LATENT_DIM).to(device)
    print(modelo)

    print("\n--- Treinando ---")
    historico = treinar(modelo, loader)

    plt.figure(figsize=(6, 3))
    plt.plot(historico)
    plt.xlabel("epoca")
    plt.ylabel("perda (treino)")
    plt.title("Curva de treino do VAE")
    plt.tight_layout()
    plt.savefig("vae_curva_treino.png", dpi=120)
    print("Curva de treino salva em 'vae_curva_treino.png'")

    print("\n--- Visualizando resultados ---")
    mostrar_resultados(modelo, dataset)

    print("\n--- Salvando pesos do modelo ---")
    torch.save(modelo.state_dict(), "vae_upsampler.pt")
    print("Pesos salvos em 'vae_upsampler.pt'")

    # Exemplo de uso com uma imagem 10x10 gerada por outro processo
    # (ex.: a saida do seu script de difusao). Descomente e ajuste:
    #
    # img_gerada = ...  # array numpy (10, 10, 3) em [0, 1]
    # img_200x200 = gerar_a_partir_de_imagem_pequena(modelo, img_gerada)
    # plt.imshow(img_200x200); plt.axis("off"); plt.show()

# ==========================================
# EXTENSÃO: VISUALIZADOR INTERATIVO DO ESPAÇO LATENTE
# ==========================================

import json
import base64
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

def obter_vetor_latente(modelo, img_pequena_10x10):
    """
    Recebe uma imagem 10x10 RGB (numpy array no intervalo [0, 1]),
    passa pelo encoder e retorna mu, logvar e o vetor latente z.
    """
    modelo.eval()
    with torch.no_grad():
        # Prepara a imagem: ajusta os canais de (H, W, C) para (C, H, W) e adiciona dimensão de batch
        x = torch.from_numpy(img_pequena_10x10.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        mu, logvar = modelo.encoder(x)
        z = modelo.reparametrizar(mu, logvar)
        return mu.squeeze(0).cpu().numpy(), logvar.squeeze(0).cpu().numpy(), z.squeeze(0).cpu().numpy()


def executar_visualizador(modelo, z_inicial, porta=8080):
    """
    Gera a interface HTML/JS local e inicia um servidor HTTP mínimo 
    para decodificar o vetor latente em tempo real usando o PyTorch.
    """
    modelo.eval()
    
    # HTML completo com CSS e JavaScript incorporados (Single-page app limpa)
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>VAE Latent Space Explorer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            width: 100%;
            height: 100%;
        }}
        .sidebar {{
            width: 450px;
            background-color: #1e1e1e;
            border-right: 1px solid #333;
            display: flex;
            flex-direction: column;
            padding: 20px;
            box-sizing: border-box;
        }}
        .main-content {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #0a0a0a;
        }}
        h2 {{
            margin-top: 0;
            font-size: 1.4rem;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            color: #ffffff;
        }}
        .toolbar {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}
        button {{
            background-color: #2a2a2a;
            color: #fff;
            border: 1px solid #444;
            padding: 10px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }}
        button:hover {{
            background-color: #3a3a3a;
        }}
        button.danger {{
            background-color: #5a2a2a;
        }}
        button.danger:hover {{
            background-color: #7a3a3a;
        }}
        .sliders-container {{
            flex: 1;
            overflow-y: auto;
            padding-right: 10px;
        }}
        .slider-box {{
            margin-bottom: 12px;
            background: #252525;
            padding: 8px 12px;
            border-radius: 4px;
            
        }}
        .slider-header {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 4px;
            font-family: monospace;
        }}
        .slider-header span {{
            color: #aaa;
        }}
        input[type=range] {{
            width: 100%;
            margin: 0;
        }}
        .img-preview-box {{
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #333;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            text-align: center;
        }}
        #reconstructed-img {{
            max-width: 400px;
            max-height: 400px;
            width: 400px;
            height: 400px;
            display: block;
            background-color: #000;
            border: 1px solid #444;
            image-rendering: pixelated;
        }}
        .img-label {{
            margin-top: 10px;
            font-size: 0.9rem;
            color: #888;
        }}
        ::-webkit-scrollbar {{
            width: 6px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #444;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Latent Vector Controls</h2>
            <div class="toolbar">
                <button id="btn-reset" class="danger">Reset</button>
                <button id="btn-perturb">Perturbation</button>
                <button id="btn-export">Export Latent</button>
                <button id="btn-import">Import Latent</button>
            </div>
            <input type="file" id="file-input" style="display: none;" accept=".json">
            <div class="sliders-container" id="sliders-wrapper"></div>
        </div>
        <div class="main-content">
            <div class="img-preview-box">
                <img id="reconstructed-img" src="" alt="Gerando reconstrução...">
                <div class="img-label">Reconstrução do Decoder (200x200)</div>
            </div>
        </div>
    </div>

    <script>
        const LATENT_DIM = {len(z_inicial)};
        const originalZ = {list(z_inicial)};
        let currentZ = [...originalZ];
        let debounceTimeout = null;

        function gaussianRandom(mean = 0, stdev = 0.1) {{
            const u = 1 - Math.random();
            const v = 1 - Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            return z * stdev + mean;
        }}

        function inicializarSliders() {{
            const wrapper = document.getElementById('sliders-wrapper');
            wrapper.innerHTML = '';
            for (let i = 0; i < LATENT_DIM; i++) {{
                const box = document.createElement('div');
                box.className = 'slider-box';

                const header = document.createElement('div');
                header.className = 'slider-header';
                header.innerHTML = `<span>Dimensão ${{i}}</span><span id="val-${{i}}">${{currentZ[i].toFixed(4)}}</span>`;

                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = '-5';
                slider.max = '5';
                slider.step = '0.01';
                slider.value = currentZ[i];
                slider.id = `slider-${{i}}`;

                slider.addEventListener('input', (e) => {{
                    const val = parseFloat(e.target.value);
                    currentZ[i] = val;
                    document.getElementById(`val-${{i}}`).innerText = val.toFixed(4);
                    enviarReconstrucaoDebounce();
                }});

                box.appendChild(header);
                box.appendChild(slider);
                wrapper.appendChild(box);
            }}
        }}

        function atualizarUI() {{
            for (let i = 0; i < LATENT_DIM; i++) {{
                document.getElementById(`slider-${{i}}`).value = currentZ[i];
                document.getElementById(`val-${{i}}`).innerText = currentZ[i].toFixed(4);
            }}
            enviarReconstrucao();
        }}

        function enviarReconstrucaoDebounce() {{
            clearTimeout(debounceTimeout);
            debounceTimeout = setTimeout(enviarReconstrucao, 20);
        }}

        function enviarReconstrucao() {{
            fetch('/decode', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ z: currentZ }})
            }})
            .then(res => res.json())
            .then(data => {{
                document.getElementById('reconstructed-img').src = 'data:image/png;base64,' + data.image;
            }})
            .catch(err => console.error('Erro de rede:', err));
        }}

        // Eventos dos botões
        document.getElementById('btn-reset').addEventListener('click', () => {{
            currentZ = [...originalZ];
            atualizarUI();
        }});

        document.getElementById('btn-perturb').addEventListener('click', () => {{
            currentZ = currentZ.map(v => v + gaussianRandom(0, 0.15));
            atualizarUI();
        }});

        document.getElementById('btn-export').addEventListener('click', () => {{
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentZ));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", "latent_vector.json");
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        }});

        document.getElementById('btn-import').addEventListener('click', () => {{
            document.getElementById('file-input').click();
        }});

        document.getElementById('file-input').addEventListener('change', (e) => {{
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(evt) {{
                try {{
                    const parsed = JSON.parse(evt.target.result);
                    if (Array.isArray(parsed) && parsed.length === LATENT_DIM) {{
                        currentZ = parsed;
                        atualizarUI();
                    }} else {{
                        alert('Arquivo JSON inválido ou dimensão incorreta.');
                    }}
                }} catch (err) {{
                    alert('Erro ao ler o arquivo JSON.');
                }}
            }};
            reader.readAsText(file);
        }});

        // Inicialização
        inicializarSliders();
        enviarReconstrucao();
    </script>
</body>
</html>
"""

    class LatentExplorerHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return # Silencia os logs internos do servidor para não sujar o terminal
            
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == '/decode':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                z_array = data['z']
                
                # Executa o decoder do PyTorch usando os dados recebidos
                with torch.no_grad():
                    z_tensor = torch.tensor([z_array], dtype=torch.float32).to(device)
                    x_rec = modelo.decoder(z_tensor) # shape esperado: (1, 3, 200, 200)
                    
                    # Processamento para transformar em imagem válida
                    img_np = x_rec.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (200, 200, 3)
                    img_np = np.clip(img_np, 0.0, 1.0)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Converte de RGB para BGR para que o OpenCV codifique corretamente
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode('.png', img_bgr)
                    base64_str = base64.b64encode(buffer).decode('utf-8')
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = json.dumps({'image': base64_str})
                self.wfile.write(response.encode('utf-8'))

    def abrir_navegador():
        webbrowser.open(f"http://localhost:{porta}")

    server = HTTPServer(('localhost', porta), LatentExplorerHandler)
    print(f"\n=======================================================")
    print(f"Servidor iniciado em: http://localhost:{porta}")
    print(f"Explorador do Espaço Latente Ativo.")
    print(f"Pressione Ctrl+C no terminal para encerrar.")
    print(f"=======================================================\n")
    
    # Abre o navegador automaticamente em uma nova thread
    threading.Timer(1.0, abrir_navegador).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDesligando o visualizador...")
        server.server_close()


# Execução automática de teste ao rodar o script diretamente
if __name__ == "__main__":
    print("\n[Inicializando Teste do Visualizador Interativo]")
    
    # Garante que o modelo e o dataset existem na memória
    if 'modelo' in locals() and 'pequenas' in locals():
        # Captura a primeira imagem disponível do dataset original (10x10)
        img_teste = pequenas[0]
        
        # Executa os passos da extração latente pedida
        mu, logvar, z = obter_vetor_latente(modelo, img_teste)
        print(f"Vetor Latente 'z' extraído com sucesso. Dimensões: {z.shape}")
        
        # Dispara a interface gráfica local na porta 8080
        executar_visualizador(modelo, z, porta=8080)
