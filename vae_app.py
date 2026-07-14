import torch
import json
import base64
import cv2
import requests
import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer
from vae_modelo import VAEUpsampler

# ==========================================
# CONFIGURAÇÕES INICIAIS
# ==========================================
CSV_PATH = "banco_de_dados_animes_limpo.csv"
MODEL_PATH = "vae_upsampler.pt"
PORT = 8080

device = torch.device("cpu")
modelo = VAEUpsampler()
try:
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    modelo.eval()
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Aviso: Não foi possível carregar {MODEL_PATH}. Erro: {e}")

# Carrega o banco de dados
df = pd.read_csv(CSV_PATH)

# Variável global para guardar a entrada base da sessão atual
estado_sessao = {
    "x_pequena_tensor": None
}

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def buscar_url_por_nome(query):
    """Busca o nome do anime em qualquer coluna de texto do CSV."""
    for col in df.select_dtypes(include=['object']).columns:
        mask = df[col].astype(str).str.contains(query, case=False, na=False)
        resultados = df[mask]
        if not resultados.empty:
            if 'Image_URL' in resultados.columns:
                return resultados.iloc[0]['Image_URL']
    return None

def baixar_e_preparar_imagem(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=5)
    img_arr = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_grande = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    img_pequena = cv2.resize(img, (10, 10), interpolation=cv2.INTER_AREA)
    
    grande_tensor = torch.from_numpy(img_grande.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    pequena_tensor = torch.from_numpy(img_pequena.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    
    return grande_tensor, pequena_tensor, img_grande, img_pequena

def tensor_para_base64(tensor):
    img_np = tensor.squeeze(0).numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0.0, 1.0)
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

# ==========================================
# SERVIDOR HTTP
# ==========================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Silencia logs no terminal
        
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open("interface.html", "r", encoding="utf-8") as f:
                self.wfile.write(f.read().encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        # --- NOVA ROTA: AUTOCOMPLETE ---
        if self.path == '/autocomplete':
            query = data.get('query', '')
            sugestoes = set()
            
            if query and len(query) >= 2:
                for col in df.select_dtypes(include=['object']).columns:
                    if 'url' in col.lower(): continue # Ignora colunas de link
                    
                    mask = df[col].astype(str).str.contains(query, case=False, na=False)
                    matches = df[mask][col].dropna().unique()
                    
                    for m in matches:
                        sugestoes.add(str(m))
                        if len(sugestoes) >= 10: break
                    if len(sugestoes) >= 10: break
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"sugestoes": list(sugestoes)[:10]}).encode())

        # --- ROTA DE BUSCA PRINCIPAL ---
        elif self.path == '/search':
            query = data.get('query', '')
            url = buscar_url_por_nome(query)
            
            if not url:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Anime não encontrado"}).encode())
                return

            try:
                grande_tensor, pequena_tensor, _, _ = baixar_e_preparar_imagem(url)
                estado_sessao["x_pequena_tensor"] = pequena_tensor

                with torch.no_grad():
                    mu, logvar = modelo.encoder(pequena_tensor)
                    z = modelo.reparametrizar(mu, logvar)
                    x_rec, _, _ = modelo(pequena_tensor)
                
                resposta = {
                    "img_original": tensor_para_base64(grande_tensor),
                    "img_pequena": tensor_para_base64(pequena_tensor),
                    "img_reconstruida": tensor_para_base64(x_rec),
                    "z": z.squeeze(0).numpy().tolist()
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(resposta).encode())
            
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        # --- ROTA DE DECODE DOS SLIDERS ---
        elif self.path == '/decode':
            z_array = data.get('z', [])
            if not z_array or estado_sessao["x_pequena_tensor"] is None:
                self.send_response(400)
                self.end_headers()
                return

            print("⚡ Sliders movidos! Calculando nova imagem...") # Avisa no terminal que conectou!

            with torch.no_grad():
                z_tensor = torch.tensor([z_array], dtype=torch.float32).to(device)
                
                # Gera o resíduo/detalhes
                delta = modelo.decoder(z_tensor)
                
                # O TRUQUE: Vamos multiplicar a força dos detalhes por 10 para FORÇAR a visualização!
                multiplicador = 10.0 
                delta_amplificado = delta * multiplicador

                base = torch.nn.functional.interpolate(
                    estado_sessao["x_pequena_tensor"], 
                    size=delta.shape[-2:], mode="bilinear", align_corners=False
                )
                
                # Soma a base com o delta amplificado
                x_rec = torch.clamp(base + delta_amplificado, 0.0, 1.0)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"img_reconstruida": tensor_para_base64(x_rec)}).encode())

if __name__ == "__main__":
    server = HTTPServer(('localhost', PORT), Handler)
    print(f"Servidor rodando em http://localhost:{PORT}")
    print("Aperte Ctrl+C para encerrar.")
    server.serve_forever()