import torch
import torch.nn as nn

# ==============================================================================
# 1. O ENCODER (O "Compressor")
# O trabalho do Encoder é olhar para a imagem pequena (10x10) e resumi-la
# em 128 números principais (o espaço latente). Ele não cospe 128 números diretos,
# mas sim as "médias" e "variâncias" para podermos gerar esses números.
# ==============================================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # self.conv é uma sequência de filtros de imagem (Convoluções)
        # Eles extraem texturas, bordas e cores da imagem.
        self.conv = nn.Sequential(
            # Entra 3 canais (RGB), sai 32 mapas de características
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            # Entra 32, sai 64 mapas de características
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            # Achata tudo num grande vetor linear (uma fila de números)
            nn.Flatten(),
        )
        
        # Como a imagem que você passa tem 10x10, após o flatten teremos 64 * 10 * 10 números.
        # Estas duas camadas finais pegam esse vetorzão e reduzem para 128 (latent_dim).
        # Uma gera o vetor de Médias (mu) e a outra o vetor de Desvio (logvar).
        self.fc_mu = nn.Linear(64 * 10 * 10, latent_dim)
        self.fc_logvar = nn.Linear(64 * 10 * 10, latent_dim)

    def forward(self, x):
        h = self.conv(x) # Passa a imagem pelas convoluções
        return self.fc_mu(h), self.fc_logvar(h) # Devolve os dois vetores paralelos


# ==============================================================================
# 2. O DECODER (O "Desenhista")
# O Decoder faz o processo inverso. Ele recebe 128 números (do espaço latente
# ou dos seus sliders na web) e tenta "descompactar" isso gerando texturas e
# detalhes de alta resolução (200x200).
# ==============================================================================
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Recebe os 128 números e expande de volta para o tamanho antes do achatamento
        self.fc = nn.Linear(latent_dim, 64 * 10 * 10)
        
        # Aqui ele vai inflando a imagem pouco a pouco até chegar em 200x200
        self.net = nn.Sequential(
            # Transforma a fila de números de volta num formato de imagem (Canais, Altura, Largura)
            nn.Unflatten(1, (64, 10, 10)),
            
            # ConvTranspose2d ("Deconvolução") estica a imagem para o dobro do tamanho (20x20)
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            # Estica para 40x40
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            
            # Upsample: Usa matemática normal (bilinear) para esticar de 40x40 para 50x50
            nn.Upsample(size=(50, 50), mode="bilinear", align_corners=False),
            # Refina a imagem no tamanho 50x50 sem mudar a resolução
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            
            # ConvTranspose2d estica de 50x50 para 100x100
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            # ConvTranspose2d final: estica de 100x100 para 200x200 e devolve para 3 canais (RGB)
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        return self.net(self.fc(z)) # Expande os 128 números e passa pelo desenhista


# ==============================================================================
# 3. O VAE COMPLETO (O "Gerente")
# Esta é a classe principal que une o Encoder e o Decoder e gerencia o fluxo
# da imagem do início ao fim.
# ==============================================================================
class VAEUpsampler(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    # O Truque da Reparametrização: A mágica que permite o modelo treinar!
    # A rede não pode gerar números puramente aleatórios, senão o gradiente (o erro)
    # não consegue fluir de volta no treinamento. Essa função pega a média (mu) e
    # adiciona um "ruído" controlado pela variância (logvar).
    def reparametrizar(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar) # Transforma log-variância em desvio padrão
        return mu + sigma * torch.randn_like(sigma) # mu + (desvio * aleatoriedade)

    def forward(self, x_pequena):
        # 1. Manda a imagem pequena (10x10) pro Encoder extrair os dados
        mu, logvar = self.encoder(x_pequena)
        
        # 2. Gera os 128 números do espaço latente (a variável 'z' que vai pro seu site)
        z = self.reparametrizar(mu, logvar)
        
        # 3. Manda o 'z' pro Decoder desenhar os DETALHES (o resíduo/texturas de alta resolução)
        delta = self.decoder(z)
        
        # 4. Pega a imagem pequena original (10x10) e estica para 200x200 
        # usando um borrado normal, sem rede neural. Fica apenas as "manchas" de cores.
        base = torch.nn.functional.interpolate(x_pequena, size=delta.shape[-2:], mode="bilinear", align_corners=False)
        
        # 5. O GRANDE TRUQUE: Soma as cores borradas (base) com as texturas nítidas (delta).
        # O torch.clamp garante que as cores finais não passem do limite máximo de brilho (1.0) ou fiquem escuras demais (0.0).
        return torch.clamp(base + delta, 0.0, 1.0), mu, logvar