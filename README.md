## GTA V - Semantic Segmentation

Exemplo de segmentação semantica com imagens aéreas do jogo GTA, obtidas do repositório https://github.com/jiupinjia/gtav-sattellite-imagery-dataset

### Fluxo do Modelo

```text id="q4n8xd"
Imagem RGB
   ↓
Encoder (reduz resolução)
   ↓
Middle (aprende padrões)
   ↓
Decoder (reconstrói)
   ↓
Mapa segmentado
```

### Entrada da Rede

A entrada é uma imagem RGB:

```text id="t5c1yr"
(3 canais)
```

Exemplo:

```text id="j2v8pn"
256 x 256 x 3
```

Os 3 canais são:

* vermelho;
* verde;
* azul.

---

### Encoder (Downsampling)

Extrair características importantes.

O encoder aprende:

* bordas;
* texturas;
* formatos;
* regiões urbanas;
* prédios.

---

### Primeira Camada

```text id="r8k3ws"
down1.conv.0.weight
(64, 3, 3, 3)
```

---

### O que significa?

#### Estrutura

```text id="f1x7qp"
(64 filtros, 3 canais, 3x3)
```

#### Interpretação

A rede cria:

```text id="u9m4bc"
64 filtros convolucionais
```

Cada filtro:

* olha os 3 canais RGB;
* possui tamanho 3x3;
* aprende padrões visuais.

### Quantidade de parâmetros

```text id="n6q2zd"
1728 parâmetros
```

Cálculo:

64 \times 3 \times 3 \times 3 = 1728

### O que esses filtros aprendem?

Nos primeiros layers:

* bordas;
* contrastes;
* linhas;
* texturas simples.

Mais profundamente:

* telhados;
* ruas;
* regiões urbanas;
* formas complexas.

### Bias

```text id="h3p8yr"
down1.conv.0.bias
(64,)
```

Cada filtro possui:

```text id="w5z1mv"
1 bias
```

O bias ajuda no ajuste fino da ativação.

### Segunda Convolução

```text id="k7x4tn"
down1.conv.2.weight
(64, 64, 3, 3)
```

Agora:

* entram 64 mapas de características;
* saem 64 novos mapas.

A rede começa a combinar padrões aprendidos anteriormente.

### MaxPooling

Depois do bloco:

```text id="e4v9cq"
MaxPool2d(2)
```

A resolução é reduzida pela metade.

Exemplo:

```text id="p1f8rb"
256x256 -> 128x128
```

Isso:

* reduz custo computacional;
* aumenta visão global da imagem.

### Segundo Encoder

```text id="s6k2wm"
down2
```

Agora a rede aprende padrões mais complexos.

Quantidade de filtros aumenta:

```text id="d9r5hy"
64 -> 128
```

Mais filtros:

* mais capacidade;
* mais detalhes;
* mais parâmetros.

### Bottleneck (Middle)

#### Parte central da U-Net

```text id="m3x8ca"
middle.conv
```

Aqui a rede possui:

```text id="b7q1tv"
256 filtros
```

Ela aprende representações abstratas da imagem.

### Interpretação

Nesse ponto a rede já entende:

* “isso parece prédio”;
* “isso parece rua”;
* “isso parece vegetação”.

### Decoder (Upsampling)

Agora começa a reconstrução.

### UpSampling

```text id="r2n7pk"
up1.weight
(256, 128, 2, 2)
```

Isso é:

### ConvTranspose2d

Também chamado de:

* deconvolution;
* upsampling convolucional.

### Objetivo

Aumentar resolução:

```text id="f9v4lw"
64x64 -> 128x128
```

### Skip Connections

A U-Net possui:

```text id="u5q8dr"
torch.cat([u1, d2], dim=1)
```

Isso concatena:

* informações profundas;
* informações espaciais do encoder.

### Por que isso é importante?

Sem isso:

* a segmentação perderia detalhes;
* bordas ficariam ruins.

As skip connections ajudam a preservar:

* contornos;
* formas;
* localização dos objetos.

### Última Camada

```text id="h8w2cp"
final.weight
(2, 64, 1, 1)
```

### O que significa?

#### Convolução 1x1

Ela transforma:

```text id="k1r6zb"
64 features -> 2 classes
```

As classes são:

| Classe | Valor |
| ------ | ----- |
| Fundo  | 0     |
| Prédio | 1     |

### Saída Final

A saída da rede possui formato:

```text id="v4c9mq"
(2, altura, largura)
```

Cada pixel recebe:

* probabilidade de fundo;
* probabilidade de prédio.

### Predição

Depois você usa:

```python id="y7t3kw"
torch.argmax(...)
```

Isso escolhe:

```text id="z6m1pb"
a classe mais provável
```

para cada pixel.

### Total de Parâmetros

```text id="a2k8xr"
1.862.914 parâmetros
```

Isso significa que a rede possui:

```text id="q9v5fc"
quase 2 milhões de valores treináveis
```

### O que o treinamento faz?

Durante o treinamento:

* a rede faz predições;
* calcula erro;
* ajusta pesos usando backpropagation.

### Backpropagation

O algoritmo modifica os pesos para minimizar a loss.

A cada epoch:

```text id="j5c2nf"
pesos são refinados
```

para melhorar a segmentação.

### Papel da CrossEntropyLoss

```python id="m8q4ty"
nn.CrossEntropyLoss()
```

Ela compara:

* máscara real;
* máscara prevista.

### O que o modelo aprende no seu problema?

Como seu dataset possui prédios:

a rede aprende:

* padrões geométricos;
* telhados;
* regiões urbanas;
* contrastes entre construções e solo.

### Limitações atuais do modelo

* apenas 2 níveis encoder/decoder;
* sem BatchNorm;
* sem Dropout;
* sem Data Augmentation;
* sem métricas IoU/Dice.

### Melhorias possíveis

* Batch Normalization;
* Dropout;
* DeepLabV3;
* Attention U-Net;
* Data augmentation;
* Dice Loss;
* IoU metric;
* Transfer Learning.
