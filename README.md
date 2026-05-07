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

## Desafio

Use essa estrutura base e implemente melhorias. Compare os resultados usando a métrica IoU para ver qual melhoria teve mais impacto na segmentação. Sugiro testar cada melhoria isoladamente para entender seu efeito.

## 1. Data Augmentation (Augmentação de Dados)

* Use rotações leves, mudanças de brilho, contraste e espelhamento horizontal.
* Em segmentação, se você girar a imagem, **deve girar a máscara exatamente da mesma forma**.
* **Cuidado**: Isso pode piorar o modelo. Faça testes para ver se a augmentação está ajudando ou atrapalhando.

## 2. Métrica de Avaliação: IoU (Intersection over Union)
A `CrossEntropyLoss` é ótima para o computador otimizar, mas ela não diz muita coisa para nós, humanos. Em segmentação, a métrica de ouro é o **IoU** (também chamado de Jaccard Index).

Ele mede o quanto a sua predição se sobrepõe ao ground truth (máscara real).:

$$IoU = \frac{Area\ of\ Overlap}{Area\ of\ Union}$$

Se o seu IoU for **0.7** (70%) ou mais, seu modelo já está ficando muito bom. Se estiver abaixo de **0.5**, ele ainda está cometendo erros grosseiros de contorno.

## 3. Early Stopping e Validação

*   **Validação:** Durante o treino, a cada época, rode o modelo no dataset de teste (sem atualizar os pesos) para ver a "Loss de Validação". Com isso, você pode monitorar se o modelo está melhorando ou se já está começando a "decorar" os dados de treino (overfitting).
*   **Early Stopping:** Se a Loss de Validação parar de cair por 3 épocas seguidas, interrompa o treino automaticamente e salve a melhor versão dos pesos. Isso economiza tempo e hardware.

## 4. Lidar com o Desequilíbrio de Classes (Dice Loss)
Em imagens do GTA, o "fundo" (asfalto, céu, prédios) geralmente ocupa 90% da imagem, enquanto o objeto que você quer segmentar (ex: um pedestre ou um semáforo) ocupa apenas 10%. 

A `CrossEntropy` tende a focar na maioria. Se o modelo disser que "tudo é fundo", ele terá 90% de acerto, mas será um modelo inútil.
*   **Sugestão:** Tente usar uma função de perda chamada **Dice Loss** combinada com a CrossEntropy. Ela força a rede a dar importância para o acerto do objeto, não importa o quão pequeno ele seja na tela.

## 5. Learning Rate Scheduler
Começar com um Learning Rate de **0.001** é bom, mas conforme o modelo chega perto da solução ideal, esse "passo" pode ser grande demais, fazendo o modelo "saltar" sobre o ponto de erro mínimo.

*   **Solução:** Use um `ReduceLROnPlateau`. Ele monitora a perda e, se ela parar de cair, reduz o Learning Rate automaticamente (ex: de 0.001 para 0.0001) para fazer um ajuste fino.

### Exemplo de como ficaria a métrica IoU:

```python
def calculate_iou(pred, target, n_classes=2):
    # Calcula a interseção e união para cada classe
    ious = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(1, n_classes): # Ignora o fundo (classe 0) se preferir
        inter = ((pred == cls) & (target == cls)).sum().float().item()
        union = ((pred == cls) | (target == cls)).sum().float().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(inter / union)
    return np.nanmean(ious)
```

## 6. Transfer Learning

Treinar uma U-Net do zero é difícil, especialmente com poucos dados. Pode usar um ResNet ou EfficientNet como o "Encoder" da sua U-Net.

```python
import segmentation_models_pytorch as smp
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2)
```

## 7. Outras Arquiteturas

Teste outras arquiteturas além da U-Net para ver se consegue melhorar a segmentação. Sugestões:

* **Attention U-Net**: Adiciona "portões de atenção" nas Skip Connections. Isso faz a rede ignorar áreas irrelevantes da imagem e focar apenas no objeto de interesse.

* **DeepLabV3**: É o concorrente da U-Net. Em vez de focar apenas em alta resolução, ele usa Atrous Convolutions (convoluções dilatadas) para enxergar o contexto global da imagem sem perder detalhes. Para o GTA-V (cenários urbanos amplos), o DeepLabV3 costuma performar muito bem.


## Entrega

1. Implemente melhorias no modelo base (data augmentation, IoU, early stopping, etc).
2. Compare os resultados usando a métrica IoU.
3. Envie o código e um relatório breve explicando quais melhorias implementou e qual foi o impacto de cada uma no desempenho do modelo.
4. Inclua visualizações gráficas (loss, IoU, matriz de confusão, comparações, etc.) para ilustrar os resultados.