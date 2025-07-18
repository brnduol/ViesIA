# ViesIA: Sistema de Diagnóstico de Viés

## Motivação
Com o avanço acelerado da inteligência artificial (IA) na era digital, os modelos de aprendizado profundo (Deep Learning) têm se tornado cada vez mais presentes em diversas aplicações do cotidiano. No entanto, esses modelos frequentemente reproduzem padrões e preconceitos presentes nos dados utilizados durante seu treinamento. Como esses dados são, em grande parte, gerados por seres humanos ou suas atividades, eles podem carregar vieses históricos e sociais, resultando em modelos que expressam comportamentos discriminatórios, como machismo, racismo e misoginia.

A identificação desses vieses pode ser realizada por meio da análise das predições dos modelos e da forma como eles respondem a diferentes tipos de entrada. A partir dessa analise, é possível inferir a presença de viés e avaliar seu impacto nas decisões automatizadas.

Diante desse cenário, este trabalho propõe o desenvolvimento de uma ferramenta para identificar e analisar vieses em modelos de Deep Learning, com o objetivo de sinalizar comportamentos discriminatórios e, assim, contribuir para a construção de sistemas mais justos e responsáveis.

## Métricas de Fairness
Para determinar e quantificar a existência de vieses nos modelos de Deep Learning, existem as chamadas métricas de fairness, ou métricas de justiça. Essas métricas geralmente buscam identificar se há algum grupo desprivilegiado que esteja sendo prejudicado pelo modelo (por exemplo, modelos de reconhecimento facial falhando ao identificar pessoas negras).

### Predictive Equality (FPR Diff)
Predictive Equality, ou diferença na taxa de falsos positivos (FPR), é uma métrica de fairness que verifica se um modelo comete o mesmo tipo de erro: prever positivo quando a verdade é negativa, em proporções semelhantes para diferentes grupos. Ela foca no impacto dos falsos positivos, avaliando possíveis prejuízos causados por decisões incorretamente favoráveis.

FPR diff = P(falso positivo | grupo desprivilegiado) - P(falso positivo | grupo privilegiado)

Ou equivalente, usando contagens:

FPR diff = ((falsos positivos | desprivilegiados) / (negativos reais | desprivilegiados)) - ((falsos positivos | privilegiados) / (negativos reais | desprivilegiados))

Um valor de FPR diff próximo de zero indica que o modelo erra do mesmo modo para ambos os grupos, enquanto valores positivos ou negativos apontam possíveis vieses, sugerindo revisão ou ajuste do modelo.

### Disparate Impact (DI)
Disparate Impact (ou Impacto Desproporcional) é uma métrica de justiça usada para avaliar se um modelo de machine learning está tratando grupos demográficos (como raça, gênero, etc.) de forma desigual, por meio da análise da distribuição de resultados positivos atribuídos pelo modelo a cada um desses grupos. Sua fórmula é:

DI = P(predição positiva | grupo desprivilegiado) / P(predição positiva | grupo privilegiado)

Os resultados desse cálculo indicam:
- DI = 1, indícios de justiça entre os grupos
- DI < 1, grupo desprivilegiado recebe menos resultados positivos
- DI > 1, grupo desprivilegiado recebe mais resultados positivos
- DI < 0.8, threshold comum utilizado para evidências legais de forte discriminação

### Statistical Parity Difference
A diferença de paridade estatística é uma métrica de avaliação de imparcialidade usada para verificar se um determinado modelo está produzindo resultados tendenciosos entre diferentes grupos, sendo definida por um grupo sensível e calculada pela fórmula:

SPD = P(resultado positivo | grupo desprivilegiado) - P(resultado positivo | grupo privilegiado)

Ou em termos de contagem de dados:

SPD = (resultado positivo / desprivilegiados) - (resultado positivo / privilegiados)

A SPD mede, portanto, a proporção de resultados favoráveis entre um grupo monitorado e um grupo de referência (grupo sensível).

## Estrutura do Projeto

```
├── data/                      # Arquivos de entrada (datasets)
│   ├── dataset.csv
│   └── teste.csv
├── project/
│   ├── functions/             # Funções auxiliares
│   │   ├── helpers.py
│   │   └── metrics.py
│   ├── templates/             # Templates HTML (se aplicável)
│   ├── app.py                 # Arquivo principal da aplicação Flask
│   └── models.py              # Modelos ou funções principais do projeto
├── .env                       # Variáveis de ambiente (não versionado)
├── .gitignore
├── README.md
└── requirements.txt
```

---


## Como executar o programa

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/viesia.git
cd viesia
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```


### Configuração de Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
APP_SECRET_KEY=424264
OPEN_AI_KEY=SuaChaveAqui
```

* `APP_SECRET_KEY`: Usado pelo Flask para segurança da sessão.
* `OPEN_AI_KEY`: Sua chave da API da OpenAI. [Obtenha aqui](https://platform.openai.com/account/api-keys)

---

### Executar a aplicação

Após configurar as variáveis de ambiente:

```bash
python3 project/app.py
```

A aplicação será iniciada localmente, normalmente em `http://127.0.0.1:5000`.

---

## Testes e Dados

Você pode usar os arquivos `dataset.csv` e `teste.csv` dentro da pasta `data/` como exemplos para rodar a aplicação e testar funcionalidades.

---
