# 📄🔎 Ingestão e Busca Semântica com LangChain e PostgreSQL

Aplicação RAG em Python que lê `document.pdf`, divide seu texto em chunks, cria embeddings e os armazena no PostgreSQL com pgVector. Um chat no terminal recupera os 10 trechos semanticamente mais relevantes e responde somente com base neles.

Quando o contexto recuperado não contém a resposta, a aplicação instrui o modelo a retornar exatamente:

```text
Não tenho informações necessárias para responder sua pergunta.
```

## 🛠️ Tecnologias

- 🐍 Python 3.11 ou superior
- 🦜 LangChain
- 🐘 PostgreSQL 17 com pgVector
- 🐳 Docker e Docker Compose
- 🤖 OpenAI ou Google Gemini

Defaults por provedor:

| Provedor | Embeddings | Modelo de resposta |
| --- | --- | --- |
| OpenAI | `text-embedding-3-small` | `gpt-5-nano` |
| Gemini | `models/embedding-001` | `gemini-2.5-flash-lite` |

## 🗂️ Estrutura

```text
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf
├── src/
│   ├── config.py
│   ├── providers.py
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
└── tests/
```

## 1. 🐍 Preparar o ambiente Python

Na raiz do repositório:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

No Windows PowerShell, a ativação equivalente é:

```powershell
venv\Scripts\Activate.ps1
```

## 2. 🔐 Configurar as variáveis de ambiente

Copie o template:

```bash
cp .env.example .env
```

O `.env` não deve ser versionado. Escolha apenas um provedor.

### 🤖 OpenAI

```dotenv
AI_PROVIDER=openai
OPENAI_API_KEY=preencha-com-sua-chave
GOOGLE_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-5-nano
```

Crie a chave na plataforma da OpenAI. O modelo `gpt-5-nano` precisa estar disponível para o projeto associado à chave.

### ✨ Gemini

```dotenv
AI_PROVIDER=gemini
OPENAI_API_KEY=
GOOGLE_API_KEY=preencha-com-sua-chave
EMBEDDING_MODEL=models/embedding-001
LLM_MODEL=gemini-2.5-flash-lite
```

Crie a chave no Google AI Studio. Os limites e modelos disponíveis dependem da conta e podem mudar.

Se `EMBEDDING_MODEL` e `LLM_MODEL` ficarem vazios, a aplicação usa os defaults da tabela. As demais configurações locais já vêm preenchidas:

```dotenv
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PDF_PATH=document.pdf
COLLECTION_NAME=document_pdf
```

Caminhos relativos em `PDF_PATH` são resolvidos a partir da raiz do projeto.

## 3. 🐘 Subir PostgreSQL e pgVector

```bash
docker compose up -d
docker compose ps
```

O Compose inicia o PostgreSQL, aguarda o healthcheck e executa `CREATE EXTENSION IF NOT EXISTS vector`. Os dados ficam no volume nomeado `postgres_data`.

## 4. 📥 Ingerir o PDF

Coloque o arquivo desejado em `document.pdf` ou altere `PDF_PATH`, depois execute:

```bash
python src/ingest.py
```

Saída esperada:

```text
Ingestão concluída: <quantidade> chunks armazenados.
```

O PDF é dividido com `chunk_size=1000` e `chunk_overlap=150`. Executar a ingestão novamente substitui somente `COLLECTION_NAME`, evitando chunks duplicados e preservando outras coleções.

Sempre execute a ingestão novamente depois de alterar `AI_PROVIDER`, `EMBEDDING_MODEL`, `PDF_PATH` ou o conteúdo do PDF.

## 5. 💬 Executar o chat

```bash
python src/chat.py
```

Exemplo:

```text
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

A cada pergunta, a aplicação:

1. cria o embedding da pergunta;
2. executa `similarity_search_with_score(..., k=10)`;
3. concatena apenas o conteúdo dos resultados;
4. envia o contexto e a pergunta ao modelo com regras de resposta fundamentada.

Digite `sair`, `exit` ou `quit` para encerrar. `Ctrl+C` e `Ctrl+D` também encerram sem traceback.

## ✅ Testes

Os testes usam clientes falsos e não fazem chamadas a APIs nem ao PostgreSQL:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## 🧹 Encerrar e limpar o banco

Para parar os containers preservando os vetores:

```bash
docker compose down
```

Para apagar também o volume e todos os vetores locais:

```bash
docker compose down -v
```

O segundo comando é destrutivo para os dados locais do Compose; depois dele será necessário ingerir o PDF novamente.

## 🩺 Solução de problemas

- 🔑 **Chave obrigatória ausente:** confira `AI_PROVIDER` e preencha somente `OPENAI_API_KEY` ou `GOOGLE_API_KEY`, conforme o provedor.
- 🐘 **Conexão recusada pelo PostgreSQL:** execute `docker compose ps`, aguarde o serviço `postgres` ficar saudável e confirme que a porta 5432 está livre.
- 🔍 **Coleção vazia ou respostas sem contexto:** execute `python src/ingest.py` antes do chat e confirme que ingestão e busca usam o mesmo `.env`.
- 🔄 **Erro após trocar provedor/modelo de embeddings:** reexecute a ingestão; vetores criados por modelos diferentes não devem compartilhar a mesma coleção.
- 📄 **PDF sem texto:** o projeto não executa OCR. Use um PDF que contenha texto selecionável.
- ⏳ **Cota, acesso ao modelo ou limite de requisições:** confira a conta e a documentação oficial do provedor; a aplicação reporta a falha sem imprimir a chave.
- 🗑️ **Recriar o banco local:** use `docker compose down -v`, suba novamente e reexecute a ingestão, sabendo que isso apaga os vetores locais.

## ⚠️ Observação sobre respostas fundamentadas

O prompt restringe o modelo ao contexto recuperado e define a resposta-padrão fora do contexto. Como modelos generativos não oferecem uma garantia criptográfica de obediência ao prompt, valide perguntas representativas do documento antes de usar a solução em um cenário crítico.
