-- Passo 1: criar um .venv:

    python -m venv .venv

-- Passo 2: entrar no .venv:

    source .venv/bin/activate

-- Passo 3: instalar todas as dependências do projeto:

    pip install -r requirements.txt

-- Passo 4: toda vez que instalar uma dependência nova ao projeto, gere um novo arquivo requirements utilizando o comando abaixo:

    pip freeze > requirements.txt
