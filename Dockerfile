# Image d'ENTRAÎNEMENT du projet churn (TP3).
#   docker build -t mlt-churn-train:0.1.0 .
#   docker run --rm -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/artifacts:/app/artifacts" mlt-churn-train:0.1.0
# Les données et les artefacts ne sont pas copiés dans l'image : ils sont montés.

# Image de départ : Python 3.12, variante slim
FROM python:3.12-slim

# Dossier de travail dans l'image (créé s'il n'existe pas)
WORKDIR /app

# D'abord les bibliothèques : cette étape reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install -r requirements.txt

# Ensuite le code du projet, puis le paquet churn lui-même
COPY . .
RUN pip install .

# Commande lancée au démarrage du conteneur
CMD ["python", "-m", "churn.train", "--config", "config/train.yaml"]
