# Étape 1 : Utiliser une image Python légère
FROM python:3.11-slim

# Étape 2 : Définir le dossier de travail
WORKDIR /app
ENV PYTHONPATH=/app

# Étape 3 : Installer les dépendances
COPY requirements.txt .
RUN echo "📦 Installation des dépendances..." \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Étape 4 : Copier tout le projet dans le conteneur
COPY . .

# Étape 5 : Exposer le port de l'API
EXPOSE 8000

# Étape 6 : Commande de lancement
CMD echo "🚀 Lancement de l'API FastAPI..." && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
