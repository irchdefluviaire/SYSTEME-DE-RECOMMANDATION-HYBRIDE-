# Deploiement Google Cloud de KmerAI

Cette configuration lance cinq services sur une VM Ubuntu : Caddy, Streamlit,
FastAPI, PostgreSQL/pgvector et Neo4j. Seul le port HTTP 80 est public.

## Initialisation sur la VM

Depuis `/opt/kmerai` :

```bash
docker compose build
docker compose up -d postgres neo4j
docker compose --profile init run --rm init-neo4j
docker compose --profile init run --rm init-pgvector
docker compose up -d api ui caddy
```

Les deux commandes d'initialisation sont idempotentes. Ne lancez jamais
`load_neo4j.py --clear` sur une base a conserver.

## Verification

```bash
docker compose ps
curl -fsS http://127.0.0.1/api/health
curl -fsS http://127.0.0.1/_stcore/health
docker compose logs --tail=100 api ui
```

L'interface est disponible sur `http://IP_VM/`, Swagger sur
`http://IP_VM/docs` et l'API sous le prefixe `http://IP_VM/api/`.

## Exploitation

```bash
docker compose logs -f --tail=100
docker compose restart api ui
docker compose down
```

`docker compose down` conserve les volumes. N'ajoutez pas `-v` sauf si vous
voulez supprimer definitivement les donnees PostgreSQL et Neo4j.
