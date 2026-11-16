# Corpus Ingestion Pipeline

## Overview

The ingestion pipeline (`scripts/ingestion/`) builds the molecule corpus that the hybrid retrieval searches. It fetches small molecules from the ChEMBL API, standardizes them with RDKit, computes Morgan fingerprints and Uni-Mol embeddings, and inserts the results into PostgreSQL (`pgvector`) idempotently.

The pipeline runs offline, separately from the application, so requests at runtime search a local database instead of calling an external API for similar compounds.

## Data Flow

```mermaid
flowchart TD
    subgraph External["External Source"]
        ChEMBL["ChEMBL REST API<br/>(/molecule.json?molecule_type=Small+molecule)"]
    end

    subgraph Pipeline["Ingestion Pipeline (scripts/ingestion/)"]
        Fetch["fetch_molecule_pages()<br/>Async paging, 1,000 per page"]
        Std["standardize_smiles()<br/>RDKit Cleanup + canonical SMILES"]
        FP["morgan_fingerprint()<br/>2048-bit ECFP4 bitstring"]
        Embed["unimol_embeddings()<br/>Uni-Mol (512-d)"]
    end

    subgraph Database["PostgreSQL 17 + pgvector"]
        M["molecules<br/>(id, standardized_smiles, morgan_fingerprint)"]
        MS["molecule_sources<br/>(provider, molecule_id, source_identifier)"]
        ME["molecule_embeddings<br/>(id, molecule_id, embedding_model, embedding_model_version, embedding)"]
    end

    ChEMBL -->|"Paginated JSON"| Fetch
    Fetch -->|"Raw SMILES + ChEMBL IDs"| Std
    Std -->|"Standardized SMILES"| FP
    Std -->|"Standardized SMILES"| Embed

    FP -->|"Insert, on conflict do nothing"| M
    Std -->|"Map ChEMBL ID to molecule UUID"| MS
    Embed -->|"Insert missing embeddings"| ME
```

## Modules

| Module | Location | Responsibility |
|---|---|---|
| **Pipeline runner** | `scripts/ingestion/ingest.py` | Entry point, settings, per-page loading and commits, progress logging |
| **Paging** | `scripts/ingestion/fetch_molecules.py` | Async iterator over pages of ChEMBL small molecules |
| **Source clients** | `scripts/ingestion/source_data_models/` | `DataSource` (HTTP base class) and `ChEMBLSource` |
| **Standardization and fingerprints** | `scripts/ingestion/fingerprints.py` | RDKit `rdMolStandardize.Cleanup`, canonical isomeric SMILES, 2048-bit Morgan fingerprints (radius 2 / ECFP4) as bitstrings |
| **Uni-Mol embeddings** | `scripts/ingestion/embeddings.py` | 512-dimensional vectors from `UniMolRepr` (weights from `dptech/Uni-Mol-Models`) |

## Processing Steps

### 1. Paging (`fetch_molecules.py`)
- Requests `https://www.ebi.ac.uk/chembl/api/data/molecule.json` with `limit=1000`, ChEMBL's maximum page size.
- The request filters on the server: small molecules only, with a structure, returning only `molecule_chembl_id` and `molecule_structures`.
- Advances `offset` page by page and stops when `page_meta.next` is null or `max_molecules` is reached. The last page shrinks so the total never exceeds `max_molecules`.

### 2. Standardization (`fingerprints.py`)
- RDKit parses the ChEMBL canonical SMILES. Records it cannot parse are counted as `unparseable` and skipped.
- `rdMolStandardize.Cleanup` removes explicit hydrogens, disconnects metal atoms (`CC(=O)O[Na]` becomes `CC(=O)[O-].[Na+]`), normalizes functional groups (nitro `CN(=O)=O` becomes `C[N+](=O)[O-]`), and reionizes the molecule.
- The result is written out as canonical isomeric SMILES, which serves as the molecule's unique key.

> [!NOTE]
> `Cleanup` does **not** remove salts or solvents or keep only the largest fragment: sodium acetate stays `CC(=O)[O-].[Na+]`, so different salt forms of the same compound are stored as separate molecules.

### 3. Morgan Fingerprints (`fingerprints.py`)
- Radius 2 (equivalent to ECFP4), folded to 2048 bits.
- Stored in `molecules.morgan_fingerprint` (`bit(2048)`). Retrieval computes Tanimoto similarity as 1 − pgvector's Jaccard distance.

### 4. Uni-Mol Embeddings (`embeddings.py`)
- **Uni-Mol** is a transformer pretrained on 3D molecular structures. For each SMILES it returns a 512-dimensional vector (`np.ndarray` of shape `(N, 512)`).
- The weights are downloaded from Hugging Face (`dptech/Uni-Mol-Models`) on first use and cached locally.
- Embedding is the slowest step, so only molecules without an embedding for the current model and version (`unimol` / `unimolv1`) are embedded. Re-runs skip the rest.

### 5. Loading (`ingest.py`)
Each page is written in one SQLAlchemy session and committed on its own, using PostgreSQL `ON CONFLICT DO NOTHING`:
1. **`molecules`:** inserts `standardized_smiles` and `morgan_fingerprint`; the UUIDs of new and existing rows are then looked up by SMILES.
2. **`molecule_sources`:** links `molecule_id` to `provider = "ChEMBL"` and the ChEMBL ID as `source_identifier`.
3. **`molecule_embeddings`:** stores the 512-d vector with `embedding_model = "unimol"` and `embedding_model_version = "unimolv1"`.

## Running the Pipeline

### Prerequisites
The database must be running and migrated:

```bash
# 1. Start the database container
docker compose up db -d

# 2. Run the migrations
cd backend
uv run alembic upgrade head
cd ..
```

### Run
From the repository root:

```bash
uv run python -m scripts.ingestion.ingest
```

### Settings
Read from environment variables or the `.env` in the current working directory (the repository root):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@127.0.0.1:5433/molecules` | Target database |
| `MAX_MOLECULES` | `10000` | Maximum number of molecules fetched in one run |
| `PAGE_SIZE` | `1000` | Molecules per ChEMBL page |
| `EMBEDDING_MODEL_VERSION` | `unimolv1` | Uni-Mol checkpoint used for the embeddings |

> [!IMPORTANT]
> `.env.example` sets `DATABASE_URL` to the Docker hostname `db:5432`, which overrides the default above and does not resolve on the host. Switch it to the commented `127.0.0.1:5433` line before running ingestion.

### Resuming
The script can be stopped (Ctrl+C) and re-run at any point:
- Pages are committed one by one, so finished pages are kept.
- Molecules already in the corpus are skipped by the unique constraint on `standardized_smiles`.
- Existing `molecule_sources` rows are kept.
- Molecules still missing an embedding are embedded on the next run.
