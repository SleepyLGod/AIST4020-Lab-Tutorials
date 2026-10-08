# Semantic Query Processing with DocETL

Start with the [document lab](semantic_query_processing.ipynb): inspect six company-report excerpts, try four semantic operators, combine them into a query, and explore query optimization with MOAR.

The first setup cell automatically downloads [document_lab_setup.zip](document_lab_setup.zip?raw=true) from a fixed Git commit and verifies its SHA-256. Students do not need to upload files. A matching local copy can also be used. The download and six-document loading have been verified in a fresh local directory; the latest notebook still needs a fresh Colab run.

The previous [movie-review lab](semantic_query_processing_movies.ipynb) is preserved byte-for-byte. Its GitHub download and [movie_lab_setup.zip](movie_lab_setup.zip?raw=true) are unchanged. The movie bundle contains its support code and the same data as `movie_lab_data.zip`.

Choose one model provider. DeepSeek is the default; Ollama, NVIDIA, and custom providers are optional. Model calls, MOAR search, and the separate Prompt Rewriting experiment have explicit switches, all off by default.

The [full reference](semantic_query_processing_full.ipynb) is the earlier, longer notebook. It is retained for comparison rather than as a second student entry.

The document bundle includes the upstream dataset card, original URLs, excerpt hashes, and source spans. The upstream collection declares CC-BY-SA-4.0. It also includes the classroom reference answers in `document_data/golden_answers.json` and the [scoring notes](document_scoring.md); these answers are used for scoring, not model inputs. Only short excerpts are included, not complete source documents. The unpacked `document_data/` directory stays Git-ignored. Movie-data provenance is documented in [DATA_SOURCES.md](DATA_SOURCES.md).

For maintainers, `prepare_data.py` reproduces the movie subset. `document_lab_support.py` reuses the provider setup in `lab_support.py`. Before running document tests in a fresh checkout, unpack the bundle's `document_data/` directory beside the notebook. Run `python -m unittest -q test_full_lab test_lab test_document_lab`; source checks require a DocETL 0.3.0 checkout. These tests do not make model requests. Rebuild the course bundle and update the notebook checksum together when its bundled files change.
