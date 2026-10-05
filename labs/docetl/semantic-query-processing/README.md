# Semantic Query Processing with DocETL

Start with [semantic_query_processing.ipynb](semantic_query_processing.ipynb). The first setup cell downloads and checks the course files from GitHub when needed. For offline local use, keep [movie_lab_setup.zip](movie_lab_setup.zip?raw=true) beside the notebook. It contains the support code and the same data as `movie_lab_data.zip`.

Choose one model provider in the notebook. Queries and MOAR search have separate switches so that running setup does not start paid requests.

The [full reference](semantic_query_processing_full.ipynb) is the earlier, longer notebook. It is retained for comparison rather than as a second student entry.

Data provenance, sampling, and the upstream license declaration are documented in [DATA_SOURCES.md](DATA_SOURCES.md).

For maintainers, `prepare_data.py` reproduces the teaching subset from the source CSVs. `lab_support.py` contains loading, validation, and recording helpers. Run offline checks with `python -m unittest -q test_full_lab test_lab`; they require the notebook dependencies and a local DocETL 0.3.0 source checkout for source checks. These tests do not make model requests.
