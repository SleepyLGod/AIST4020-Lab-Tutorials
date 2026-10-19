# Semantic Query Processing with DocETL

Start with the [student notebook](semantic_query_processing.ipynb): read six restaurant reviews, try four semantic operators, and compare the original query with two real MOAR rewrites. Running the optimizer yourself is optional.

Open the [notebook in Colab](https://colab.research.google.com/github/SleepyLGod/AIST4020-Lab-Tutorials/blob/main/labs/docetl/semantic-query-processing/semantic_query_processing.ipynb). The folded setup cell automatically downloads the course bundle from a fixed Git commit and checks its SHA-256. No upload is needed. Locally, an existing matching `textbook_lab_setup.zip` is reused. This version has not yet passed a fresh end-to-end Colab run.

The previous [company-report lab](semantic_query_processing_company_reports.ipynb) is preserved byte-for-byte under its new filename. Its fixed download of `document_lab_setup.zip` is unchanged.

The previous [movie-review lab](semantic_query_processing_movies.ipynb) is preserved byte-for-byte. Its GitHub download and [movie_lab_setup.zip](movie_lab_setup.zip?raw=true) are unchanged. The movie bundle contains its support code and the same data as `movie_lab_data.zip`.

Choose one backend in the notebook. Model calls and optional live MOAR search are off by default. The saved fusion and prompt rewrites can be executed without running a search.

The [full reference](semantic_query_processing_full.ipynb) is the earlier, longer notebook. It is retained for comparison rather than as a second student entry.

The restaurant-review source and selection are recorded in `textbook_data/provenance.json`; reference decisions are in `textbook_data/reference.json`. Saved MOAR evidence and validation limits are in `textbook_saved_run/`. Review redistribution terms still need verification before publishing the new bundle. The older company-report reference and [scoring notes](document_scoring.md) remain available; movie-data provenance is in [DATA_SOURCES.md](DATA_SOURCES.md).

For maintainers, run `python -m unittest -q test_textbook_lab` for the active notebook. Run `test_full_lab`, `test_lab`, and `test_document_lab` for the preserved versions; company-report tests require unpacking `document_data/` from its bundle, and source checks require a DocETL 0.3.0 checkout. These tests make no model requests. Rebuild a bundle and update its notebook checksum together when bundled files change.
