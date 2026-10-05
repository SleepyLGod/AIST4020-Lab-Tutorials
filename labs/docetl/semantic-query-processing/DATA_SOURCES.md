# Movie-review teaching data

This subset contains 96 review excerpts: three disjoint splits of 32 reviews for the demo, optimization, and test. It was sampled from the SemBench movie scenario (`files/movie/data/sf_2000`) in our local SemBench-IVM checkout.

## Sources and license

- Benchmark and task context: [SemBench](https://github.com/SemBench/SemBench).
- Original dataset: [Clapper: Massive Rotten Tomatoes Movies & Reviews](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews), published by Andrea Villa on Kaggle.
- The [Kaggle public dataset API](https://www.kaggle.com/api/v1/datasets/list?search=clapper%20massive%20rotten%20tomatoes) lists the original dataset as **CC0: Public Domain**, checked on 2026-10-05. [CC0-1.0 terms](https://creativecommons.org/publicdomain/zero/1.0/) allow copying, adaptation, and redistribution within the scope of the rights waived by the publisher.

We distribute this teaching subset on the basis of that upstream declaration. We have not independently audited the rights of each review author or publisher. The CC0 declaration does not waive rights held by other parties. The excerpts are third-party material; this course does not claim authorship or endorsement by their authors, Kaggle, Rotten Tomatoes, or SemBench.

## Preparation and files

`prepare_data.py` samples records by a fixed hash ordering, checks duplicate IDs, decodes HTML entities, and separates model inputs from reference labels. It does not rewrite reviews with a model. `data/manifest.json` records the source CSV hashes, selected IDs, sampling rules, and output hashes.

`data/` and `movie_lab_data.zip` contain the same seven JSON files. The ZIP is a convenience for loading the data into Colab. Reference labels are for checking results and are not included in model prompts. The balanced or stratified teaching samples do not estimate either movie's overall reception.

Runtime logs, credentials, and executed notebook copies are not part of this data package.
