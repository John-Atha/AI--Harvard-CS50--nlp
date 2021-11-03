# Harvard CS50's Introduction to Aritificial Intelligence with Python 2021 course

## Project 6b - Questions

* An AI to answer questions using document and passage retrieval.

### Implementation

* The goal was to implement the `load_files`, `tokenize`, `compute_idfs`, `top_files`, `top_sentences` functions.
* For the document retrieval, made by the `top_files` function, we use tf-idf to find the most relative documents to the question.
* For the passage retrieval, made by the `top_sentences` function, we use a combination of idf and a query term density measure.
* The `FILE_MATCHES` constant specifies how many documents will be matched for a certain query.
* I observe that slightly increasing this parameter (for example from 1 to 3) improves the answer detection.
* The `SENTENCE_MATCHES` constant might seem helpful when we can not find only one proper answer, and more answers are needed to cover the question.
* I chose to keep it at 1, to answer the question with only one sentence.

- - -

* Developer: Giannis Athanasiou
* Github Username: John-Atha
* Email: giannisj3@gmail.com