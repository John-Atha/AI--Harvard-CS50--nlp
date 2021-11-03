import nltk
import sys
import os
import string
import math

FILE_MATCHES = 3
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    
    filenames = [file for _, _, file in os.walk(os.path.join(".", directory))][0]

    for filename in filenames:
        with open(os.path.join(".", directory, filename)) as f:
            files[filename] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopwords = set(nltk.corpus.stopwords.words("english"))

    def filter_word(word):
        chars = [char for char in word.lower() if char not in string.punctuation]
        return ''.join(chars)

    return [filter_word(word) for word in nltk.word_tokenize(document) if word.isalpha() and word not in stopwords]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf = dict()
    docs_containing_word = dict()
    for doc in documents:
        for word in documents[doc]:
            if not docs_containing_word.get(word):
                docs_containing_word[word] = [doc]
            elif doc not in docs_containing_word[word]:
                docs_containing_word[word].append(doc)
    for word in docs_containing_word:
        idf[word] = math.log(len(documents)/len(docs_containing_word[word]))
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words),
            `files` (a dictionary mapping names of files to a list of their words), and
            `idfs` (a dictionary mapping words to their IDF values),
    return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    score = dict()
    ranking = []
    # compute the score of each file
    for file in files:
        tfs = dict()
        # compute the tfs for each word of the query at the current file
        for word in files[file]:
            if word not in query:
                continue
            elif not tfs.get(word):
                tfs[word] = 1
            else:
                tfs[word] += 1
        # sum the idf*tfs for each word of the query
        words = set.intersection(set(tfs.keys()), set(idfs.keys()), query)
        score[file] = sum([idfs[word]*tfs[word] for word in words])        
    
    ranking = [file for file in sorted(score, key=score.get, reverse=True)][:n]

    return ranking


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words),
            `sentences` (a dictionary mapping sentences to a list of their words), and
            `idfs` (a dictionary mapping words to their IDF values), 
    return 
        a list of the `n` top sentences that match the query, 
        ranked according to idf. 
        If there are ties, preference should be given to sentences that have a higher query term density.
    """
    score = dict()
    ranking = []

    # compute the score of each sentence
    for sentence in sentences:

        sentence_words = set(tokenize(sentence))
        words = set.intersection(sentence_words, set(idfs.keys()), query)

        idfs_sum = sum([idfs[word] for word in words])
        query_term_density = len(set.intersection(sentence_words, query)) / len(sentence_words)

        score[sentence] = (idfs_sum, query_term_density)

        ranking = [sentence for sentence in sorted(score, key=score.get, reverse=True)][:n]

    return ranking


if __name__ == "__main__":
    main()
