import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
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
    dictionary = {}

    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf-8") as ofile:
            dictionary[file] = ofile.read()

    return dictionary


def tokenize(document):

    tokenized = nltk.tokenize.word_tokenize(document.lower())

    final_list = [x for x in tokenized if x not in string.punctuation and x not in nltk.corpus.stopwords.words("english")]

    return final_list


def compute_idfs(documents):
    idf_dictio = {}
    doc_len = len(documents)

    unique_words = set(sum(documents.values(), []))

    for word in unique_words:
        count = 0
        for doc in documents.values():
            if word in doc:
                count += 1

        idf_dictio[word] = math.log(doc_len/count)

    return idf_dictio


def top_files(query, files, idfs, n):
    scores = {}
    for filename, filecontent in files.items():
        file_score = 0
        for word in query:
            if word in filecontent:
                file_score += filecontent.count(word) * idfs[word]
        if file_score != 0:
            scores[filename] = file_score

    sorted_by_score = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_by_score[:n]


def top_sentences(query, sentences, idfs, n):
    scores = {}
    for sentence, sentwords in sentences.items():
        score = 0
        for word in query:
            if word in sentwords:
                score += idfs[word]

        if score != 0:
            density = sum([sentwords.count(x) for x in query]) / len(sentwords)
            scores[sentence] = (score, density)

    sorted_by_score = [k for k, v in sorted(scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)]

    return sorted_by_score[:n]



if __name__ == "__main__":
    main()
