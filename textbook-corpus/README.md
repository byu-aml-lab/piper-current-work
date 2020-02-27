# textbook-corpus-scripts
The scripts for creating the textbook corpus as well as running synthetic tests on the corpus with ankura and the metadata map

get_textbook.py requires selenium web browser and will scrape all textbooks from OpenStax.org

cut_footer.py removes excess footer html from the raw scraped data

fragment.py creates individual HTML snippets of the paragraphs, questions, and glossary terms in the textbook data.

create_corpus.py creates an ankura corpus and classification file that needs to be copied to the .ankura folder (usually found in home directory after ankura is installed)

num_topics.py gives accuracies for the corpus using various numbers of topics and the maximum number of labeled documents.

presetdata_test.py finds accuracy on pre-labeled data beginning at 2 and doubling until it reaches the maximum number of possible pre-labeled documents

wsol.py finds accuracy on pre-labeled data using only the subset of questions that has a question and solution in the corpus beginning at 1 and doubling until it reaches the maximum number of possible pre-labeled documents.


This data requires specific forks of ankura and metadata-map to run:

ankura: https://github.com/piperarmstrong/ankura

metadata-map: https://github.com/piperarmstrong/metadatamap

