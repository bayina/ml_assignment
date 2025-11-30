# # Assignment-04/nltk_assignment.py


# import os # for setting NLTK data directory
# import nltk # main NLTK package
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk import pos_tag

# # ---- CONFIG: custom data directory ----
# NLTK_DATA_DIR = "/Users/chandhini/nltk_data"

# # Ensure dir and force NLTK to use it
# os.makedirs(NLTK_DATA_DIR, exist_ok=True)
# os.environ["NLTK_DATA"] = NLTK_DATA_DIR
# nltk.data.path.clear()
# nltk.data.path.append(NLTK_DATA_DIR)

# # ---- Ensure resources (handles punkt_tab requirement) ----
# def ensure(res):
#     try:
#         # Map resource names to a representative file lookup
#         lookups = {
#             "punkt": "tokenizers/punkt",
#             "punkt_tab": "tokenizers/punkt_tab/english/",
#             "stopwords": "corpora/stopwords",
#             "wordnet": "corpora/wordnet",
#             "omw-1.4": "corpora/omw-1.4",
#             "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
#             "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
#         }
#         nltk.data.find(lookups[res])
#     except LookupError:
#         nltk.download(res, download_dir=NLTK_DATA_DIR, quiet=False)

# # Minimal required
# for r in [
#     "punkt",
#     "punkt_tab",  # new dependency for NLTK >=3.9
#     "stopwords",
#     "wordnet",
#     "omw-1.4",
#     "averaged_perceptron_tagger",        # older tagger name
#     "averaged_perceptron_tagger_eng",    # newer split name; keep both to be safe
# ]:
#     ensure(r)

# # ---- Pipeline ----
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words("english"))

# def get_wordnet_pos(tag):
#     if tag.startswith("J"): return wordnet.ADJ
#     if tag.startswith("V"): return wordnet.VERB
#     if tag.startswith("N"): return wordnet.NOUN
#     if tag.startswith("R"): return wordnet.ADV
#     return wordnet.NOUN

# def process_sentence(sentence):
#     tokens = [t.lower() for t in word_tokenize(sentence) if t.isalnum() and t.lower() not in stop_words]
#     tagged = pos_tag(tokens)
#     rows = []
#     for token, pos in tagged:
#         wn_pos = get_wordnet_pos(pos)
#         rows.append((
#             token,
#             stemmer.stem(token),
#             WordNetLemmatizer().lemmatize(token, wn_pos),
#             pos
#         ))
#     return rows

# sentences = [
#     "The duck will duck under the table.",
#     "I will book a room to read a book."
# ]

# for s in sentences:
#     print(f"\nSentence: {s}")
#     print(f"{'token':<12} {'stem':<12} {'lemma':<12} {'POS':<10}")
#     print("-" * 52)
#     for token, stem, lemma, pos in process_sentence(s):
#         print(f"{token:<12} {stem:<12} {lemma:<12} {pos:<10}")

# Assignment-03/nltk_assignment_table_notes_full.py
# Fills the "comment" column for EVERY row.
# Custom NLTK path + punkt_tab handling retained.

import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# --- CONFIG ---
NLTK_DATA_DIR = "/Users/chandhini/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
nltk.data.path.clear()
nltk.data.path.append(NLTK_DATA_DIR)

def ensure(res, lookup):
    try:
        nltk.data.find(lookup)
    except LookupError:
        nltk.download(res, download_dir=NLTK_DATA_DIR, quiet=False)

ensure("punkt", "tokenizers/punkt")
ensure("punkt_tab", "tokenizers/punkt_tab/english/")
ensure("stopwords", "corpora/stopwords")
ensure("wordnet", "corpora/wordnet")
ensure("omw-1.4", "corpora/omw-1.4")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    ensure("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def wn_pos(tag: str):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def comment(token: str, pos: str, stem: str, lemma: str) -> str:
    # Assignment-required ambiguity notes
    if token == "duck":
        n = WordNetLemmatizer().lemmatize(token, wordnet.NOUN)
        v = WordNetLemmatizer().lemmatize(token, wordnet.VERB)
        if pos.startswith("NN"):   return "‘duck’ as NOUN (animal); lemma same."
        if pos.startswith("VB"):   return "‘duck’ as VERB (lower one’s head/body)."
        # If tagger gives JJ or other, still record the noun-vs-verb impact
        return f"Ambiguous NOUN vs VERB; lemmas N='{n}', V='{v}'; tagger={pos}."
    if token == "book":
        n = WordNetLemmatizer().lemmatize(token, wordnet.NOUN)
        v = WordNetLemmatizer().lemmatize(token, wordnet.VERB)
        if pos.startswith("NN"):   return "‘book’ as NOUN (publication)."
        if pos.startswith("VB"):   return "‘book’ as VERB (to reserve)."
        return f"Ambiguous NOUN vs VERB; lemmas N='{n}', V='{v}'; tagger={pos}."

    # Task-specific illustrative note
    if token == "table":
        return 'Stem "tabl" is not a real word; lemma keeps dictionary form.'

    # POS-type notes to avoid blank comments
    if pos.startswith("NN"):       return "noun"
    if pos.startswith("VB"):       return "verb"
    if pos.startswith("JJ"):       return "adjective"
    if pos.startswith("RB"):       return "adverb"
    if pos == "PRP" and token == "i":
        return "pronoun kept as-is"

    # Fallback: highlight stem vs lemma difference if any
    if stem != lemma:
        return "Stem truncates; lemma preserves word form."

    return "—"

def process_sentence(sentence: str):
    tokens = [t.lower() for t in word_tokenize(sentence) if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words]
    tagged = pos_tag(tokens)
    rows = []
    for tok, p in tagged:
        lp = wn_pos(p)
        st = stemmer.stem(tok)
        lm = lemmatizer.lemmatize(tok, lp)
        rows.append((tok, st, lm, p, comment(tok, p, st, lm)))
    return rows

def print_table(title: str, rows):
    print(f"\nSentence: {title}")
    print(f"{'token':<12} {'stem':<12} {'lemma':<12} {'POS':<8} {'comment'}")
    print("-" * 110)
    for t, s, l, p, c in rows:
        print(f"{t:<12} {s:<12} {l:<12} {p:<8} {c}")

sentences = [
    "The duck will duck under the table.",
    "I will book a room to read a book."
]

for s in sentences:
    print_table(s, process_sentence(s))

