# Homework 4 — Part C: Coding

## Overview
This assignment covers two Natural Language Processing (NLP) problems using Python and spaCy.

- **Q1:** Tokenization, stopword removal, lemmatization, and POS filtering.  
- **Q2:** Named Entity Recognition (NER) and pronoun ambiguity detection.

All code is implemented in [`homework-4.ipynb`](homework-4.ipynb) with inline comments explaining each step.

---

## Q1 — Tokenization, Stopword Removal & POS Filtering

### Objective
Perform the following steps on the given text using spaCy:
1. Segment into tokens  
2. Remove stopwords  
3. Apply lemmatization (not stemming)  
4. Keep only verbs and nouns (using POS tags)

### Input
```
"John enjoys playing football while Mary loves reading books in the library."
```

### Expected Output
A list of lemmas containing only nouns and verbs:
```
['John', 'enjoy', 'play', 'football', 'Mary', 'love', 'read', 'book', 'library']
```

### Implementation Highlights
- Uses `spacy.load("en_core_web_sm")` for the English language model.  
- Iterates through tokens and filters by:
  - `token.is_stop == False`
  - `token.pos_ in {"NOUN", "VERB", "PROPN"}`
- Outputs lemmatized tokens only.

---

## Q2 — NER & Pronoun Ambiguity Detection

### Objective
Perform Named Entity Recognition (NER) and detect possible pronoun ambiguity.

### Input
```
"Chris met Alex at Apple headquarters in California. He told him about the new iPhone launch."
```

### Tasks
1. Extract named entities and their types (e.g., PERSON, ORG, GPE).  
2. If the text contains any pronoun (`he`, `she`, `they`), print:
```
Warning: Possible pronoun ambiguity detected!
```

### Implementation Highlights
- Uses spaCy’s `doc.ents` for entity extraction.  
- Checks for pronouns by examining `token.lower_`.  
- Outputs both entity details and warnings if detected.

---

## Example Execution
```bash
$ python -q homework-4.ipynb
Entities: [('Chris','PERSON'),('Alex','PERSON'),('Apple','ORG'),('California','GPE'),('iPhone','PRODUCT')]
Warning: Possible pronoun ambiguity detected!
```

---

## Repository Structure
| File | Description |
|------|--------------|
| `homework-4.ipynb` | Jupyter notebook with solutions for Q1 and Q2 |
| `README.md` | Assignment documentation |
| `requirements.txt` | Dependencies list (spaCy and English model) |

---

## Dependencies
```
spacy
```
To install the English model:
```bash
python -m spacy download en_core_web_sm
```

---

## Notes
- All code cells are commented clearly.  
- The repository follows assignment submission rules and README guidelines.  
- Any updates after the deadline are counted as late submissions.
