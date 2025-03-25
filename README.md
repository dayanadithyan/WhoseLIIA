# Stylometric Analysis Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![CI Status](https://github.com/dayanadithyan/stylometry-dh/actions/workflows/python-ci.yml/badge.svg)

A Python package for analyzing writing style through quantitative metrics.

Please read: This is still in alpha. Analytics are fine but a ways to go before releasing a functional library.

Update 2/27: First features working at basic level
![image](https://github.com/user-attachments/assets/f826b33a-6858-4204-9e58-8a33d802001d)


Update 3/26: Revision to methodology. What worked, what didn't, and what next

# Stylometry Analysis Framework

Focusing now on grammatical structures rather than traditional word frequency analysis.

## Next steps (March, 2025)

When applied to Milton's Paradise Lost, the framework reveals:

1. Significantly higher use of periodic sentences compared to his prose work Areopagitica (19.0% vs 10.7%)
2. Strategic deployment of periodic style for narrative emphasis and thematic development
3. Variations in periodic sentence usage across different books, with notable decreases in Books 1 and 7
4. Presence of distinctive Latinate constructions such as ablative absolutes
5. Complex sentence structures that challenge both human readers and computational tools

## Next steps: Focus on how to identify biblical reference before coding. We'll start with Blood Meridian, difficult, but if done right, easier on Milton

## Proposed structure (WIP)

| **Biblical Reference Category** | **Sub-category**       | **Description**                                                              | **NLP Technique**                                                                                                       |
|-----------------------------------|------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **Direct Quotations**             | With attribution       | Explicitly identified as biblical                                            | Pattern Matching & Named Entity Recognition (NER)                                                                     |
|                                   | Without attribution    | Implicit biblical knowledge assumed                                          | Text Similarity Analysis                                                                                                |
| **Paraphrases**                   | Close paraphrase       | Maintaining most semantic content                                            | Semantic Similarity Metrics (e.g., cosine similarity on embeddings)                                                     |
|                                   | Loose paraphrase       | Preserving core meaning with significant reformulation                       | Paraphrase Detection Algorithms combined with Topic Modeling                                                          |
| **Allusions**                     | Nominal allusions      | References to biblical names, places, events                                 | Named Entity Recognition (NER) to extract proper nouns and link them to biblical contexts                               |
|                                   | Verbal allusions       | Echoes of distinctive biblical phrasing                                      | Keyphrase Extraction and Text Pattern Matching                                                                        |
|                                   | Thematic allusions     | Invocation of biblical themes or motifs                                      | Topic Modeling (e.g., LDA)                                                                                               |
| **Inversions**                    | Moral inversions       | Biblical moral principles deliberately reversed                              | Sentiment Analysis and Contrastive Learning                                                                            |
|                                   | Narrative inversions   | Biblical story patterns upended                                              | Sequence Modeling (e.g., RNNs or Transformers)                                                                         |
|                                   | Character inversions   | Biblical figures recast in antithetical roles                                | Entity Sentiment & Role Analysis                                                                                       |
| **Structural Adaptations**        | Syntactic mimicry      | Sentence structures that echo biblical cadences                              | Syntactic Parsing (using dependency or constituency parsers)                                                           |
|                                   | Rhetorical patterns    | Use of biblical rhetorical devices                                           | Discourse Analysis                                                                                                     |
|                                   | Narrative frameworks   | Episode structures that parallel biblical narratives                         | Narrative Segmentation and Sequence Analysis                                                                           |
| **Philosophical Engagement**      | Epistemological challenges | Questioning biblical claims to truth                                      | Argument Mining                                                                                                        |
|                                   | Ethical critiques      | Challenging biblical moral frameworks                                        | Sentiment and Stance Detection                                                                                         |
|                                   | Ontological explorations | Engaging with biblical concepts of being                                   | Concept Extraction and Ontology-Based Modeling                                                                         |
| **Apocalyptic Imagery**           | End-times references   | Imagery from Revelation and other apocalyptic texts                          | Image and Metaphor Detection using Semantic Role Labeling                                                              |
|                                   | Judgment imagery       | Scenes that evoke divine judgment                                            | Sentiment Analysis combined with Imagery Recognition                                                                   |
|                                   | Cosmic warfare         | Depictions that echo biblical spiritual conflict                             | Event Extraction and Thematic Analysis                                                                                 |
