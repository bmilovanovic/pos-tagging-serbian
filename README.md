# pos-tagging-serbian
Results of the paper "Part-of-Speech Tagging for Serbian language using Natural Language Toolkit" by Boro Milovanović and Ranka Stanković.

This research is done at the University of Belgrade, doctoral study program "Intelligent systems" with the mentorship of Prof. Ranka Stanković and Prof. Cvetana Krstev.

Here is the description of the files:

- **train_taggers.py** - Code for the training and evaluation. Running this can produce other files with the tagger scores.
- **transform_talc_pos_to_ud_and_smd_pos.py** - Script that transforms unseen text tagged with unfamiliar tagset to one of the two tagsets used in the training.
- **tagged-data-example.txt** - First part of the data set. The whole training set is not yet publicly available. Training data used in this research is a part of collection that originated in the research of paper: *R. Stanković, B. Šandrih, C. Krstev, M. Utvić, and M. Škorić, “Machine Learning and Deep Neural Network-Based Lemmatization and Morphosyntactic Tagging for Serbian,” Proc. International Conference on Language Resources and Evaluation, pp. 3954‑3962, May 2020.*
- **unseen_text.txt** - Out-of-domain text used for the final evaluation of the best taggers.
- **perceptron.pkl** - Saved model of the top performing tagger.
- **tutorial/pos-tagging-serbian-tutorial.ipynb** - Tutorial for the tagging in the NLTK, in Serbian. I can translate it to English if someone happens to find it useful.
