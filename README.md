# pos-tagging-serbian
Results of the paper "Part-of-Speech Tagging for Serbian language using Natural Language Toolkit" by Boro Milovanović, Ranka Stanković, and Cvetana Krstev

Here is the description of the files:

- **train_taggers.py** - Code for the training and evaluation. Running this can produce other scripts.
- **transform_talc_pos_to_ud_and_smd_pos.py** - Script that transforms unseen text tagged with unfamiliar tagset to one of the two tagsets used in the training.
- **tagged-data-example.txt** - First part of the data set. The whole training set is not yet publicly available.
- **unseen_text.txt** - Out-of-domain text used for the final evaluation of the best taggers.
- **perceptron.pkl** - Saved model of the top performing tagger.
- **tutorial/pos-tagging-serbian-tutorial.ipynb** - Tutorial for the tagging in the NLTK, in Serbian. I can translate it to English if someone happens to find it useful.