import pickle
import sys
from pathlib import Path

# Run this program with two arguments, e.g. tag_text.py perceptron_smd_pos.pkl TekstZaEvaluacijuTagiranja.txt vertical
# Arguments are as following:
# pickled_tagger.name - name of the tagger to be unpickled and used for tagging
# Untagged_data.txt - text file with the tokens that should be tagged
# vertical|horizontal - vertical if the tokens are divided by the new line, horizontal if the separator is space ' '.

# take parameters
pickled_tagger_name = sys.argv[1]
input_file_path = sys.argv[2]
file_orientation = sys.argv[3]

# load data
perceptron_tagger = pickle.load(open(pickled_tagger_name, "rb"))
input_data = Path(input_file_path).read_text(encoding='utf-8-sig')
if file_orientation == 'vertical':
    tokens = input_data.splitlines()
else:
    tokens = input_data.split()

# tag
tagged_tokens = perceptron_tagger.tag(tokens)

# save the results
results = ''
for tt in tagged_tokens:
    results += "{}\t{}\n".format(tt[0], tt[1])

output_file_path = "{}_by_{}.txt".format(input_file_path, pickled_tagger_name)
Path(output_file_path).write_text(results, encoding='utf-8-sig')

print("The results saved successfully to {}".format(output_file_path))
