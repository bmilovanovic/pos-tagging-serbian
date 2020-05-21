import sys
from pathlib import Path


def map_talc_to_smd(token, tag):
    if token == 'se':
        return 'PAR'
    switcher = {
        'NOM:com': 'N',  # common noun   vrata ‘door’, kuća ‘house’, strast ‘passion’
        'NOM:col': 'N',  # collective noun   lišće ‘leaves’, pilad ‘chicks’, kamenje ‘stones’
        'NOM:nam': 'N',  # proper noun   Duško, Beograd ‘Belgrade’, Afrika ‘Africa’
        'NOM:num': 'N',  # numeral noun  dvojica ‘two men’, trojica ‘three men’
        'NOM:approx': 'N',  # approximate noun   desetak ‘about ten’, pedesetak ‘about fifty’
        'VER': 'V',  # main verb jedem ‘I eat’, radio ‘worked’
        'VER:aux': 'V',  # auxiliary verb sam ‘I am’, ćete ‘you will’
        'PRO:per': 'PRO',  # personal pronoun  ja ‘I’, mene ‘me’, ti ‘you’ (sg.), vi ‘you’ (pl.)
        'PRO:intr': 'PRO',  # interrogative pronoun    ko ‘who’, šta ‘what’
        'PRO:dem': 'PRO',  # demonstrative pronoun ovaj ‘this’ (m.sg.), ona ‘that’ (f.sg.), ti ‘those’ (m.pl.)
        'PRO:ind': 'PRO',  # indefinite pronoun    neko ‘somebody’, niko ‘nobody’, svako ‘everybody’
        'PRO:pos': 'PRO',  # possessive pronoun    moj ‘mine’, naši ‘ours’, njihovi ‘theirs’
        'PRO:rel': 'PRO',  # relative pronoun  koji ‘who/which/that’
        'PRO:ref': 'PRO',  # reflexive pronoun sebe, se ‘self’
        'PRO:num': 'NUM',  # numeral pronoun   jedan ‘one’, drugi ‘other’, ‘another’
        'ADJ': 'A',  # adjective in positive nov ‘new’, lepa ‘beautiful’
        'ADJ:comp': 'A',  # adjective in comparative noviji ‘newer’, lepša ‘more beautiful’
        'ADJ:sup': 'A',  # adjective in superlative  najnoviji ‘newest’, najlepša ‘the most beautiful’
        'ADJ:intr': 'A',  # interrogative adjective  which ‘lequel’, kakav ‘what’ (adj.), koliki ‘of what size’
        'ADJ:dem': 'A',  # demonstrative adjective ovaj ‘this’, ona ‘that’, ti ‘those’
        'ADJ:ind': 'A',  # indefinite adjective neki ‘some’, nijedan ‘no’, svaki ‘every’
        'ADJ:pos': 'A',  # possessive adjective moj ‘my’, naši ‘our’, njihovi ‘their’
        'ADJ:rel': 'A',  # relative adjective čiji ‘whose’, kakav ‘what’ (adj.), koliki ‘of the size’
        'NUM:car': 'NUM',  # cardinal number jedan ‘one’ (m.), jedna ‘one’ (f.), dvadeset ‘twenty’
        'NUM:ord': 'NUM',  # ordinal number prvi ‘first’, druga ‘second’, dvadeseti ‘twentieth’
        'NUM:col': 'NUM',  # collective number dvoje ‘two people’, petoro ‘five people’, dvadesetoro ‘twenty people’
        'ADV': 'ADV',  # adverb pametno ‘intelligently’, nespretno ‘clumsily’
        'ADV:comp': 'ADV',  # adverb in comparative bolje ‘better’, pametnije ‘smarter’
        'ADV:sup': 'ADV',  # adverb in superlative najbolje ‘best’, najpametnije ‘smartest’
        'ADV:intr': 'ADV',  # interrogative adverb kako ‘how’, gde ‘where’, kad ‘when’
        'ADV:rel': 'ADV',  # relative adverb kako ‘how’, gde ‘where’, kad ‘when’
        'ADV:ind': 'ADV',  # indefinite adverb nekako ‘in some way’, igde ‘anywhere’
        'CONJ:coor': 'CONJ',  # coordination conjunction i ‘and’, ali ‘but’, ili ‘or’
        'CONJ:sub': 'CONJ',  # subordination conjunction da ‘that’, jer ‘because’, iako ‘although’
        'PREP': 'PREP',  # preposition na ‘on’, pod ‘under’, u ‘in’
        'PAR': 'PAR',  # particle da ‘yes’, ne ‘no’, čak ‘even’
        'INT': 'INT',  # interjection ah ‘ah’, apćiha ‘achoo’, hej ‘hey’
        'SENT': 'SENT',  # strong punctuation . ! ?
        'PONC': 'PUNCT',  # weak punctuation , ; : ( )
        'PONC:cit': 'PUNCT',  # quotation marks « » " „ “
        'STR': 'X',  # foreign word chéri
        'ABR': 'X',  # abbreviation dr, itd. (etc.)
        'LET': 'X',  # letter A, p, L
        'NUM': 'NUM',  # litteral number 12, 252, XII
        'PAGE': 'X',  # page number 7, 10
        'ID': 'X',  # page number indicator @@
    }
    mapped = switcher[tag]
    if mapped is None:
        mapped = 'X'
    return mapped


def map_talc_to_ud(token, tag):
    if token == 'se':
        return 'PART'
    switcher = {
        'NOM:com': 'NOUN',  # common noun   vrata ‘door’, kuća ‘house’, strast ‘passion’
        'NOM:col': 'NOUN',  # collective noun   lišće ‘leaves’, pilad ‘chicks’, kamenje ‘stones’
        'NOM:nam': 'PROPN',  # proper noun   Duško, Beograd ‘Belgrade’, Afrika ‘Africa’
        'NOM:num': 'NOUN',  # numeral noun  dvojica ‘two men’, trojica ‘three men’
        'NOM:approx': 'NOUN',  # approximate noun   desetak ‘about ten’, pedesetak ‘about fifty’
        'VER': 'VERB',  # main verb jedem ‘I eat’, radio ‘worked’
        'VER:aux': 'AUX',  # auxiliary verb sam ‘I am’, ćete ‘you will’
        'PRO:per': 'PRON',  # personal pronoun  ja ‘I’, mene ‘me’, ti ‘you’ (sg.), vi ‘you’ (pl.)
        'PRO:intr': 'PRON',  # interrogative pronoun    ko ‘who’, šta ‘what’
        'PRO:dem': 'DET',  # demonstrative pronoun ovaj ‘this’ (m.sg.), ona ‘that’ (f.sg.), ti ‘those’ (m.pl.)
        'PRO:ind': 'DET',  # indefinite pronoun    neko ‘somebody’, niko ‘nobody’, svako ‘everybody’
        'PRO:pos': 'DET',  # possessive pronoun    moj ‘mine’, naši ‘ours’, njihovi ‘theirs’
        'PRO:rel': 'DET',  # relative pronoun  koji ‘who/which/that’
        'PRO:ref': 'PRON',  # reflexive pronoun sebe, se ‘self’
        'PRO:num': 'NUM',  # numeral pronoun   jedan ‘one’, drugi ‘other’, ‘another’
        'ADJ': 'ADJ',  # adjective in positive nov ‘new’, lepa ‘beautiful’
        'ADJ:comp': 'ADJ',  # adjective in comparative noviji ‘newer’, lepša ‘more beautiful’
        'ADJ:sup': 'ADJ',  # adjective in superlative  najnoviji ‘newest’, najlepša ‘the most beautiful’
        'ADJ:intr': 'ADJ',  # interrogative adjective  which ‘lequel’, kakav ‘what’ (adj.), koliki ‘of what size’
        'ADJ:dem': 'ADJ',  # demonstrative adjective ovaj ‘this’, ona ‘that’, ti ‘those’
        'ADJ:ind': 'ADJ',  # indefinite adjective neki ‘some’, nijedan ‘no’, svaki ‘every’
        'ADJ:pos': 'ADJ',  # possessive adjective moj ‘my’, naši ‘our’, njihovi ‘their’
        'ADJ:rel': 'ADJ',  # relative adjective čiji ‘whose’, kakav ‘what’ (adj.), koliki ‘of the size’
        'NUM:car': 'NUM',  # cardinal number jedan ‘one’ (m.), jedna ‘one’ (f.), dvadeset ‘twenty’
        'NUM:ord': 'NUM',  # ordinal number prvi ‘first’, druga ‘second’, dvadeseti ‘twentieth’
        'NUM:col': 'NUM',  # collective number dvoje ‘two people’, petoro ‘five people’, dvadesetoro ‘twenty people’
        'ADV': 'ADV',  # adverb pametno ‘intelligently’, nespretno ‘clumsily’
        'ADV:comp': 'ADV',  # adverb in comparative bolje ‘better’, pametnije ‘smarter’
        'ADV:sup': 'ADV',  # adverb in superlative najbolje ‘best’, najpametnije ‘smartest’
        'ADV:intr': 'ADV',  # interrogative adverb kako ‘how’, gde ‘where’, kad ‘when’
        'ADV:rel': 'ADV',  # relative adverb kako ‘how’, gde ‘where’, kad ‘when’
        'ADV:ind': 'ADV',  # indefinite adverb nekako ‘in some way’, igde ‘anywhere’
        'CONJ:coor': 'CCONJ',  # coordination conjunction i ‘and’, ali ‘but’, ili ‘or’
        'CONJ:sub': 'SCONJ',  # subordination conjunction da ‘that’, jer ‘because’, iako ‘although’
        'PREP': 'ADP',  # preposition na ‘on’, pod ‘under’, u ‘in’
        'PAR': 'PART',  # particle da ‘yes’, ne ‘no’, čak ‘even’
        'INT': 'INTJ',  # interjection ah ‘ah’, apćiha ‘achoo’, hej ‘hey’
        'SENT': 'PUNCT',  # strong punctuation . ! ?
        'PONC': 'PUNCT',  # weak punctuation , ; : ( )
        'PONC:cit': 'PUNCT',  # quotation marks « » " „ “
        'STR': 'X',  # foreign word chéri
        'ABR': 'X',  # abbreviation dr, itd. (etc.)
        'LET': 'X',  # letter A, p, L
        'NUM': 'NUM',  # litteral number 12, 252, XII
        'PAGE': 'X',  # page number 7, 10
        'ID': 'X',  # page number indicator @@
    }
    mapped = switcher[tag]
    if mapped is None:
        mapped = 'X'
    return mapped


def transform(input_file_path):
    original_lines = Path(input_file_path).read_text(encoding='utf-8-sig').strip().splitlines()

    transformed_text = ''
    sentence_number = 1
    for line in original_lines:
        (token, lemma, tag) = line.split()

        smd_tag = map_talc_to_smd(token, tag)
        ud_tag = map_talc_to_ud(token, tag)
        transformed_text += "{}\t{}\t{}\t{}\t{}\n".format(sentence_number, token, smd_tag, ud_tag, lemma)

        if tag == 'SENT':
            sentence_number += 1

    Path('unseen_text.txt').write_text(transformed_text, encoding='utf-8-sig')

    print('Transformation completed.')


if __name__ == '__main__':
    try:
        input_file_path = sys.argv[1]
        transform(input_file_path)
    except:
        print('Please provide path to the file that is to be transformed.')
