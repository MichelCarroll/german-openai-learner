from enum import Enum

class PartOfSpeech(str, Enum):
    ADJ = 'ADJ'
    ADP = 'ADP'
    ADV = 'ADV'
    AUX = 'AUX'
    CONJ = 'CONJ'
    CCONJ = 'CCONJ'
    DET = 'DET'
    INTJ = 'INTJ'
    NOUN = 'NOUN'
    NUM = 'NUM'
    PART = 'PART'
    PRON = 'PRON'
    PROPN = 'PROPN'
    PUNCT = 'PUNCT'
    SCONJ = 'SCONJ'
    SYM = 'SYM'
    VERB = 'VERB'
    X = 'X'
    SPACE = 'SPACE'

# ADJ: adjective, e.g. big, old, green, incomprehensible, first
# ADP: adposition, e.g. in, to, during
# ADV: adverb, e.g. very, tomorrow, down, where, there
# AUX: auxiliary, e.g. is, has (done), will (do), should (do)
# CONJ: conjunction, e.g. and, or, but
# CCONJ: coordinating conjunction, e.g. and, or, but
# DET: determiner, e.g. a, an, the
# INTJ: interjection, e.g. psst, ouch, bravo, hello
# NOUN: noun, e.g. girl, cat, tree, air, beauty
# NUM: numeral, e.g. 1, 2017, one, seventy-seven, IV, MMXIV
# PART: particle, e.g. ’s, not,
# PRON: pronoun, e.g I, you, he, she, myself, themselves, somebody
# PROPN: proper noun, e.g. Mary, John, London, NATO, HBO
# PUNCT: punctuation, e.g. ., (, ), ?
# SCONJ: subordinating conjunction, e.g. if, while, that
# SYM: symbol, e.g. $, %, §, ©, +, −, ×, ÷, =, :), 😝
# VERB: verb, e.g. run, runs, running, eat, ate, eating
# X: other, e.g. sfpksdpsxmsa
# SPACE: space, e.g.