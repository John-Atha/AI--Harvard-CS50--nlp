import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """

S -> S Conj S | NP VP_OBJ | S Conj VP_OBJ

VP_OBJ -> VP | VP OBJ

VP -> V | V P | Adv VP | VP Adv

NP -> ADJN | Det ADJN | NP P NP

OBJ -> NP | NP Adv

ADJN -> N | Adj N | Adj ADJN

"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    # remove non-alphabetic chars and convert to lowercase
    def filter_word(word):
        return ''.join([char for char in word if char.isalpha()]).lower()

    # check if contains at least one alphabetic character
    def is_word(word):
        return any(char.isalpha() for char in word)

    res = [ filter_word(word) for word in sentence.split() if is_word(word) ]

    return res


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    res = []
    has_NP_children_dict = dict()

    def get_phrase(node):
        return ' '.join(node.leaves())

    def has_NP_children(node):
        children = [child for child in node.subtrees()]
        if not children:
            has_NP_children_dict[get_phrase(node)] = False
            return False
        res = any(child.label()=='NP' for child in children[1:])
        has_NP_children_dict[get_phrase(node)] = res
        return res

    def validate(node):
        phrase = get_phrase(node)
        if node.label()=='NP' and not has_NP_children(node):
            #print("I found phrase", phrase)
            res.append(node)
    
    def recur(subtree):
        for node in (subtree[1:] if tree!=subtree else subtree):
            phrase = get_phrase(node)
            # if subtree does not contain NP children, do not explore it
            if phrase in has_NP_children_dict.keys() and not has_NP_children_dict[phrase]:  
                continue
            validate(node)
            recur([child for child in node.subtrees()])

    recur(tree)
    return res


if __name__ == "__main__":
    main()
