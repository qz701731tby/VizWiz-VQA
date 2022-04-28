from itertools import islice

import pkg_resources
from symspellpy import SymSpell

sym_spell = SymSpell()
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, 0, 1)

# Print out first 5 elements to demonstrate that dictionary is
# successfully loaded
print(list(islice(sym_spell.words.items(), 5)))

input_term = (
    "arizona dicaming"
)
# max edit distance per lookup (per single word, not per whole input string)
suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=2)

for suggestion in suggestions:
    print(suggestion)