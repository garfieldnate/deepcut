# Format an ALT corpus for processing and generate the char maps
import sys
from collections import defaultdict


CHAR_TYPE = {
    # Brahmic 25 consonants
    u'ကခဂဃငစဆဇဈဉညဋဌဍဎဏတထဒဓနပဖဗဘမ': 'c',
    u'ဿ': 'c', # doubled ဘ

    u'ယရလဝသဟဠအ': 'c_m', # "miscellaneous" consonants

    u'ျြွှလ': 'd_mc',# medial consonant diacritics

    u'ဣဤဥဦဧဩဪ': 'v', # independent vowels

    # tone diacritics
    u'့': 'd_t', # creaky tone
    u'ာ': 'd_t', # low tone
    u'ါ': 'd_t', # alternative spelling for ာ
    u'း': 'd_t', # high tone

    u'်': 'virama', # kills inherent vowel
    u'္': 'stack', # indicates consonant stacking
    u'ငဉညနမကစတပ': 'final', # can appear in syllable-final position

    # vowels (with some tone changes)
    u'ေ': 'd_v', # changes vowel
    u'ဲ': 'd_v', # changes vowel, high tone
    u'ု': 'd_v', # changes vowel, high tone
    u'ူ': 'd_v', # changes vowel
    u'ိ': 'd_v', # changes vowel, creaky tone
    u'ီ': 'd_v', # changes vowel
    u'ံ': 'd_v', # nasalized final
    u'ၙ': 'd_v', # vocalic LL

    # Shan, Rumai, Mon and Sanskrit
    u'ၢ': 'f', # Sgaw Karen eu
    u'ႅ': 'f', # Shan e above
    u'ႄ': 'f', # Shan e
    u'ႍ': 'f', # Shan council emphatic tone
    u'ႏ': 'f', # rumai palaung tone-5
    u'ႀ': 'f', # Shan tha
    u'ၚ': 'f', # Mon nga
    u'ၻ': 'f', # Shan da
    u'ၴ': 'f', # kayah ee vowel
    u'ဳ': 'f', # Mon ii
    u'ၤ': 'f', # Sgaw Karen ke pho tone
    u'ဴ': 'f', # Mon o
    u'ၼ': 'f', # Shan na
    u'ၖ': 'f', # Sanskrit r̥
    u'ၗ': 'f', # Sanskrit r̥̄

    # sentence-related markers
    u'၊': 's', # period
    u'။': 's', # period
    u'၏': 's', # period after a verb
    u'၍': 's', # semicolon
    u'႟': 's', # Shan exclamation mark

    # words
    u'၌': 'w_at', # locative ('at')
    u'၎': 'w_ditto', # aforementioned

    # digits
    u'0123456789': 'd',
    u'၀၁၂၃၄၅၆၇၈၉': 'd',
    u'႐႑႒႓႔႕႖႗႘႙': 'd', # Shan

    # junk; required to prevent OOV during training
    '體聚新匯×窶º慧集智創・': 'j',

    # quotation marks
    u'"': 'q',
    u'“”': 'q',
    u'״': 'q', # someone misused this hebrew diacritic somewhere
    u'´': 'q',
    u"‘’": 'q',
    u"'": 'q',

    u' ': 'p',

    u'abcdefghijklmnopqrstuvwxyz': 's_e',
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}

CHAR_TYPE_FLATTEN = {}
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v

# create map of dictionary to character
CHARS = list(CHAR_TYPE_FLATTEN.keys())
CHARS += [
    u'\n', u'!', u'#', u'$', u'%', u'&', u'(', u')', u'*', u'+',
    u',', u'-', u'.', u'/',
    u':', u';', u'<', u'=', u'>', u'?', u'@',
    u'[', u'\\', u']', u'^', u'_',
    u'other', u'}', u'~', u'\ufeff',
    u'、', u'–', u'…', u'€', u'`', u'—', u'£'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}

CHAR_TYPES = set(CHAR_TYPE.values())
CHAR_TYPES.add('o')

CHAR_TYPES_MAP = {v: k for k, v in enumerate(CHAR_TYPES)}


def print_chars(alt_path):
    """Print the counts of all of the characters found in the given file"""
    chars = defaultdict(lambda: 0)
    with open(alt_path) as f:
        for line in f:
            # for whatever reason the raw Burmese file is already tokenized, so we have to remove spaces
            text = line.split('\t')[1].strip().replace(' ', '')
            for char in text:
                chars[char] += 1;
    char_diff = set(chars.keys()) - set(CHARS)
    if char_diff:
        print("Unknown chars found:")
        print(char_diff)

    print("All chars found:")
    print(chars)


if __name__ == '__main__':
    print_chars(sys.argv[1])
