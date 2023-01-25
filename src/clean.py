from collections import Counter
import emoji
from lm_config import HELPER_CONTRACTIONS
import re
import unicodedata

class BaseClean:
    """
    Base cleaner object for tokenizers.
    A call to object takes input sentences (list) and returns them
    after rule-based pre-processing by list of CLEAN_FUNCS.
    """
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_us_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]

    def __init__(self, clean_fns=None):
        """
        Define the set of cleaning functions to be used.
        clean_funs: list(of str)
        """
        if clean_fns:
            self.clean_fns = clean_fns

    def __call__(self, input_texts):
        """
        Pre-process the sentence based on clean_fns
        input_texts: list(of str)
        """
        if type(input_texts) == list:
            for fn in self.clean_fns:
                fn = eval(fn)
                input_texts = fn(input_texts)

        elif type(input_texts) == str:
            input_texts = [input_texts]
            input_texts = self(input_texts)
            input_texts = input_texts[0]

        return input_texts

class DeBertaCleanV2(BaseClean):
    """
    Cleaner class for English.
    """
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_us_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]

class ESClean(BaseClean):
    """
    Cleaner class for Spanish.
    """
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "common_es_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]

def common_es_word(data):
    """
    Remove common noise from Spanish sentences.
    1. consistent '',” -> ",‘->'
    2. [0-9]'[0-9]" -> [0-9] pie [0-9] pulgada (pie=foot, pulgada=inch)
    """
    if type(data) == list:
        return [common_es_word(d) for d in data]
    else:
        text = data
        text = re.sub("''", '"', text)
        text = re.sub("”|“", '"', text)
        text = re.sub("‘|′", "'", text)
        exps = re.findall("[0-9] {0,1}'", text)
        for exp in exps:
            text = text.replace(exp, exp[0] + "pie")

        exps = re.findall('[0-9] {0,1}"', text)
        for exp in exps:
            text = text.replace(exp, exp.replace('"', "pulgada"))

        return text


class JSClean(BaseClean):
    """
    Cleaner class for Japanese.
    """
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]

def to_symbol(data):
    """
    Hex quotes to character quotes.
    &#34; -> ", &#39; -> '
    """
    if type(data) == list:
        return [to_symbol(d) for d in data]
    else:
        text = data
        text = re.sub("&#34;", '"', text)
        text = re.sub("&#39;", "'", text)
        return text
        
def to_lower(data):
    """
    Change the sentence to small-case letters.
    """
    data = list(map(lambda x: x.lower(), data))
    return data

def common_us_word(data):
    """
    Remove common noise from English sentences.
    1. consistent '',” -> ",‘->'
    2. a/c->ac, 0z->oz
    3. [0-9]'[0-9]" -> [0-9] foot [0-9] inch
    4. men' s->men
    """
    if type(data) == list:
        return [common_us_word(d) for d in data]
    else:
        text = data
        text = re.sub("''", '"', text)
        text = re.sub("a/c", "ac", text)
        text = re.sub("0z", "oz", text)
        text = re.sub("”|“", '"', text)
        text = re.sub("‘|′", "'", text)
        exps = re.findall("[0-9] {0,1}'", text)

        for exp in exps:
            text = text.replace(exp, exp[0] + "feet")
        exps = re.findall('[0-9] {0,1}"', text)

        for exp in exps:
            text = text.replace(exp, exp.replace('"', "inch"))

        text = re.sub("men'{0,1} {0,1}s|mens' s", "men", text)

        return text


def remove_emoji(data):
    """
    Remove emojis from the dataset.
    """
    if type(data) == list:
        return [remove_emoji(d) for d in data]
    elif type(data) == str:
        return emoji.get_emoji_regexp().sub("", data)
    else:
        raise

def query_clean_v1(data):
    """
    Cleaner function for queries in the dataset.
    """
    if type(data) == list:
        return [query_clean_v1(d) for d in data]

    elif type(data) == str:
        text = data
        product_ids = re.findall("b0[0-9a-z]{8}", text)
        if product_ids:
            for i, exp in enumerate(product_ids):
                text = text.replace(exp, f"placehold{chr(97+i)}")

        exps = re.findall("[a-zA-Z]'s|s'", text)
        for exp in exps:
            text = text.replace(exp, exp[0])

        text = re.sub("\(|\)|\*|---|\+|'|,|\[|\]| -|- |\. |/ |:", " ", text)  # ignore
        text = text.strip()

        exps = re.findall("[a-zA-Z]\.", text)
        for exp in exps:
            text = text.replace(exp, exp[0])

        # ! -> l for words
        exps = re.findall("![a-zA-Z]{2}", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("!", "l"))

        # a/b -> a b
        exps = re.findall("[a-zA-Z]/[a-zA-Z]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("/", " "))

        # remove "
        text = re.sub('"', " ", text)

        # remove "
        text = re.sub("'", " ", text)

        # # + [sep] + [num] -> # + [num]
        exps = re.findall("# {1}[0-9]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace(" ", ""))

        # remove # without
        exps = re.findall("#[a-zA-Z]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("#", ""))

        if product_ids:
            for i, exp in enumerate(product_ids):
                text = text.replace(f"placehold{chr(97+i)}", exp)

        text = text.strip()

        return text

def clean_contractions(data):
    """
    Modify contractions to the full words, e.g., we're -> we are
    Full list available in lm_config.HELPER_CONTRACTIONS
    """
    helper_contractions = HELPER_CONTRACTIONS
    if verbose:
        print("#" * 10, "Step - Contractions:")
    local_vocab = {}
    temp_vocab = _check_vocab(data, local_vocab, response="unknown_list")
    temp_vocab = [k for k in temp_vocab if (_check_replace(k)) and ("'" in k)]
    temp_dict = {}
    for word in temp_vocab:
        if word in helper_contractions:
            temp_dict[word] = helper_contractions[word]
    data = list(
        map(
            lambda x: " ".join([_make_dict_cleaning(i, temp_dict) for i in x.split()]),
            data,
        )
    )
    if verbose:
        _print_dict(temp_dict)
    return data

def remove_control_char(data):
    """
    Remove noise of control characters such ^F, ^C, ^?.
    Detected by unicodedata.category(c)[0] == "C", where c is the character.
    """
    global_chars_list = list(set([c for line in data for c in line]))
    chars_dict = {c: "" for c in global_chars_list if unicodedata.category(c)[0] == "C"}
    data = list(
        map(
            lambda x: " ".join([_make_cleaning(i, chars_dict) for i in x.split()]), data
        )
    )

    return data

### Private Library variables and functions.
verbose = False
global_lower = True
WPLACEHOLDER = "[PLS]"

def _check_replace(w):
    return not bool(re.search(WPLACEHOLDER, w))


def _make_cleaning(s, c_dict):
    if _check_replace(s):
        s = s.translate(c_dict)
    return s


def _check_vocab(c_list, vocabulary, response="default"):
    try:
        words = set([w for line in c_list for w in line.split()])
        # print('Total Words :',len(words))
        u_list = words.difference(set(vocabulary))
        k_list = words.difference(u_list)

        if response == "default":
            print("Unknown words:", len(u_list), "| Known words:", len(k_list))
        elif response == "unknown_list":
            return list(u_list)
        elif response == "known_list":
            return list(k_list)
    except:
        return []


def _make_dict_cleaning(s, w_dict):
    if _check_replace(s):
        s = w_dict.get(s, s)
    return s


def _print_dict(temp_dict, n_items=10):
    run = 0
    for k, v in temp_dict.items():
        print(k, "---", v)
        run += 1
        if run == n_items:
            break

def remove_duplicate(data):
    # Duplicated dots, question marks and exclamations
    # Locallocal_vocab
    if verbose:
        print("#" * 10, "Step - Duplicated Chars:")
    local_vocab = {}
    temp_vocab = _check_vocab(data, local_vocab, response="unknown_list")
    temp_vocab = [k for k in temp_vocab if _check_replace(k)]
    temp_dict = {}

    for word in temp_vocab:
        new_word = word
        if (
            (Counter(word)["."] > 1)
            or (Counter(word)["!"] > 1)
            or (Counter(word)["?"] > 1)
            or (Counter(word)[","] > 1)
        ):
            if Counter(word)["."] > 1:
                new_word = re.sub("\.\.+", " . . . ", new_word)
            if Counter(word)["!"] > 1:
                new_word = re.sub("\!\!+", " ! ! ! ", new_word)
            if Counter(word)["?"] > 1:
                new_word = re.sub("\?\?+", " ? ? ? ", new_word)
            if Counter(word)[","] > 1:
                new_word = re.sub("\,\,+", " , , , ", new_word)
            temp_dict[word] = new_word

    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = list(
        map(
            lambda x: " ".join([_make_dict_cleaning(i, temp_dict) for i in x.split()]),
            data,
        )
    )
    return data


def remove_ending_underscore(data):
    if verbose:
        print("#" * 10, "Step - Remove ending underscore:")
    local_vocab = {}
    temp_vocab = _check_vocab(data, local_vocab, response="unknown_list")
    temp_vocab = [k for k in temp_vocab if (_check_replace(k)) and ("_" in k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if word[len(word) - 1] == "_":
            for i in range(len(word), 0, -1):
                if word[i - 1] != "_":
                    new_word = word[:i]
                    temp_dict[word] = new_word
                    break
    data = list(
        map(
            lambda x: " ".join([_make_dict_cleaning(i, temp_dict) for i in x.split()]),
            data,
        )
    )
    if verbose:
        _print_dict(temp_dict)
    return data


def remove_starting_underscore(data):
    if verbose:
        print("#" * 10, "Step - Remove starting underscore:")
    local_vocab = {}
    temp_vocab = _check_vocab(data, local_vocab, response="unknown_list")
    temp_vocab = [k for k in temp_vocab if (_check_replace(k)) and ("_" in k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if word[len(word) - 1] == "_":
            for i in range(len(word)):
                if word[i] != "_":
                    new_word = word[:i]
                    temp_dict[word] = new_word
                    break
    data = list(
        map(
            lambda x: " ".join([_make_dict_cleaning(i, temp_dict) for i in x.split()]),
            data,
        )
    )
    if verbose:
        _print_dict(temp_dict)
    return data

def clean_multiple_form(data):
    if verbose:
        print("#" * 10, "Step - Multiple form:")
    local_vocab = {}
    temp_vocab = _check_vocab(data, local_vocab, response="unknown_list")
    temp_vocab = [k for k in temp_vocab if (k[-1:] == "s") and (len(k) > 4)]
    temp_dict = {k: k[:-1] for k in temp_vocab if (k[:-1] in local_vocab)}
    data = list(
        map(
            lambda x: " ".join([_make_dict_cleaning(i, temp_dict) for i in x.split()]),
            data,
        )
    )
    if verbose:
        _print_dict(temp_dict)
    return data

def leet_clean(data):
    def __convert_leet(word):
        # basic conversion
        word = re.sub("0", "o", word)
        word = re.sub("1", "i", word)
        word = re.sub("3", "e", word)
        word = re.sub("\$", "s", word)
        word = re.sub("\@", "a", word)
        return word

    if verbose:
        print("#" * 10, "Step - L33T (with vocab check):")
    local_vocab = {}
    temp_vocab = _check_vocab(data, local_vocab, response="unknown_list")
    temp_vocab = [k for k in temp_vocab if _check_replace(k)]

    temp_dict = {}
    for word in temp_vocab:
        new_word = __convert_leet(word)

        if new_word != word:
            if (len(word) > 2) and (new_word in local_vocab):
                temp_dict[word] = new_word

    data = list(
        map(
            lambda x: " ".join([_make_dict_cleaning(i, temp_dict) for i in x.split()]),
            data,
        )
    )
    if verbose:
        _print_dict(temp_dict)
    return data