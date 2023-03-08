from utils.multiprocess_utils import main_process_print

_initialized = False
_ALL_CONSONANTS_SET = set()
_ALL_VOWELS_SET = set()
_g2p_dictionary = {
    'AP': ['AP'],
    'SP': ['SP']
}
_phoneme_list: list


def _build_dict_and_list():
    from utils.hparams import hparams
    global _g2p_dictionary, _phoneme_list

    _set = set()
    with open(hparams['g2p_dictionary'], 'r', encoding='utf8') as _df:
        _lines = _df.readlines()
    for _line in _lines:
        _pinyin, _ph_str = _line.strip().split('\t')
        _g2p_dictionary[_pinyin] = _ph_str.split()
    for _list in _g2p_dictionary.values():
        [_set.add(ph) for ph in _list]
    _phoneme_list = sorted(list(_set))
    main_process_print('| load phoneme set:', _phoneme_list)


def _initialize_consonants_and_vowels():
    # Currently we only support two-part consonant-vowel phoneme systems.
    for _ph_list in build_g2p_dictionary().values():
        _ph_count = len(_ph_list)
        if _ph_count == 0 or _ph_list[0] in ['AP', 'SP']:
            continue
        elif len(_ph_list) == 1:
            _ALL_VOWELS_SET.add(_ph_list[0])
        else:
            _ALL_CONSONANTS_SET.add(_ph_list[0])
            _ALL_VOWELS_SET.add(_ph_list[1])


def _initialize():
    global _initialized
    if not _initialized:
        _build_dict_and_list()
        _initialize_consonants_and_vowels()
        _initialized = True


def get_all_consonants():
    _initialize()
    return sorted(_ALL_CONSONANTS_SET)


def get_all_vowels():
    _initialize()
    return sorted(_ALL_VOWELS_SET)


def build_g2p_dictionary() -> dict:
    _initialize()
    return _g2p_dictionary


def build_phoneme_list() -> list:
    _initialize()
    return _phoneme_list
