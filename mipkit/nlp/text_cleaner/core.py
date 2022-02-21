"""
 The MIT License (MIT)
 Copyright (c) 2021 Cong Vo
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 Provided license texts might have their own copyrights and restrictions
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
"""

import logging
import re
import sys
from unicodedata import category
from bs4 import BeautifulSoup
from emoji import UNICODE_EMOJI, demojize, emojize
from ftfy import fix_text

from . import constants
from .specials import save_replace

log = logging.getLogger()

# fall back to `unicodedata`
try:
    from unidecode import unidecode

except:
    from unicodedata import normalize

    def unidecode(x):
        return normalize("NFD", x).encode("ASCII", "ignore").decode("utf-8")

    log.warning(
        "Since the GPL-licensed package `unidecode` is not installed, using Python's `unicodedata` package which yields worse results."
    )


def fix_strange_quotes(text):
    """
    Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
    """
    text = constants.SINGLE_QUOTE_REGEX.sub("'", text)
    text = constants.DOUBLE_QUOTE_REGEX.sub('"', text)
    return text


def fix_bad_unicode(text, normalization="NFC"):
    """
    Fix unicode text that's "broken" using `ftfy <http://ftfy.readthedocs.org/>`_;
    this includes mojibake, HTML entities and other code cruft,
    and non-standard forms for display purposes.
    Args:
        text (str): raw text
        normalization ({'NFC', 'NFKC', 'NFD', 'NFKD'}): if 'NFC',
            combines characters and diacritics written using separate code points,
            e.g. converting "e" plus an acute accent modifier into "é"; unicode
            can be converted to NFC form without any change in its meaning!
            if 'NFKC', additional normalizations are applied that can change
            the meanings of characters, e.g. ellipsis characters will be replaced
            with three periods
    Returns:
        str
    """
    # trying to fix backslash-replaced strings (via https://stackoverflow.com/a/57192592/4028896)
    try:
        text = text.encode("latin", "backslashreplace").decode("unicode-escape")
    except:
        pass

    return fix_text(text, normalization=normalization)


def to_ascii_unicode(text, lang="en", no_emoji=False):
    """
    Try to represent unicode data in ascii characters similar to what a human
    with a US keyboard would choose.
    Works great for languages of Western origin, worse the farther the language
    gets from Latin-based alphabets. It's based on hand-tuned character mappings
    that also contain ascii approximations for symbols and non-Latin alphabets.
    """
    # normalize quotes before since this improves transliteration quality
    text = fix_strange_quotes(text)

    if not no_emoji:
        text = demojize(text, use_aliases=True)

    lang = lang.lower()
    # special handling for German text to preserve umlauts
    if lang == "de":
        text = save_replace(text, lang=lang)

    text = unidecode(text)

    # important to remove utility characters
    if lang == "de":
        text = save_replace(text, lang=lang, back=True)

    if not no_emoji:
        text = emojize(text, use_aliases=True)

    return text


def normalize_whitespace(
    text, no_line_breaks=False, strip_lines=True, keep_two_line_breaks=False
):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more line breaks with a single newline. Also strip leading/trailing whitespace.
    """
    if strip_lines:
        text = "\n".join([x.strip() for x in text.splitlines()])

    if no_line_breaks:
        text = constants.MULTI_WHITESPACE_TO_ONE_REGEX.sub(" ", text)
    else:
        if keep_two_line_breaks:
            text = constants.NONBREAKING_SPACE_REGEX.sub(
                " ", constants.TWO_LINEBREAK_REGEX.sub(r"\n\n", text)
            )
        else:
            text = constants.NONBREAKING_SPACE_REGEX.sub(
                " ", constants.LINEBREAK_REGEX.sub(r"\n", text)
            )

    return text.strip()


# used below to keep `normalize_whitespace` as a parameter in `clean`
def _normalize_whitespace(*kwargs):
    return normalize_whitespace(*kwargs)


def replace_urls(text, replace_with="<URL>"):
    """Replace all URLs in ``text`` str with ``replace_with`` str."""
    return constants.URL_REGEX.sub(replace_with, text)


def replace_emails(text, replace_with="<EMAIL>"):
    """Replace all emails in ``text`` str with ``replace_with`` str."""
    return constants.EMAIL_REGEX.sub(replace_with, text)


def replace_phone_numbers(text, replace_with="<PHONE>"):
    """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
    return constants.PHONE_REGEX.sub(replace_with, text)


def replace_numbers(text, replace_with="<NUMBER>"):
    """Replace all numbers in ``text`` str with ``replace_with`` str."""
    return constants.NUMBERS_REGEX.sub(replace_with, text)


def replace_digits(text, replace_with="0"):
    """Replace all digits in ``text`` str with ``replace_with`` str, i.e., 123.34 to 000.00"""
    return re.sub(r"\d", replace_with, text)


def replace_currency_symbols(text, replace_with="<CUR>"):
    """
    Replace all currency symbols in ``text`` str with string specified by ``replace_with`` str.
    Args:
        text (str): raw text
        replace_with (str): if None (default), replace symbols with
            their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
            otherwise, pass in a string with which to replace all symbols
            (e.g. "*CURRENCY*")
    Returns:
        str
    """
    if replace_with is None:
        for k, v in constants.CURRENCIES.items():
            text = text.replace(k, v)
        return text
    else:
        return constants.CURRENCY_REGEX.sub(replace_with, text)


def replace_punct(text, replace_with=" "):
    """
    Replace punctuations from ``text`` with whitespaces.
    Args:
        text (str): raw text
    Returns:
        str
    """
    return re.sub(r"[^a-zA-Z\d]", replace_with, text)


# def replace_punct(text, replace_with=" "):
#     return text.translate(
#         dict.fromkeys(
#             (i for i in range(sys.maxunicode)
#              if category(chr(i)).startswith("P")),
#             replace_with,
#         ))


def remove_punct(text):
    """
    Replace punctuations from ``text`` with whitespaces.
    Args:
        text (str): raw text
    Returns:
        str
    """
    return text.translate(constants.PUNCT_TRANSLATE_UNICODE)


# def remove_punct(text):
#     """
#     Replace punctuations from ``text`` with whitespaces.
#     Args:
#         text (str): raw text
#     Returns:
#         str
#     """
#     return re.sub(r"[^a-zA-Z\d", " ", text)


def remove_punct_with_ignoration(text, ignore_list):
    return re.sub(fr"[^a-zA-Z\d{''.join(ignore_list)}']", " ", text)


def remove_emoji(text):
    for x in UNICODE_EMOJI["en"]:
        if x in text:
            text = text.replace(x, "")
    return text


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_website_links(text):
    template = re.compile(r"https?://\S+|www\.\S+")  # Removes website links
    return template.sub(r"", text)


def remove_stopwords(text, stopwords_regex=constants.STOPWORDS_REGEX):
    return stopwords_regex.sub("", text)


def expand_contractions(s):
    def replace(match):
        return constants.CONTRACTIONS_DICT[match.group(0)]

    return constants.CONTRACTIONS_REGEX.sub(replace, s)


def remove_extra_spaces(text):
    return re.sub(" +", " ", text)  # Remove Extra Spaces


# Remove repetitions
# TODO: Add remove repetitions
def remove_repetition(text):
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1", text)


def clean(
    text,
    fix_unicode=True,
    to_ascii=True,
    lower=True,
    normalize_whitespace=True,
    no_line_breaks=False,
    strip_lines=False,
    keep_two_line_breaks=False,
    no_urls=False,
    no_emails=False,
    no_phone_numbers=False,
    no_numbers=False,
    no_digits=False,
    no_currency_symbols=False,
    no_punct=False,
    no_emoji=False,
    no_contractions=False,
    no_website_links=False,
    no_stopwords=False,
    no_html_tags=False,
    no_extra_spaces=False,
    ignore_puncts: list = [],
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    replace_with_punct="",
    lang="en",
):
    """
    Normalize various aspects of a raw text. A convenience function for applying all other preprocessing functions in one go.
    Args:
        text (str): raw text to preprocess
        fix_unicode (bool): if True, fix "broken" unicode such as
            mojibake and garbled HTML entities
        to_ascii (bool): if True, convert non-to_ascii characters
            into their closest to_ascii equivalents
        lower (bool): if True, all text is lower-cased
        no_line_breaks (bool): if True, strip line breaks from text
        no_urls (bool): if True, replace all URL strings with a special URL token
        no_emails (bool): if True, replace all email strings with a special EMAIL token
        no_phone_numbers (bool): if True, replace all phone number strings
            with a special PHONE token
        no_numbers (bool): if True, replace all number-like strings
            with a special NUMBER token
        no_digits (bool): if True, replace all digits with a special DIGIT token
        no_currency_symbols (bool): if True, replace all currency symbols
            with a special CURRENCY token
        no_punct (bool): if True, remove all punctuation (replace with empty string)
        ignore_puncts (list): provide a list of ignored punctuations
        no_contractions: if True, expand all contractions
        no_html_tags (bool): if True, remove all html tags
        replace_with_url (str): special URL token, default "<URL>",
        replace_with_email (str): special EMAIL token, default "<EMAIL>",
        replace_with_phone_number (str): special PHONE token, default "<PHONE>",
        replace_with_number (str): special NUMBER token, default "<NUMBER>",
        replace_with_digit (str): special DIGIT token, default "0",
        replace_with_currency_symbol (str): special CURRENCY token, default "<CUR>",
        replace_with_punct (str): replace punctuations with this token, default "",
        lang (str): special language-depended preprocessing. Besides the default English ('en'), only German ('de') is supported

    Returns:
        str: input ``text`` processed according to function args
    Warning:
        These changes may negatively affect subsequent NLP analysis performed
        on the text, so choose carefully, and preprocess at your own risk!
    """

    if text is None:
        return ""

    text = str(text)

    if fix_unicode:
        text = fix_bad_unicode(text)
    if no_currency_symbols:
        text = replace_currency_symbols(text, replace_with_currency_symbol)
    if to_ascii:
        text = to_ascii_unicode(text, lang=lang, no_emoji=no_emoji)
    if no_emoji and not to_ascii:
        text = remove_emoji(text)
    if no_urls:
        text = replace_urls(text, replace_with_url)
    if no_html_tags:
        text = remove_html_tags(text)
    if no_emails:
        text = replace_emails(text, replace_with_email)
    if no_phone_numbers:
        text = replace_phone_numbers(text, replace_with_phone_number)
    if no_numbers:
        text = replace_numbers(text, replace_with_number)
    if no_digits:
        text = replace_digits(text, replace_with_digit)
    if no_contractions:
        text = expand_contractions(text)
    if no_website_links:
        text = remove_website_links(text)
    if no_punct:
        if replace_with_punct == "":
            if len(ignore_puncts) == 0:
                text = remove_punct(text)
            else:
                text = remove_punct_with_ignoration(text, ignore_puncts)
        else:
            text = replace_punct(text, replace_with_punct)

    if lower:
        text = text.lower()
    if no_stopwords:
        text = remove_stopwords(text)
    # if no_extra_spaces:
    # text = remove_extra_spaces(text)
    if normalize_whitespace:
        text = _normalize_whitespace(
            text, no_line_breaks, strip_lines, keep_two_line_breaks
        )
    return text
