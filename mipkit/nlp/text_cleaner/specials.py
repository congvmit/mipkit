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

import unicodedata

# add new languages here
specials = {
    "de": {
        "case_insensitive": [["ä", "ae"], ["ü", "ue"], ["ö", "oe"]],
        "case_sensitive": [["ß", "ss"]],
    }
}
escape_sequence = "xxxxx"


def norm(text):
    return unicodedata.normalize("NFC", text)


def save_replace(text, lang, back=False):
    # perserve the casing of the original text
    # TODO: performance of matching

    # normalize the text to make sure to really match all occurences
    text = norm(text)

    possibilities = (specials[lang]["case_sensitive"] +
                     [[norm(x[0]), x[1]]
                      for x in specials[lang]["case_insensitive"]] +
                     [[norm(x[0].upper()), x[1].upper()]
                      for x in specials[lang]["case_insensitive"]])
    for pattern, target in possibilities:
        if back:
            text = text.replace(escape_sequence + target + escape_sequence,
                                pattern)
        else:
            text = text.replace(pattern,
                                escape_sequence + target + escape_sequence)
    return text
