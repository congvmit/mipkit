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

from . import core
from . import utils


class TextCleaner:
    @utils.autoargs()
    def __init__(
        self,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
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
        no_html_tags=False,
        no_stopwords=False,
        no_extra_spaces=False,
        ignore_puncts=[],
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        replace_with_punct="",
        lang="en",
    ):
        pass

    def clean(self, text):
        return core.clean(
            text,
            fix_unicode=self.fix_unicode,
            to_ascii=self.to_ascii,
            lower=self.lower,
            normalize_whitespace=self.normalize_whitespace,
            no_line_breaks=self.no_line_breaks,
            strip_lines=self.strip_lines,
            keep_two_line_breaks=self.keep_two_line_breaks,
            no_urls=self.no_urls,
            no_emails=self.no_emails,
            no_phone_numbers=self.no_phone_numbers,
            no_numbers=self.no_numbers,
            no_digits=self.no_digits,
            no_currency_symbols=self.no_currency_symbols,
            no_punct=self.no_punct,
            no_emoji=self.no_emoji,
            no_contractions=self.no_contractions,
            no_website_links=self.no_website_links,
            no_stopwords=self.no_stopwords,
            no_html_tags=self.no_html_tags,
            no_extra_spaces=self.no_extra_spaces,
            ignore_puncts=self.ignore_puncts,
            replace_with_url=self.replace_with_url,
            replace_with_email=self.replace_with_email,
            replace_with_phone_number=self.replace_with_phone_number,
            replace_with_number=self.replace_with_number,
            replace_with_digit=self.replace_with_digit,
            replace_with_currency_symbol=self.replace_with_currency_symbol,
            replace_with_punct=self.replace_with_punct,
            lang=self.lang,
        )

    def __call__(self, text):
        return self.clean(text)
