# encoding=utf-8

import re

from bs4 import BeautifulSoup


class DocPreprocessor(object):
    email_ph = "EMAIL"
    url_ph = "URL"
    ref_ph = "REF"
    version_ph = "VERSION"
    sha_ph = "SHA"
    digit_ph = "DIGIT"

    @staticmethod
    def removeEscape(text, quote_char="'"):
        # remove the quotechar in the beginning
        text = re.sub(r'\\' + quote_char, quote_char, text)
        text = re.sub(r'\\[rt]', r' ', text)
        # convert \\n to a special token
        text = re.sub(r'\\n', r'\n', text)
        return text

    @staticmethod
    def removeHtmlComment(text):
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        return text

    @staticmethod
    def removeHtmlTags(text):
        return BeautifulSoup(text, features="html.parser").get_text()

    @classmethod
    def removeEmail(cls, text):
        email_pattern = r'[\w][\w.-]*@[\w][\w-]*(\.[\w][\w-]*)+'
        text = re.sub(email_pattern, cls.email_ph, text)
        return text

    @classmethod
    def removeUrl(cls, text):
        url_pattern = r'https?://[-a-zA-Z0-9@:%._+~#?=/]+(?=($|[^-a-zA-Z0-9@:%._+~#?=/]))'
        text = re.sub(url_pattern, cls.url_ph, text)
        return text

    @classmethod
    def removeRef(cls, text):
        """
        remove link like #12345
        """
        ref_pattern = r'#[\d]+'
        text = re.sub(ref_pattern, cls.ref_ph, text)
        return text

    @classmethod
    def removeVersion(cls, text):
        version_pattern = r'(^|\s|-)[\d]+(\.[\d]+){1,}'
        text = re.sub(version_pattern, r'\1'+cls.version_ph, text)
        return text

    @classmethod
    def removeDigits(cls, text):
        sha_pattern = r'(^|\s)[\dA-Fa-f-]{7,}(?=(\s|$))'
        digit_pattern = r'(^|\s|-)[\d]+(?=(\s|$))'
        text = re.sub(sha_pattern, r'\1'+cls.sha_ph, text)
        text = re.sub(digit_pattern, r'\1'+cls.digit_ph, text)
        return text

    @staticmethod
    def convertMdToPlain(text):
        # using .+? to match as less as possible
        pattern = r'`(.+?)`'
        text = re.sub(pattern, r'\1', text)
        return text

    @staticmethod 
    def asciiFilter(tokens, min_eng_ratio=0.5):
        if len(tokens) == 0:
            return []
        regex = re.compile(r'[^\x00-\x7f]')
        valid_tokens = list(filter(lambda t: regex.search(t) == None, tokens))
        ratio = len(valid_tokens) / len(tokens)
        if ratio < min_eng_ratio:
            return []
        return valid_tokens
