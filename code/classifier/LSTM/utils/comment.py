# encoding=utf-8
import re
from typing import List

import nltk as nltk

from .doc_preprocessor.doc_preprocessor import DocPreprocessor
from .javatokenizer.tokenizer import tokenize_text_with_con
from .edit import match_sents


class CommentCleaner(object):
    def __init__(self, replace_digit: bool):
        self.replace_digit = replace_digit

    def clean(self, desc: str):
        # preprocess desc: remove email, url, ref, digits
        desc = DocPreprocessor.removeHtmlTags(desc)
        desc = DocPreprocessor.removeEmail(desc)
        desc = DocPreprocessor.removeUrl(desc)
        desc = DocPreprocessor.removeRef(desc)
        desc = DocPreprocessor.removeVersion(desc)
        if self.replace_digit:
            desc = DocPreprocessor.removeDigits(desc)
        desc = re.sub(r'//', '', desc)
        return desc


def tokenize_desc_with_con(desc: str) -> List[str]:
    return tokenize_text_with_con(desc)


class JavadocDescPreprocessor:
    def __init__(self, comment_cleaner: CommentCleaner):
        self.comment_cleaner = comment_cleaner

    @staticmethod
    def my_sent_tokenize(javadoc: str) -> List[str]:
        sents = nltk.sent_tokenize(javadoc)
        new_sents = []
        for sent in sents:
            # use sent.strip() to make sure there are no \n at the beginning and the end.
            sub_sents = re.split(r'(?<!^)((\n{2,}(?!$))|(\n\s*(?=[A-Z][a-z\s])))', sent.strip(), flags=re.DOTALL)
            new_sents += [sub.strip() for sub in sub_sents if sub and sub.strip()]
        return new_sents

    @staticmethod
    def _filter_trivial_sents(sents: List[List[str]]):
        """
        filter out the sentences only with <con> and punctuations
        """
        new_sents = []
        for sent in sents:
            # including _
            temp_sent = " ".join(sent)
            temp_sent = re.sub(r'<con>', '', temp_sent)
            temp_sent = re.sub(r'[-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~_]', '', temp_sent).strip()
            if not temp_sent:
                continue
            new_sents.append(sent)
        return new_sents

    def _preprocess_single_desc(self, javadoc):
        clean_javadoc = self.comment_cleaner.clean(javadoc)
        src_sents = self.my_sent_tokenize(clean_javadoc)
        # my_sent_tokenize has already guarantee that sent is not empty
        src_sent_tokens = [tokenize_desc_with_con(sent) for sent in src_sents]
        src_sent_tokens = self._filter_trivial_sents(src_sent_tokens)
        return src_sent_tokens, src_sents

    def preprocess_desc(self, src_javadoc, dst_javadoc):
        src_sent_tokens, src_sents = self._preprocess_single_desc(src_javadoc)
        dst_sent_tokens, dst_sents = self._preprocess_single_desc(dst_javadoc)
        matches = match_sents(src_sent_tokens, dst_sent_tokens)
        comments = []
        for src_index, (dst_index, dis) in matches.items():
            comments.append({
                'src_sent': src_sents[src_index],
                'dst_sent': dst_sents[dst_index],
                'src_sent_tokens': src_sent_tokens[src_index],
                'dst_sent_tokens': dst_sent_tokens[dst_index],
                'dis': dis
            })
        return comments


def test_javadoc_desc_preprocessor():
    processor = JavadocDescPreprocessor(CommentCleaner(False))
    src = "I am the first sentence. This is another word\nThat is for deletion\n\nNothing to match\n"
    dst = "It is the first sentence. That is for addition and reorder.\nThis is the other word"
    results = processor.preprocess_desc(src, dst)
    print(results)


if __name__ == '__main__':
    test_javadoc_desc_preprocessor()

