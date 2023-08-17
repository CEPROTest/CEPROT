# encoding=utf-8
from unittest import TestCase

import nltk
import unittest
from .doc_preprocessor import DocPreprocessor


class TestDocPreprocessor(unittest.TestCase):
    def test_remove_escape(self):
        test = [
            "this is a \\'test\\'\\t\\r\\none"
        ]
        result = [
            "this is a 'test'  \none"
        ]
        for t, r in zip(test, result):
            cur = DocPreprocessor.removeEscape(t, "'")
            self.assertEqual(r, cur)

    def test_ascii_filter(self):
        test = [
        ]
        result_90 = [[], "This does n't contain any non-ascii characters".split()]
        result_40 = ["This non-ascii".split(), "This does n't contain any non-ascii characters".split()]
        for t, r_90, r_40 in zip(test, result_90, result_40):
            tokens = nltk.word_tokenize(t)
            r = DocPreprocessor.asciiFilter(tokens, 0.9)
            self.assertEqual(r, r_90)
            tokens = nltk.word_tokenize(t)
            r = DocPreprocessor.asciiFilter(tokens, 0.4)
            self.assertEqual(r, r_40)

    def test_remove_html_comment(self):
        test = [
            "test <!-- describe the changes you have made here : what , why , ... --> test",
            "<!-- multiline\nhtml comment\ntest--> test",
        ]
        result = [
            "test  test",
            " test"
        ]
        for t, r in zip(test, result):
            cur = DocPreprocessor.removeHtmlComment(t)
            self.assertEqual(r, cur)

    def test_remove_html_tags(self):
        test = [
            "Test case for\n<a href=\"URL\"\n  >PDFBOX-90</a> - Support explicit retrieval of page labels."
        ]
        result = [
            "Test case for\nPDFBOX-90 - Support explicit retrieval of page labels."
        ]
        for t, r in zip(test, result):
            cur = DocPreprocessor.removeHtmlTags(t)
            self.assertEqual(r, cur)

    def test_remove_email(self):
        test = [
            "\n\nCo-Authored-By: albertzaharovits <albert.zaharovits@gmail.com>",
            "gg_xx@hehe.edu.au",
        ]
        result = [
            "\n\nCo-Authored-By: albertzaharovits <EMAIL>",
            "EMAIL"
        ]
        for t, r in zip(test, result):
            cur = DocPreprocessor.removeEmail(t)
            self.assertEqual(r, cur)

    def test_remove_links(self):
        test = [
        ]
        result = [
            "This contains a [link](URL) as you see URL",
            "Please see URL for more info about REF",
            "fix REF in this [pr](URL)",
            "'reported' here: URL"
        ]
        for t, r in zip(test, result):
            cur = DocPreprocessor.removeUrl(t)
            cur = DocPreprocessor.removeRef(cur)
            self.assertEqual(r, cur)

    def test_remove_digits(self):
        test = [
            "123 I have a single 5 7",
            "I have multi test222 5678 test-258 11x22",
            "since commit sha : 11aabbccddfff",
            "From commit abef12345-1eb-1b0-ace0-1e85946e1d7"
        ]
        result = [
            "DIGIT I have a single DIGIT DIGIT",
            "I have multi test222 DIGIT test-DIGIT 11x22",
            "since commit sha : SHA",
            "From commit SHA"
        ]
        for t, r in zip(test, result):
            cur = DocPreprocessor.removeDigits(t)
            self.assertEqual(r, cur)

    def test_convert_md_to_plain(self):
        test = [
            "The assertion `assertOpsOnPrimary` does `not` store seq_no",
            "Merge branch `master` `into` test/engine-primary-version"
        ]
        result = [
            "The assertion assertOpsOnPrimary does not store seq_no",
            "Merge branch master into test/engine-primary-version"
        ]
        for t, r in zip(test, result):
            cur = DocPreprocessor.convertMdToPlain(t)
            self.assertEqual(r, cur)


if __name__ == "__main__":
    unittest.main()
