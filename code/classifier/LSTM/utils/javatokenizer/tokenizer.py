# encoding=utf-8

import re
from typing import List, Union

import nltk
from antlr4 import *
from .JavaLexer import JavaLexer


def camel_case_split(identifier):
    return re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()


def tokenize_identifier_raw(token, keep_underscore=True):
    regex = r'(_+)' if keep_underscore else r'_+'
    id_tokens = []
    for t in re.split(regex, token):
        if t:
            id_tokens += camel_case_split(t)
    # note: do not use lowercase!
    return list(filter(lambda x: len(x) > 0, id_tokens))


def tokenize_identifier(token, with_con=False):
    if with_con:
        id_tokens = " <con> ".join(tokenize_identifier_raw(token, keep_underscore=True)).split()
    else:
        id_tokens = [t.lower() for t in tokenize_identifier_raw(token, keep_underscore=False)]
    return id_tokens


def tokenize_text(text):
    str_tokens = []
    nltk_tokenized = " ".join(nltk.word_tokenize(text))
    # split according to punctuations
    # NOTE: do not care _, which will be taken care of by tokenized_identifier
    content_tokens = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', nltk_tokenized).split()
    for t in content_tokens:
        str_tokens += tokenize_identifier(t)
    return str_tokens


def tokenize_text_with_con(text):
    def _tokenize_word(word):
        new_word = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', word)
        subwords = nltk.word_tokenize(new_word)
        new_subwords = []
        for w in subwords:
            new_subwords += tokenize_identifier_raw(w, keep_underscore=True)
        return new_subwords

    tokens = []
    for word in text.split():
        if not word:
            continue
        tokens += " <con> ".join(_tokenize_word(word)).split()
    return tokens


def tokenize_string_literal(str_literal, with_con=False):
    """
    str_literal: str, STRING_LITERAL.text
    return: list of tokens
    """
    if with_con:
        str_tokens = tokenize_text_with_con(str_literal[1:-1])
    else:
        str_tokens = tokenize_text(str_literal[1:-1])
    return ['"'] + str_tokens + ['"']


def tokenize_java_code_origin(code_str) -> List[Token]:
    """
    get java tokens and return the original CommonTokens
    """
    input_stream = InputStream(code_str)
    lexer = JavaLexer(input_stream)
    tokens = []
    while True:
        t = lexer.nextToken()
        if t.text == '<EOF>':
            break
        tokens.append(t)
    return tokens


def tokenize_java_code_raw(code_str: Union[str, List[Token]]) -> List[Token]:
    """
    get java tokens without splitting compound words
    """
    literal_mapping = {
        JavaLexer.DECIMAL_LITERAL: "DECIMAL_LITERAL",
        JavaLexer.HEX_LITERAL: "HEX_LITERAL",
        JavaLexer.OCT_LITERAL: "OCT_LITERAL",
        JavaLexer.BINARY_LITERAL: "BINARY_LITERAL",
        JavaLexer.FLOAT_LITERAL: "FLOAT_LITERAL",
        JavaLexer.HEX_FLOAT_LITERAL: "HEX_FLOAT_LITERAL",
        JavaLexer.BOOL_LITERAL: "BOOL_LITERAL",
        JavaLexer.CHAR_LITERAL: "CHAR_LITERAL",
        JavaLexer.NULL_LITERAL: "NULL_LITERAL"
    }
    if isinstance(code_str, str):
        origin_tokens = tokenize_java_code_origin(code_str)
    else:
        origin_tokens = code_str
    new_tokens = []
    for t in origin_tokens:
        if t.type in [JavaLexer.COMMENT, JavaLexer.LINE_COMMENT, JavaLexer.WS]:
            continue
        elif t.type in list(literal_mapping.keys()):
            t.text = literal_mapping[t.type]
            new_tokens.append(t)
        else:
            # for keywords and identifiers
            new_tokens.append(t)
    return new_tokens


def tokenize_java_code(code_str, with_con=False):
    literal_mapping = {
        JavaLexer.DECIMAL_LITERAL: "DECIMAL_LITERAL",
        JavaLexer.HEX_LITERAL: "HEX_LITERAL",
        JavaLexer.OCT_LITERAL: "OCT_LITERAL",
        JavaLexer.BINARY_LITERAL: "BINARY_LITERAL",
        JavaLexer.FLOAT_LITERAL: "FLOAT_LITERAL",
        JavaLexer.HEX_FLOAT_LITERAL: "HEX_FLOAT_LITERAL",
        JavaLexer.BOOL_LITERAL: "BOOL_LITERAL",
        JavaLexer.CHAR_LITERAL: "CHAR_LITERAL",
        JavaLexer.NULL_LITERAL: "NULL_LITERAL"
    }
    origin_tokens = tokenize_java_code_origin(code_str)
    tokens = []
    for t in origin_tokens:
        if t.type in [JavaLexer.COMMENT, JavaLexer.LINE_COMMENT, JavaLexer.WS]:
            continue
        if t.type == JavaLexer.STRING_LITERAL:
            tokens += tokenize_string_literal(t.text, with_con=with_con)
        elif t.type in list(literal_mapping.keys()):
            tokens.append(literal_mapping[t.type])
        elif t.type == JavaLexer.IDENTIFIER:
            tokens += tokenize_identifier(t.text, with_con=with_con)
        else:
            # for keywords
            tokens.append(t.text)
    return tokens

