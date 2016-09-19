#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import re
import unicodedata
import string
import json

from fuzzywuzzy import utils
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from cleanco import cleanco

"""
BYTE STRING VS. UNICODE STRING in Python 2.7
[Ref](http://stackoverflow.com/questions/10060411/byte-string-vs-unicode-string-python)
>>> a = u"αά".encode('utf-8')
>>> a
'\xce\xb1\xce\xac'
>>> type(a)
<type 'str'>
>>> a.decode('utf-8')
u'\u03b1\u03ac'
>>> print(a.decode('utf-8'))
αά

REFER OFFICIAL [DOC](https://docs.python.org/2/howto/unicode.html)

OTHER REF: [StackExchange](http://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte)
answer by Alastair McCormack
"""


def convert_unicode(x):
    """
    Input::
        x: a unicode expressed as a unicode with u inside, eg. x=u"u'abc'", or u"u'abc"
    Output::
        return: converted into unicode, and filter non-printable char
    """
    try:
        x = (x.strip()[2:-1]).decode('unicode_escape')#unicode(x.strip()[2:-1]).decode('unicode_escape')\
    except UnicodeDecodeError:
        x = unicode(x.strip()[2:]).decode('unicode_escape')
    x = filter(lambda a: a in string.printable, x)
    # unicode_escape: unicode-to-str encoding and str-to-unicode decoding
    return x


def strip_u(x):
    """
    Input::
        x: a str with unicode inside, eg. x="u'abc'", the unicode inside may or may not contain any non-ascii code
    Output::
        return: converted into unicode
    """
    # string_escape: str-to-str encoding and decoding
    return x.strip()[2:-1].decode('string_escape')#http://stackoverflow.com/questions/5186839/python-replace-with

"""
u'好' -> '\xe5\xa5\xbd' :: encoded into byte format; u'\u597d' :: in normal unicode format
PROBLEM:: If given '\xe5\xa5\xbd' or '\u597d' (without u), how to correctly transform them into correct unicode format
"""
def encode_str_with_unicode(x):
    """Given a str, return the corresponding unicode.
    e.g. "\u597d" -> u"\u597d"; '\xe5\xa5\xbd' -> u"\u597d"; '\xeb' -> u"\xeb"
    """
    try:
        x = unicode(x, 'utf8')## '\xe5\xa5\xbd' => u'\u597d' (correct!!); '\u597d' => u'\\u597d' (WRONG!!!)
    except UnicodeDecodeError:## e.g. '\xeb' will cause error, because it is the byte encoded by "latin1"
        x = unicode(x, 'latin1')
    try:##NOTE:: u'\xeb' == u"\u00eb" BUT:: u'\xe5\xa5\xbd' == u'\u597d'; because the former is within 256, but the later is not
        x = x.decode('unicode_escape')#To solve u'\\u597d', i.e. convert it into u'\u597d'
    except UnicodeEncodeError:# Error raise because:: 'ascii' codec can't encode character u'\u597d' in position 0: ordinal not in range(128)
        pass
    return x

def reencode_with_unicode(x):
    """Given any string/unicode, convert into string and encode them with ascii"""
    # x = x.decode('string_escape')
    if isinstance(x, str):
        x = encode_str_with_unicode(x)
    elif isinstance(x, unicode):
        x = x
    else:
        raise UnicodeError("Not a str nor unicode.")
    return x

def reencode_str_with_ascii(x):
    """
    Input::
        x: type(x) is str. it may or may not contain any non-ascii code
    Output::
        return: str, with non-ascii code converted to ascii code if possible
    """
    try:
        x.decode("ascii")#x has only ascii char
    except UnicodeDecodeError:#x has non-ascii char
        x = encode_str_with_unicode(x)
        x = unicodedata.normalize('NFKD', x).encode('ascii','ignore')
    return x

def reencode_unicode_with_ascii(x):
    """
    Input::
        x: type(x) is unicode
    Output::
        return: str, with non-ascii code converted to ascii code if possible
    """
    x = unicodedata.normalize('NFKD', x).encode('ascii','ignore')
    return x

def reencode_with_ascii(x):
    """Given any string/unicode, convert into string and encode them with ascii"""
    # x = x.decode('string_escape')
    type_x = type(x)
    if type_x is str:
        x = reencode_str_with_ascii(x)
    elif type_x is unicode:
        x = reencode_unicode_with_ascii(x)
    return x

def clean_sematics_job_title(x):
    x = reencode_with_ascii(x)
    x = x.lower().strip()
#   x = " ".join(re.sub("\.\.\..*\.\.\.|.* w |.*leader |.*director |.*manager |\(.+\)|\/|-|\(|\)|\'|\"|@|,|\*|\||\?|:|!|_|\.|\&|;", " ", x).split())#remove contents in (), substitute various separators with white space, and then join them use single space
    x = " ".join(re.sub(r"\.\.\..*\.\.\.|.*leader |.*director |.*manager |\(.+\)|\/|-|\(|\)|\'|\"|@|,|\*|\||\?|:|!|_|\.|\&|;|^pt ", " ", x).split())#compared above, \.* w pattern is removed because it is not applicable here
    x = cleanco(x).clean_name()
    x = " ".join(re.sub(r" the|the ", " ", x).split())#Strip "the" in the account name -> strip it here because we need to take lower case first!
    return x#cleanco(utils.full_process(x, force_ascii=True)).clean_name()

def clean_sematics_city(x):
    if isinstance(x, basestring):
        x = reencode_with_ascii(x)
    else:
        return x
    return remove_chars_in_city(x)#cleanco(utils.full_process(x, force_ascii=True)).clean_name()

def remove_chars_in_city(x):
    x = x.lower().strip()
    if ("quezon" in x.split()) or ("london" in x.split()):
        return " ".join(re.sub(r"\/|-|\(|\)|\'|\"|@|,|\*|\||\?|:|!|_|\.|\&|;|\bcommune\b|\bkraj\b|\bapskritis\b|\bof\b", " ", x).split())##Do not remove "city" inside the string
    x = " ".join(re.sub(r"\/|-|\(|\)|\'|\"|@|,|\*|\||\?|:|!|_|\.|\&|;|\bcommune\b|\bkraj\b|\bapskritis\b|\bcity\b|\bctiy\b|\bof\b", " ", x).split())#compared above, \.* w pattern is removed because it is not applicable here
    return x


def clean_sematics_fast(x):
    """A fast version of cleaning sematics. The input is assume to be unicode encoded in 'utf8' """
    x = unicodedata.normalize('NFKD', x).encode('ascii','ignore')
    x = " ".join(re.sub(r"<|=|>|`|\^|\]|\[|\&|\.|;|~|}|{|\||#|\\\|\+|\"|\/|-|\(|\)|\'|@|,|\*|\?|:|!|_", " ", x).split())
    return re.sub(r" the|the ", " ", x).strip().lower()




## <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---
## <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---
"""
GET STRING OBJECTS INSTEAD OF UNICODE ONES FROM JSON IN PYTHON

FOR DETAIL, REF [HERE](http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python#13105359)
"""
def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data
## <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---
## <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---


















