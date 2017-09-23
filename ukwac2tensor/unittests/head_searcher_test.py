#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Testing head_searcher.py
"""
import ast
import re
import unittest
import xml.etree.ElementTree as et
from collections import OrderedDict as od
import os
import sys
sys.path.insert(0, os.path.join(os.pardir, 'ukwac2tensor'))
import head_searcher as hs


def load_xml_tests(test_name):
    '''This function reads test_name xml file and returns an ordered dict of
    parsed etree objects'''
    with open(test_name, 'r') as f:
        fdata = f.read()
    etrees = od()
    for sent in enumerate(fdata.split('\n\n')):
        if not sent[1]:
            continue
        etrees[sent[0]] = et.fromstring(sent[1])
    return etrees


class HeadSearcherTest(unittest.TestCase):
    def test_map_to_malt_indexes(self):
        tests_path = os.path.join('test_data', 'map_to_malt_indexes_tests.xml')
        for etr in load_xml_tests(tests_path).values():
            maltdata = etr.findtext('malt')
            lemsent = etr.findtext('lemsent')
            nonxml = re.sub(r'amp;', 'amp;amp;', etr.findtext('result'))
            result = od(ast.literal_eval(nonxml))
            # result = od(ast.literal_eval(etr.findtext('result')))
            self.assertEqual(hs.map_to_malt_indexes(lemsent, maltdata), result)

    def test_switcher_function(self):
        tests_path = os.path.join('test_data', 'switcher_function_tests.xml')
        for etr in load_xml_tests(tests_path).values():
            tup_arg = ast.literal_eval(etr.findtext('tup_arg'))
            remapped = od(ast.literal_eval(etr.findtext('remapped')))
            deptype = etr.findtext('deptype')
            govs = ast.literal_eval(etr.findtext('govs'))
            head = ast.literal_eval(etr.findtext('head'))
            self.assertEqual(hs.alg_controller(tup_arg, remapped, deptype, govs),
                             head)


if __name__ == '__main__':
    hs._globals_init()
    unittest.main(verbosity=2)
