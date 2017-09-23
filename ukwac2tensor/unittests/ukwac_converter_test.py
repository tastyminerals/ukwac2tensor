#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Testing ukwac_converter.py
"""
import unittest
import xml.etree.ElementTree as et
import re
import os
import sys
sys.path.insert(0, os.path.join(os.pardir, 'ukwac2tensor'))
from ukwac_converter import extract_ukwac_data


def load_xml_tests(test_name):
    '''This function reads test_name xml file and returns an ordered dict of
    parsed etree objects'''
    with open(test_name, 'r') as f:
        fdata = f.read()
    return et.fromstring(fdata)


def input_reader(test_path):
    '''Reading ukwac_converter_tests_input.xml'''
    with open(test_path, 'r') as f:
        fdata = f.read()
    return fdata


class UkwacConverterTest(unittest.TestCase):
    def test_extract_ukwac_data(self):
        """
        Testing contractions and normalization.
        <Since I am using \r\n as a sent separator and xml.etree findtext
        method seems to convert them to \n\n we need to get them back>
        """
        inp = os.path.join('test_data', 'ukwac_converter_tests_input.xml')
        outp = os.path.join('test_data', 'ukwac_converter_tests_output.xml')
        output = load_xml_tests(outp)
        result = re.sub('\n``\n ', ' \r\n``\r\n ', output.findtext('result'))
        result = re.sub('\n``\n', '\r\n``\r\n', result)
        test = extract_ukwac_data(input_reader(inp))[0]
        self.assertEqual(test, result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
