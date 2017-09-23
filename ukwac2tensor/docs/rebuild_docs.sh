#!/bin/bash
# This script rebuilds project docs
# Project structure of Sphinx for python2:
#  project/docs
#  project/ukwac2tensor
#  project/unittests
# 
# Building docs with Sphinx:
#  sphinx-quickstart2
#  insert "sys.path.insert(0, os.path.abspath(os.path.pardir))" into conf.py
#  sphinx-apidoc2 -o . ../ukwac2tensor/
#  sphinx-build2 . ukwac2tensor_docs

sphinx-build2 . html

