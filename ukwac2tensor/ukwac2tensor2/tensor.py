#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script creates 3-order tensor out of word0, relation(link), word1 tuples.
It can compute either tuple counts or tuple PMI score.
The result tensor is stored in .h5 file.

Complete ukWaC heads corpus requires ~9.5h to process and about ~7GB of RAM!
"""
from __future__ import division
import argparse
import gzip
import xml.etree.cElementTree as cet
import os
import re
import sys
from collections import defaultdict
from math import log
import pandas as pd
import numpy as np


__author__ = "p.shkadzko@gmail.com"


### uncomment below to generate script execution .png graph
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput


def parse_heads_xml(files_list, wfilter):
    """
    Parse the file with phrase heads and create a dict of
    ``{(word0, link, word1): cnt}`` corresponding to "gov", "dep type", "dep"
    tags in the source xml.
    Check

    Args:
        | *files_list* (list) -- a list of files to process
        | *wfilter* (list) -- a word filter, if a word is not present in the
          filter, it is skipped and not included into the tensor

    Returns:
        *dict* -- a dictionary of triple counts::

            {('august-n', 'A2', 'surround-v'): 51,
             ('enquiry-n', 'A1', 'justify-v'): 200,
             ('september-n', 'AM-TMP', 'set-v'): 121, ...}

    """
    def update_progress():
        """
        Display a simple progress bar.
        """
        bar = round((100 - (done / total * 100)), 2)
        sys.stdout.write('\r{0} {1}'.format(fname, bar))
        sys.stdout.flush()

    def filtered(word):
        """
        Checks whether a given word is in the filter list.

        Args:
            *word* (str) -- a word of ``apple-n`` notation
        Returns:
            *bool* -- ``True`` if word is in the list, ``False`` otherwise

        """
        # word = word.rsplit('-', 1)[0]  # for Ottokar vocab, comment out
        if not wfilter:  # if no vocab filter was supplied, return always True
            return True
        elif word in wfilter:
            return True
        return False

    def valid(head, pos, alg, deptype):
        """
        Checks whether xml element is a singleton and should be included into
        the tensor.
        Check if xml element "algorithm" attribute is in allowed ALGS list so
        that besides ``head_searcher.py`` you could also control which
        results of MALT, MALT_SPAN or LINEAR to include.
        Do various "junk" checking.

        <Even though we were able to successfully find a phrase head it does
        not mean that the head or its governor can be a valid word. It might
        happen so, that instead of a phrase head we receive a chunk of
        punctuation chars or a gardbled word or something we do not want to
        see in our tensor. For this reason we need to do some light regex
        checking.>

        <TODO:
        We need to check if a role contains infinitivals like "to live to
        dance". Remove "to" token and reconnect each verb with its governor.
        This is relatively rare case that is why I decided skip it for now.
        Besides, it drags in unecessary comlexity.>

        Args:
            | *head* (str) -- phrase head or a governor token
            | *pos* (str) -- word POS-tag
            | *alg* (str) -- algorithm (strategy) used to retrive the token
              (if the token is a phrase head, ``None`` if it is a governor)
            | *deptype* (str) -- dependency type

        Returns:
            *bool* -- ``True`` if a given token is good, ``False`` otherwise

        """
        # check for whitespaces and None
        if ' ' in head:
            return False
        # checking if elem.text is not "junk"
        if len(head) > 53:
            return False
        elif len(re.sub(r'\w', '', head)) > 3:
            return False
        elif not re.sub(r'\W', '', head):
            return False
        jnk = '|'.join(['&amp;amp;', '@ord@', '^[^\w]+.*', '.*[^\w]+$',
                        '[0-9*]+', '.*@[\w]+.[\w]+', '_+'])
        rjnk = re.compile(jnk)
        if rjnk.match(head):
            return False
        # check which failed nodes to include
        failed_tag = alg in ['FAILED']
        if hasgov and INCLUDE_FAILED and failed_tag:
            # checks if singleton has allowed postag "not/rb/14"
            if pos in ['jj', 'rb', 'vv', 'vb', 'vh']:
                return True
            elif pos in ['md'] and deptype == 'AM-MOD':
                return True
            elif deptype == 'AM-NEG':
                return True
        elif hasgov and not INCLUDE_FAILED and not failed_tag:
            return True
        elif hasgov and INCLUDE_FAILED and not failed_tag:
            return True
        # check for allowed strategies
        if alg:
            return alg in ALGS
        return False

    global INCLUDE_FAILED
    global ALGS
    # for progress bar
    total = len(files_list)
    done = len(files_list)
    # start processing
    table_dic = defaultdict(int)
    for fname in files_list:
        print('\nprocessing ' + fname + '...')
        hasgov = False
        if fname.endswith('.gz'):
            fheads = gzip.open(fname, 'r')
        else:
            fheads = open(fname, 'r')
        try:
            xml_iter = cet.iterparse(fheads, events=('start', 'end'))
        except:
            print("Error while processing {0}.".format(fname))
            print("Skipping this file for now.")
            print("If you still want to include it into the tensor, \n\
            try to rebuild it with head_searcher.py first\n\
            and then recreate the tensor again.")
            continue
            
        for event, elem in xml_iter:
            if event == 'end':
                if elem.tag == 'governor':
                    hasgov = True
                    algorithm = elem.attrib.get('algorithm')
                    dtype = elem.attrib.get('type')
                    head, pos = elem.text.rsplit('/', 2)[:2]
                    if valid(head, pos, algorithm, dtype):
                        selem = elem.text.rsplit('/', 2)
                        strip_gov = '-'.join([selem[0].lower(),
                                              selem[1][0].lower()])
                        if filtered(strip_gov):
                            tuple_key = (strip_gov, )
                        else:
                            hasgov = False
                    else:
                        hasgov = False
                elif elem.tag == 'dep':
                    # deciding which algorithm heads to include
                    algorithm = elem.attrib.get('algorithm')
                    dtype = elem.attrib.get('type')
                    head, pos = elem.text.rsplit('/', 2)[:2]
                    if valid(head, pos, algorithm, dtype):
                        selem = elem.text.rsplit('/', 2)
                        if selem[1] in ['prp']:
                            selem[1] = 'nn'
                        elif selem[1] == 'vbg':
                            selem[1] = 'nn'
                        stripped_dep = '-'.join([selem[0].lower(),
                                                 selem[1][0].lower()])
                        if filtered(stripped_dep):
                            tuple_key += (elem.attrib['type'], )
                            tuple_key += (stripped_dep, )
                            # check if tuple_key is a mirror pair:
                            # ('outline-v', 'V', 'outline-v')
                            # we are not counting these
                            if tuple_key[0] == tuple_key[-1]:
                                tuple_key = tuple_key[0],
                                continue
                            table_dic[tuple_key] += 1
                            tuple_key = tuple_key[0],
                elem.clear()
        del xml_iter
        done -= 1
        fheads.close()
        update_progress()
        
    # switch for word0(dep) link word1(gov)
    print '\nTENSOR SIZE:', len(table_dic.keys())
    tensor_dict = dict([(tuple(reversed(k)), v) for k, v in table_dic.items()])
    return tensor_dict


def switch2pmi(pdf):
    """
    Using current tensor counts, compute pmi scores.
    Use equation: PMI' = log(p(w0,w1|link) / (p(w0|link)p(w1|link)))
    Once pmi scores are computed, replace current counts with pmi.

    Args:
        *pdf* (DataFrame) -- tensor containing counts of word triples.

    Returns:
        *dict* -- new tensor dict containing pmi scores.
    """
    # numerator: counts(w0,w1,link) * counts(link)
    # define counts(link)
    link_sum = pdf.groupby(['link'])['counts'].sum()
    # join counts(link) to main pdf
    pdf_main = pdf.join(link_sum, on='link', how='left', rsuffix='_link')
    # multiply counts(w0,w1,link) * counts(link) using joined pdf
    numer = pdf_main['counts'] * pdf_main['counts_link']

    # denominator: counts(w0,link) * counts(w1,link)
    w0l_sum = pdf.groupby(['word0', 'link'], as_index=True).sum()
    w1l_sum = pdf.groupby(['word1', 'link'], as_index=True).sum()
    # first join counts(w1, link) to main pdf indexing on w1, link
    pdf_w1l = pdf.set_index(['word1','link']).join(w1l_sum, how='left', rsuffix='_w1l').reset_index()
    # first join counts(w0, link) to main pdf indexing on w0, link
    pdf_w1l_w0l = pdf_w1l.set_index(['word0', 'link']).join(w0l_sum, how='left', rsuffix='_w0l').reset_index()
    # count(w0,link) * count(w1,link)
    denom = pdf_w1l_w0l['counts_w0l'] * pdf_w1l_w0l['counts_w1l']
    # calculating ppmi
    pdf_w1l_w0l['ppmi'] = np.log(numer / denom) + log(1)

    # export results to dict
    return pdf_w1l_w0l.set_index(['word0', 'link', 'word1'])['ppmi'].to_dict()


def go_pandas(args):
    """
    Run xml parser, create a tensor from the returned counts dictionary,
    compute PMI' scores and replace the counts.
    Save the new dataframe in `.h5` file.

    Args:
        *args* (dict) -- arguments supplied by the user

    """
    # init env vars
    global INCLUDE_FAILED
    INCLUDE_FAILED = args.include_failed
    global ALGS
    if args.alg:
        ALGS = args.alg.split(',')
        algs = '-'.join(args.alg.split(','))
    # h5 file name format
    if not INCLUDE_FAILED:
        h5fname = ''.join(['tensor{0}.'.format('_pmi' if args.pmi else ''), algs, '.h5'])
    else:
        h5fname = ''.join(['tensor{0}.'.format('_pmi' if args.pmi else ''), algs, '-with-FAILED', '.h5'])
    if args.out:
        try:
            os.mkdir(args.out)
        except OSError:
            print("Can't create {0} or it already exists".format(args.out))
        h5fname = os.path.join(args.out, h5fname)
    if args.dir:
        files_list = [os.path.join(args.dir, i)
                      for i in os.listdir(args.dir)]
    if args.file:
        files_list = [args.file]
    # defining vocab to filter governors
    filtset = False
    if args.filter:
        with open(args.filter, 'r') as ffile:
            filtlst = ffile.read().split('\n')
            filtlst.remove('')  # comment out this line for Ottokar's vocab
            filtset = frozenset(filtlst)
    # start processing
    giant_govdep_dic = parse_heads_xml(files_list, filtset)

    # convert to pdf in to compute pmi scores
    # Example of govdep_tup: ('call-v', 'AM-MNR', 'aptly-r', '1')
    print 'Creating pdf...'
    govdep_tup = [(tup[0], tup[1], tup[2], giant_govdep_dic[tup])
                  for tup in iter(giant_govdep_dic)]
    pdf = pd.DataFrame.from_records(govdep_tup)
    pdf.columns = ['word0', 'link', 'word1', 'counts']

    # compute PMI' score for each (word0, link, word1) triple and replace cnt
    if args.pmi:
        print "Computing PMI' scores..."
        new_govdep_dic = switch2pmi(pdf)
        # convert to pdf again
        print 'Creating new pdf...'
        pmi_govdep_tup = [(tup[0], tup[1], tup[2], new_govdep_dic[tup])
                          for tup in iter(new_govdep_dic)]
        pdf = pd.DataFrame.from_records(pmi_govdep_tup)
        pdf.columns = ['word0', 'link', 'word1', 'pmi']

    # store in h5
    print 'Dumping pdf to {0}'.format(h5fname)
    h5file = pd.HDFStore(h5fname, 'a', complevel=9, complib='blosc')
    try:
        h5file.append("tensor", pdf,
                      data_columns=['word0', 'link', 'word1'],
                      nan_rep='_!NaN_',
                      min_itemsize={'word0': 55, 'link': 15, 'word1': 55})
        h5file.close()
    except Exception as err:
        print(err)
        sys.exit(1)


# globals
INCLUDE_FAILED = False
ALGS = []


if __name__ == '__main__':
    prs = argparse.ArgumentParser(description="""
    This script converts files with phrase heads into pandas data frame
    object and stores it in roles.h5 -- distributional memory tensor.

    Getting data from tensor:
        import pandas as pd
        pdf = pd.HDFStore('tensor.MALT-MALT_SPAN-LINEAR.h5')
        pdf['tensor']

    """)
    prs.add_argument('-d', '--dir',
                     default=os.getcwd(),
                     help='Specify a directory where files processed by head \
                           searching script are located.',
                     required=False)
    prs.add_argument('-f', '--file',
                     help='Specify the result xml file to process',
                     required=False)
    prs.add_argument('-o', '--out',
                     help='Specify output directory. If not specified current\
                           dir is used.',
                     required=False)
    prs.add_argument('-filter', '--filter',
                     help='You can specify a file with words that shall be \
                           counted. \
                           Words not present in the file shall be skipped. \
                           Word format: make-n',
                     required=False)
    prs.add_argument('-include-failed', '--include-failed',
                     action='store_true',
                     help='Include FAILED singletons into the tensor.',
                     required=False)
    prs.add_argument('-pmi', '--pmi',
                     action='store_true',
                     help='Replace tensor counts with pmi scores.',
                     required=False)
    prs.add_argument('-a', '--alg',
                     default='MALT,MALT_SPAN,LINEAR',
                     help='Choose the results of which head searching strategy\
                           to include into the final tensor. Available \
                           strategies: MALT, MALT_SPAN, LINEAR. Usage example:\
                           ./tensor.py --alg MALT,MALT_SPAN',
                     required=False)
    arguments = prs.parse_args()
    go_pandas(arguments)

    ### uncomment below to generate script execution .png graph
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'tensor.png'
    # with PyCallGraph(output=graphviz):
    #     go_pandas(arguments)

    # accessing hdf file
    """
    import pandas as pd
    pd.read_hdf('tensor.h5', key='tensor')
    """
