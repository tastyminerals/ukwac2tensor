#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SENNA role labeler does not provide head detection functionality.
This script attempts to find phrase head using the results of Malt parser and
some additional heuristics heavily influenced by Magerman's head percolation
table.
We implement an algorithm which is applied on SENNA processed files which we
afterwards converted into xml format.
The algorithm consists of **three** major stages or fallback strategies
arbitrary called MALT, MALT_SPAN and LINEAR. The fallback strategies are
**applied sequentially** until the phrase head is found. Such a 3-step approach
allows us to achieve 90% head recall on the whole corpus. Intuitively, we give
preference to the output produced by Malt parser as opposed to linear head
search.

.. figure::  images/head_searcher.png
   :align:   center
   :scale: 85%

   execution diagram

"""
from __future__ import division
import argparse
import copy
import gzip
import os
import re
import xml.etree.ElementTree as et
import shutil
import subprocess as sb
import sys

from string import punctuation
from collections import deque, OrderedDict as od


__author__ = "p.shkadzko@gmail.com"


### uncomment below to generate script execution .png graph
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput


def _globals_init():
    """
    Initialize global list of algorithm names.
    <This function is used only by unittest scripts>
    """
    global ALGS
    ALGS = ['MALT', 'MALT_SPAN', 'LINEAR']


class Linear:
    """
    Class that implements LINEAR strategy.
    LINEAR strategy is triggered when MALT and MALT_SPAN fail or explicitly
    with ``-a LINEAR`` option. The key idea here is to return the last token
    in the stack.
    For example, for *“against the twilight redness of the bay”* node we first
    check if it is an adverbial phrase, if it is, the strategy returns the
    first VBG|VBZ|VBN, NN or RB token.
    If there are several such tokens, VBG|VBZ|VBN has the highest and RB the
    lowest priority. For example, VBG|VBZ|VBN will be returned if it precedes
    NN|RB and NN will be returned if it precedes RB accordingly.
    """
    @staticmethod
    def run(tup_arg, deptype):
        """
        Run LINEAR algorithm.
        Decide whether to call AP or NP head searching heuristics.

        Args:
            | *tup_arg* (tuple) -- tuples of arguments:
               ``('reading', 'nnp', '26'), ('festival', 'nnp', '27')``
            | *deptype* (str) -- dependency type

        Returns:
            *found head with POS-tag and position index* ``tower/NNS/12`` and
            the algorithm name

        """
        # if AdvP phrase
        # TODO add A1, A2 in future
        if re.match(r'MNR|TMP|ADV', deptype.split('-')[-1], flags=re.I):
            # if 'RB|VBG|VBZ' POS-tags exist
            if filter(lambda x: re.match(r'RB|VBG|VBZ|VBN', x[1], flags=re.I),
                      tup_arg):  # TIP: JJ not included
                return Linear._get_ap_head(tup_arg)
        return Linear._get_np_head(tup_arg)

    @staticmethod
    def _get_ap_head(tups):
        """
        Find an appropriate AP head (RB|VBG) for the role types: ADV, MNR, TMP.
        Return last VBG, NN or RB. Head priority -- VBG > NN > RB.

        Args:
            *tups* (tuple) -- tuples of arguments:
             ``('reading', 'nnp', '26'), ('festival', 'nnp', '27')``

        Returns:
            *found head with POS-tag and position index* ``tower/NNS/12`` and
            the algorithm name

        """
        for word in tuple([t for t in reversed(tups)]):
            # VBG takes over all if last
            if re.match(r'VBG|VBZ|VBN', word[1], flags=re.I):
                return '/'.join(word), 'LINEAR'
            # NN takes over RB if last
            elif re.match(r'NN', word[1], flags=re.I):
                return '/'.join(word), 'LINEAR'
            # Last RB is the head
            elif re.match(r'RB', word[1], flags=re.I):
                return '/'.join(word), 'LINEAR'

    @staticmethod
    def _get_np_head(tups, np=False):
        """
        Implement Magerman's table inspired rules to search for phrase heads.

        Example phrase: *“against the twilight redness of the bay”*

        Iterate over each token and check if it is a member of
        NN|NNS|NNP|NNPS|VBG list, if it is, pop (if stack is not empty) and
        push the current token to the stack. In such manner iterate over all
        tokens in the node and return the last element on the stack in the end.
        When encounter a token outside of the list, break the iteration but
        only if we have already pushed once and return whatever is on the
        stack.

        Suppose we encounter a possessive, we do not break the iteration in
        that case but continue, we push the next token after the possessive
        and return it in the end. Applying this algorithm to the
        example phrase above we find *“redness”* to be the phrase head.

        Args:
            *tups* (tuple) -- tuples of arguments

        Returns:
            *found head with POS-tag and position index* ``tower/NNS/12`` and
            the algorithm name

        """
        # NP head searching FSA
        postag = re.compile(r'NN|NNS|NNP|NNPS|VBG', flags=re.I)
        stack = deque()
        for word in tups:
            if not np:
                if postag.match(word[1]):
                    stack.append(word)
                    np = True
                elif re.match(r'PRP$', word[1], flags=re.I):
                    # no head found but we return single prp
                    return '/'.join(word), 'LINEAR'
            if np:
                if postag.match(word[1]):
                    stack.pop()
                    stack.append(word)
                elif re.match(r'POS', word[1], flags=re.I):
                    np = False
                else:
                    break
        if stack:
            return '/'.join(stack[-1]), 'LINEAR'
        else:
            return None  # head was not found


class MaltSpan:
    """
    MALT_SPAN strategy implementation.

    Attempt to find malt head when MALT index search fails.
    Search for the token with the lowest positional index within Malt
    tokens **block**, which is the size of the SENNA node. If token
    position with the lowest index is in a list of allowed heads, accept it
    as NP head. If there are several tokens with the same index, fall back
    to LINEAR.
    <If this is too confusing, see first the input xml file and find
    ``<malt></malt>`` tags. Then examine the example below.>

    Malt block::

          back rb 19 16 adv
          onto in 20 19 amod
          the dt 21 23 nmod
          shareholding nn 22 23 nmod
          register nns 23 20 pmod <-- lowest pos index 20

    SENNA node::

        <dep type="A2">
            back/RB/19 onto/IN/20 the/DT/21 shareholding/NN/22 register/NNS/23
        </dep>

    """
    @staticmethod
    def run(tups, remapped):
        """
        Run MALT_SPAN algorithm.

        Args:
            | *tups* (tuple) -- tuples of arguments
            | *remapped* (OrderedDict) -- a mapping of SENNA to Malt tokens
            | ``(('the', 1), ('dt', '1', '39')),``
            | ``(('rolling', 2), ('np', '2', '39')),``
            | ``(('stones', 3), ('np', '3', '39'))``
            | ``('the', 1)`` is SENNA token and its index
            | ``('dt', '1', '39')`` is Malt data
            | ``dt`` is Malt defined POS-tag for "the".
            | ``1`` is the token index.
            | ``39`` is the index of "the" governor (dependency link).

        Returns:
            *found head with POS-tag and position index* ``tower/NNS/12`` and
            the algorithm name

        """
        allowed_heads = ['nn', 'nns', 'np', 'nps', 'vbg', 'vbn', 'vvp']
        new_args = [(arg[0], int(arg[-1])) for arg in tups]
        tupargs_dup = dict(((arg[0], int(arg[-1])), arg) for arg in tups)
        itremapped = iter(remapped)
        remapped_copy = remapped.copy()
        for key in itremapped:
            if key not in new_args:
                del remapped_copy[key]
        if remapped_copy:
            # we shall not guess the head if no lowest index can be found
            if len(set([val for val in remapped_copy.values()])) > 1:
                head = min([(k, int(v[-1])) for k, v in remapped_copy.items()],
                           key=lambda x: x[1])
                tuparg_head = tupargs_dup.get(head[0])
                if tuparg_head[1] in allowed_heads:
                    return '/'.join(tuparg_head), 'MALT_SPAN'
        return None


class Malt:
    """
    MALT strategy implementation.
    In order to find the phrase head we analyze the results produced by
    Malt parser and SENNA. When SENNA processes the sentence it returns its
    governors and their dependents which we concatenate into role nodes.
    For example:
    *“Meg stood in the garden doorway , her small figure silhouetted against
    the twilight redness of the bay .”*

    The sentence above will have *“stand”* and *“silhouetted”* as governors
    where *“Meg”*, *“in the garden doorway”* shall be the role nodes of
    *“stand”* and *“her small figure”*, *“against the twilight redness of the
    bay”* shall be the role nodes of *“silhouetted”*.
    In addition to the information above we also have the following Malt parser
    output::

        token      POS-tag index depend role
        meg        np      1     2      sbj
        stand      vvd     2     0      root
        in         in      3     2      adv
        the        dt      4     6      nmod
        garden     nn      5     6      nmod
        doorway    nn      6     3      pmod
        ,          ,       7     11     p
        her        pp$     8     10     nmod
        small      jj      9     10     nmod
        figure     nn      10    11     sbj
        silhouette vvn     11    2      prn
        against    in      12    11     adv
        the        dt      13    15     nmod
        twilight   nn      14    15     nmod
        redness    nn      15    12     pmod
        of         in      16    15     nmod
        the        dt      17    18     nmod
        bay        nn      18    16     pmod
        .          sent    19    2      p

    The above example is retrieved from ``<malt></malt>`` elements in xml.
    In particular, we need to find the heads of the following three phrases:
    | *“in the garden doorway”*
    | *“her small figure”*
    | *“against the twilight redness of the bay”*

    This is done by finding SENNA's governor in Malt parser output, governor's
    immediate dependents and comparing them with the tokens from SENNA's role
    nodes.
    For our example the immediate dependents of *“stand”* are *“silhouette”*,
    *“in”* and the immediate dependents of *“silhouette”* are *“ , ”*,
    *“against”*, *“figure”*. These dependents (except *“ , ”*) are all present
    in the role nodes *“in the garden doorway”*, *“her small figure”*,
    *“against the twilight redness of the bay”* but not all of them are of
    interest to us. In particular we do not want to see prepositions as phrase
    heads, therefore the strategy skips them and returns the immediate
    dependents of the prepositions that are *“doorway”* for *“in”* and
    *“redness”* for *“against”*. Similarly, we skip any modal verbs and attempt
    to return their dependents as potential head candidates. If the phrase head
    is not a noun, verb or pronoun it is not accepted and the strategy falls
    back to the next searching strategy.

    """
    @staticmethod
    def run(tups, remapped, deptype, govs):
        """
        Use preprocessed malt data to find a tuple head. If the head can not be
        found fall back to head finding algorithm.

        Args:
            | *tups* (tuple) -- tuples of arguments
            | *remapped* (OrderedDict) -- a mapping of SENNA to Malt tokens
            | *deptype* (str) -- dependency type
            | *govs* (list) -- a list of governor tuples

        Returns:
            *found head with POS-tag and position index* ``tower/NNS/12`` and
            the algorithm name

        """
        head = None
        # creating a dict where {SENNA index: (MALT token, MALT index)}
        depdic = dict((v[1], v[0::2]) for v in remapped.values())
        updated_govs = [Malt._get_token_malt_idx(remapped, depdic, g, True)
                        for g in govs if g[0] not in punctuation]
        if not updated_govs or not updated_govs[0]:
            return None
        updated_args = [Malt._get_token_malt_idx(remapped, depdic, a)
                        for a in tups if a[0] not in punctuation]
        for arg in updated_args:
            if not arg:
                continue
            for gov in updated_govs:
                # (('take', 'VBmd'), '5', '4')
                if gov[0][1][-2:] == 'md':
                    # we check gov[-1] in case it points to md
                    if arg[-1] == gov[-1]:
                        head = arg[0]
                        break
                elif gov[0][1][-2:] == 'be':
                    if arg[-1] == gov[-1]:
                        head = arg[0]
                        break
                elif arg[-1] == gov[1]:
                    head = arg[0]
                    break
        # if head is prep skip and search for next head
        if head and head[1] in ['in', 'to']:
            return Malt.run(tups, remapped, deptype, (head,))
        if head:
            return '/'.join(head), 'MALT'
        else:
            return None

    @staticmethod
    def _get_token_malt_idx(malt_map, depdic, token, isgov=False):
        """
        Attempt to find a corresponding token in malt parse and replace the
        ukwac index with Malt index. Use mapping dict to lookup Malt indeces
        and return SENNA token with Malt index.

        <There are couple of corrections I make due to MALT <> SENNA parse
        differences. I am checking whether a verb has any ``md``, ``vbz``,
        ``vhz`` etc. parents. If it does, I mark this verb so that later we
        could correctly identify the true parent for current token.
        Here is an example of Malt output to understand better what I mean::

            she pp 1 2 sbj -->
            could md 2 0 root <--
            see vv 3 2 vc

        Notice how *"she"* points to ``md`` parent in MALT (while SENNA will
        have *"see"* as a parent).

        Another example (malt output)::

            the dt 1 2 nmod
            bird nn 2 3 sbj -->
            be vbd 3 0 root <--
            go vvn 4 3 vc
            . sent 5 3 p

        Notice how *"bird"* points to ``vbd`` instead of *"go"*.
        This is not the only MALT <> SENNA parse difference, but one which
        requires separate handling. I have created unit-tests for such cases.
        Please run unit-tests each time after you change anything.>

        Args:
            | *malt_map* (OrderedDict) -- a mapping of SENNA to MALT tokens
            | *depdic* (dict) -- a mapping from SENNA index to MALT index, tag
            | *token* (tuple) -- SENNA tuple ('settle', 'VB', '4'), which is
               used to get MALT values from *malt_map*
            | *isgov* (bool) -- True, if token is a governor and should be
               handled in a special way

        Returns:
            a tuple of SENNA and Malt token indeces where ``37`` corresponds to
            Malt token index and ``35`` points to the token's governor::

                (('like', 'VBD', '36'), '37', '35')

        """
        lookup_token = (token[0], int(token[2]))
        senna_postag = token[1].lower()
        malt_value = malt_map.get(lookup_token)
        # if token is not in remapped_malt then malt <> senna tagging
        if not malt_value:
            return None
        # checking if vv has any md parents, because MALT makes md root
        if isgov:
            if malt_value[-1] != '0' and malt_value[0][0] == 'v':
                # if here is md 0 then we update vv
                if depdic.get(malt_value[-1]) == ('md', '0'):
                    # ('take', 'VBmd', '5', '4') where 4 points to md parent
                    token = token[0], token[1] + 'md'  # marking the token
                # if we find that vv has a ["be", "have"] 0 root we update vv
                behave = ['vbz', 'vbd', 'vbp', 'vhz', 'vhd', 'vhp']
                parent = depdic.get(malt_value[-1])
                if parent and parent[0] in behave and parent[-1] == '0':
                    token = token[0], token[1] + 'be'  # marking the token
        # checking POS-tag equality,
        # this is necessary to be sure that correct tokens are merged
        if senna_postag[0] == malt_value[0][0]:  # 'vb' == ('vbn', 1, 0)
            return token, malt_value[1], malt_value[2]
        elif senna_postag == '.' and malt_value[0] == 'sent':
            return token, malt_value[1], malt_value[2]
        elif senna_postag in ['cd', 'nn'] and malt_value[0] == 'ls':
            return token, malt_value[1], malt_value[2]


def read_result_xml(fname):
    """
    Read xml file and return xml.etree object

    Args:
        *fname* (str) -- file name
    Returns:
        *etrees* (xml.etree) -- xml representation of file contents

    """
    def rewrite_orig(etr_line,erase=False):
        # rewrite the original file without malt block,lemsent and rawsent
        try:
            with gzip.open(fname, 'a') as f:
                f.write(etr_line)
                f.write('\n')
        except IOError:
            with open(fname, 'a') as f:
                f.write(etr_line)
                f.write('\n')

    # parse xml file
    try:
        with gzip.open(fname, 'r') as f:
            fdata = f.read()
    except IOError:
        with open(fname, 'r') as f:
            fdata = f.read()
    etrees = od()
    for sent in enumerate(fdata.split('\r\n')):
        if not sent[1]:
            continue
        etrees[sent[0]] = et.fromstring(sent[1])

    # we need to remove "lemsent","rawsent" and "malt" from input file
    # erase orig file
    try:
        with gzip.open(fname,'w') as f:
            pass
    except IOError:
        with open(fname, 'w') as f:
            pass

    etrees_copy = copy.deepcopy(etrees)
    for etr in etrees_copy.values():
        # removing <malt> block
        malt_element = etr.find('malt')
        if malt_element is not None:
            etr.remove(malt_element)
        # removing <rawsent> and <lemsent>
        rawsent = etr.find('rawsent')
        if rawsent is not None:
            etr.remove(rawsent)
        lemsent = etr.find('lemsent')
        if lemsent is not None:
            etr.remove(lemsent)
        etr_str = et.tostring(etr)
        rewrite_orig(etr_str)
    return etrees


def map_to_malt_indexes(lemsent, maltdata):
    """
    Assign Malt block indexing to ``<lemsent>`` tokens, because ``<lemsent>``
    number of tokens can and is often different from Malt's.

    <If we do not do this line rearrangement, we won't be able to make use of
    Malt block in our xml input file. Malt block contains **dependent ->
    governor** indeces that we use to find a phrase head.>

    <Maltdata shall have all non-normalized contractions of malt parse which,
    in case of SENNA's output, are removed by ``reduce_contractions`` function
    from **ukwac_converter.py** script and therefore do not get into
    ``lemsent``. This is also one of the reasons for SENNA - Malt index
    differences.>

    Args:
        | *lemsent* (str) -- a string from ``<lemsent></lemsent>``
        | *maltdata* (str) -- a string from ``<malt></malt>``

    Returns:
        | *result* (OrderedDict) -- a mapping between SENNA's tokens and Malt
          block, where first tuple is SENNA's ``(token, index)`` and second
          tuple is Malt's ``('POS-tag', "token's position", "token's governor
          position")``::

            [(('the', 1), ('dt', '1', '39')),
            (('rolling', 2), ('np', '3', '39')),
            (('stones', 3), ('np', '4', '39')),]

        <The above example does not show any difference between SENNA and Malt
        indeces, however in many cases Malt index shall be different.>

    """
    def line_rearrange(ml, ll, mi, li):
        """
        Fix line discrepancies between SENNA and Malt data.
        Iterate over SENNA's tokens and attempt to find idential tokens in
        Malt block.

        <This function was basically created by "trial and error" and it is
        highly obscure. The idea here is to iteratively search over SENNA or
        Malt lines looking for correct match. It involves a series of string
        checks (first strict ones like **full string match**, then
        ``startswith``, then **3 first chars only** in order to find the
        "perfect" match). Keep in mind that if you decide to change anything,
        do run unittests (and even unittests won't give you 100% guarantee:).
        This algorithm is very fragile.>

        Args:
            | *ml* (list) -- a list of Malt block elements.
              Example: ``['you', 'pp', '34', '35', 'sbj']``
            | *ll* (tuple) -- a tuple of SENNA's token and its index as
              retrieved from ``<lemsent></lemsent>``.
              Example: ``("you've", 34)``
            | *mi* (str) -- Malt (``<malt></malt>``) block index
            | *li* (str) -- SENNA (``<lemsent></lemsent>``) index

        Returns:
            *a tuple* with rearranged indeces.
            Example: ``(('always', 35), 'rb', '36', '35', 35, 34)``

        """
        # DRAGONS AHEAD!
        # adding mismatched word before iteration
        # when token diff falls on the last token
        if len(malt_lst) == mi or len(malt_lst) == mi - 1:
            return ll, ml[1], ml[2], ml[3], mi, li
        # when token diff falls on the last token
        if len(lemsent_lst) == li or len(lemsent_lst) == li - 1:
            return ll, ml[1], ml[2], ml[3], mi, li
        result[ll] = ml[1], ml[2], ml[3]
        itermalt = False
        iterlem = False
        while True:
            ml = malt_lst[mi].split()
            ll = lemsent_lst[li]
            if ml[0] == ll[0]:
                return ll, ml[1], ml[2], ml[3], mi, li
            elif ml[0].startswith(ll[0]):
                iterlem = True
                mi += 1
                li += 1
            elif ll[0].startswith(ml[0]):
                itermalt = True
                mi += 1
                li += 1
            # we need to check also 'give'<>'giving' diffs, match first 3 chars
            elif ml[0][:3] == ll[0][:3]:
                return ll, ml[1], ml[2], ml[3], mi, li
            elif iterlem:
                li += 1
            elif itermalt:
                mi += 1
            elif len(ml[0]) == len(ll[0]):
                result[ll] = ml[1], ml[2], ml[3]
                mi += 1
                li += 1
            elif len(ml[0]) > len(ll[0]):
                iterlem = True
                mi += 1
                li += 1
            elif len(ml[0]) < len(ll[0]):
                itermalt = True
                mi += 1
                li += 1
            """
            <Surprise! SENNA can randomly split a sentence in halves, then
            ``li`` will exceed ``lemsent_lst`` and iteration stops. It may not
            happen in this loop however and that is why here in the end I
            explicitly check the last elements of ``malt_lst`` to see if
            the sent was actually a splitted part. I return ``('BROKEN')`` so
            that ``line_rearrange`` knows the sentence was cut off.>
            """
            if li == len(lemsent_lst) and li < len(malt_lst):
                return ('BROKEN',)  # sentence was broken
            # if reached end of either of lists returning current line
            if mi == len(malt_lst):
                return ll, ml[1], ml[2], ml[3], mi, li
            if li == len(lemsent_lst):
                return ll, ml[1], ml[2], ml[3], mi, li

    global WASBROKEN
    global STARTFROM
    global PREV_MALT
    maltdata = re.sub(r"n't", "not", maltdata)  # replace "n't" with "not"
    malt_lst = [i.lower() for i in maltdata.split('\n') if i]
    lemsent_lst = [(i[1].lower(), i[0]) for i in enumerate(lemsent.split(), 1)
                   if i]
    if PREV_MALT == maltdata and WASBROKEN:
        midx = STARTFROM
        lidx = 1  # assuming that the first word was garbled after the split
    else:
        midx = 0
        lidx = 0
    result = od()
    # if our lemsent contains only one word we return current tokens
    if len(lemsent_lst) == 1:
        mline = malt_lst[midx].split()
        lline = lemsent_lst[0]
        result[lline] = mline[1], mline[2], mline[3]
        return result
    while True:
        mjump = 0
        ljump = 0
        mline = malt_lst[midx].split()
        lline = lemsent_lst[lidx]
        result[lline] = mline[1], mline[2], mline[3]
        if mline[0] != lline[0]:
            if lline[0] == '@card@':  # because 60%!=@card@ triggers inf loop
                result[lline] = mline[1], mline[2], mline[3]
                midx += 1
                lidx += 1
            # we need to check first if SENNA token exists in MALT data
            exists = [e for e in malt_lst if lline[0][:2] in e.split()[0]]
            # now we need to be sure it was not "'s" contraction of "be"
            if not exists and lline[0][:2] == "'s":
                exists = [e for e in malt_lst if "be" in e.split()[0]]
            if not exists:
                lidx += 1
            restuple = line_rearrange(mline, lline, midx, lidx)
            if len(restuple) == 1:  # sent is broken, saving split point
                WASBROKEN = True
                break
            lline, pos, new_idx, dep, m, l = restuple
            result[lline] = pos, new_idx, dep
            mjump = m - midx
            ljump = l - lidx
        midx += (1 + mjump)
        lidx += (1 + ljump)
        if len(malt_lst) - 1 < midx:
            break
        if len(lemsent_lst) - 1 < lidx:
            break
    """
    Checking if current sent was indexed to the end. This is necessary because
    SENNA might split the sent in parts and we need to know if the current
    sent is indeed the last part. If it is, we reset WASBROKEN and STARTFROM
    globals
    """
    lemsent_last = result[next(reversed(result))]
    malt_last = next(reversed(malt_lst)).split()[2:4]
    if lemsent_last[1:] == tuple(malt_last):
        WASBROKEN = False
        STARTFROM = 0
        PREV_MALT = False
    else:
        WASBROKEN = True
        # STARTFROM = int(lemsent_last[1])
        itmalt = iter(malt_lst)
        break_line = ' '.join(result[next(reversed(result))])
        STARTFROM = malt_lst.index([next(itmalt) for e in itmalt
                                    if str(break_line) in e][0])
        PREV_MALT = maltdata
    return result


def alg_controller(tuparg, remapped, deptype, govs):
    """
    Switch between various head searching strategies.
    If first strategy fails, fallback to the next one until the phrase head
    is found or not.

    Args:
        | *tuparg* (tuple) -- tuples of SENNA's tokens with POS-tags and
          indeces.

          Example::

                (('the', 'dt', '1'),
                 ('rolling', 'nnp', '2'),
                 ('stones', 'nnps', '3'),
                 (',', ',', '4'),
                 ('genesis', 'nnp', '5')

        | *remapped* (OrderedDict) -- a mapping between SENNA's tokens and Malt
          block, where first tuple is SENNA's ``(token, index)`` and second
          tuple is Malt's ``('POS-tag', "token's position", "token's governor
          position")``
        | *deptype* (str) -- dependency type string
        | *govs* (tuple) -- governor tuple as retrieved from SENNA's output.
          Example: ``[('pretend', 'VBP', '33')]``

    Returns:
        *head* (tuple) -- phrase head found by one of the strategies.
         Example: ``('september/nnp/25', 'LINEAR')``

    """
    # running algorithms
    head = Malt.run(tuparg, remapped, deptype, govs)
    # if head was not found by MALT switch to MALT_SPAN
    if not head and is_called('MALT_SPAN'):
        head = MaltSpan.run(tuparg, remapped)
    # if head was not found by MALT_SPAN switch to LINEAR
    if not head and is_called('LINEAR'):
        head = Linear.run(tuparg, deptype)
    # if head not found return full chunk and flag as FAILED
    if not head:
        return ' '.join(['/'.join(i for i in w) for w in tuparg]), 'FAILED'
    return head


def is_called(*calling_algs):
    """
    Use global ALGS to trigger head searching algorithms that we explicitly
    defined with `-a` option.
    """
    return [True for a in calling_algs if a in ALGS]


def head_searcher(args):
    """
    Do various postprocessing steps (see code comments).
    Assign globals, system paths, save results and clean up the left over
    files.

    Args:
        *args* (dict) -- a dict of arguments supplied by the user

    """
    def wrap(fname, head):
        """
        Write  ``<sents>``, ``</sents>`` to the result file.
        So that tensor.py could parse it afterwards.
        """
        with open(fname, 'a') as fopen:
            if head:
                fopen.write('<sents>\n')
                return 0
            else:
                fopen.write('</sents>')
        with open(fname, 'r') as fopen:
            fdata = fopen.read()
        with gzip.open(fname+'.gz', 'w') as gf:
            gf.write(fdata)
        os.remove(fname)

    def write_etree(fname, etr):
        with open(fname, 'a') as fopen:
            fopen.write(etr)
            fopen.write('\n')  # sentences delimiter

    # defining algorithms to use
    global ALGS
    ALGS = args.alg.split(',')
    # handle user specified args
    try:
        if not args.file:
            sge_id = '%04d' % (int(os.environ.get('SGE_TASK_ID')) - 1)
            # if args.dir is not set but args.file is set we just pass
            name = os.path.join(os.getcwd(), args.dir,
                                'ukwac.fixed-nowiki.node' + sge_id +
                                '.converted.xml.gz')
            files = [name]
        else:
            files = [args.file]
    except (TypeError, AttributeError):
        if not args.file and args.dir:
            files = [os.path.join(args.dir, name)
                     for name in os.listdir(args.dir)]
        elif args.file:
            files = [args.file]
        else:
            print("ERROR: Please provide input dir or input file name.")
            sys.exit()

    if args.file and args.dir:
        msg = "Please provide only the file or the directory!"
        raise ValueError(msg)
    elif not args.file and not args.dir:
        msg = "Please provide the files to process!"
        raise ValueError(msg)

    if args.out:
        try:
            os.mkdir(args.out)
        except OSError:
            print("Can't create {0} or it already exists".format(args.out))
        output_dir = os.path.join(os.getcwd(), args.out)
    else:
        output_dir = os.getcwd()
    # begin processing
    for fname in files:
        current_fname = 'heads.' + os.path.basename(fname).rstrip('.gz')
        if args.qsub:
            # we copy the file to remote machine for faster processing
            if not os.path.isdir('/local/pavels'):
                os.makedirs('/local/pavels')
            shutil.copy(fname, '/local/pavels')
            os.chdir('/local/pavels')
            current_fname = os.path.join('/local/pavels', 'heads.' +
                                         os.path.basename(fname).rstrip('.gz'))
        print('PROCESSING FILE: {0}'.format(current_fname))  # debug only
        parsed_etrees = read_result_xml(os.path.basename(fname))
        wrap(current_fname, True)
        for etr in parsed_etrees.values():
            if etr.tag == 's':
                malt_text = etr.findtext('malt')  # complete malt parse
                lem_sent = etr.findtext('lemsent')  # lemsent build from malt
                # lem_sent = etr.findsghtext('sent')  # lemsent from malt
                # rawsent = etr.findtext('rawsent')  # rawsent from .output
                remapped = {}
                # return True if algorithm is allowed
                if is_called('MALT', 'MALT_SPAN'):
                    remapped = map_to_malt_indexes(lem_sent, malt_text)
            malt_element = etr.find('malt')
            if malt_element is not None:
                etr.remove(malt_element)

            # sent = etr.find('sent')
            # if sent is not None: etr.remove(sent)

            # before processing we copy original nodes into "source" attrib
            for deps in etr.iter('dependencies'):
                for dep in deps.iter('dep'):
                    dep.set('source', dep.text)
                    # include source without POS-tags and position indeces
                    orig_senna = ' '.join([tok.rsplit('/',2)[0]
                                          for tok in dep.text.split()])
                    dep.set('text', orig_senna)

            # begin head search processing
            for pred in etr.iter('predicate'):
                govs = [tuple(g.text.rsplit('/', 2))
                        for g in pred.iter('governor')]
                for dep in pred.iter('dep'):
                    if dep.attrib['type'] in 'V':
                        continue
                    tup_arg = tuple([tuple(t.lower().rsplit('/', 2))
                                     for t in dep.text.split()])
                    # starting head search
                    head = alg_controller(tup_arg, remapped,
                                          dep.attrib['type'],
                                          govs)
                    dep.text = head[0]
                    dep.set('algorithm', head[-1])
            # removing <rawsent> and <lemsent>
            rawsent = etr.find('rawsent')
            if rawsent is not None:
                etr.remove(rawsent)
            lemsent = etr.find('lemsent')
            if lemsent is not None:
                etr.remove(lemsent)
            write_etree(current_fname, et.tostring(etr).lower())
        wrap(current_fname, False)

        if args.qsub:
            # when done processing
            clean_up(current_fname, output_dir)


def clean_up(fname, output_dir):
    """
    Remove input and output files after processing.

    Args:
        | *fname* (str) -- file name
        | *output_dir* (str) -- dir where the results are saved

    """
    try:
        sb.call(['ssh', 'forbin'])
        shutil.move(fname+'.gz', output_dir)
        #os.remove(fname)
    except RuntimeError:
        print 'WARNING! clean_up() failed'


WASBROKEN = False
STARTFROM = 0
PREV_MALT = False


if __name__ == '__main__':
    prs = argparse.ArgumentParser(description="""
    This script parses files processed by SENNA role labeler and attempts to
    find phrase heads. In order to accomplish this, it first tries to use malt
    parse information that comes from original ukwac corpus and if this fails
    it uses slightly less strict malt span search, if the two above still fail
    it executes head searching based on Magerman's head percolation table.""")
    prs.add_argument('-d', '--dir',
                     help='Specify directory where processed ukwac files are \
                     located.',
                     required=False)
    prs.add_argument('-f', '--file',
                     help='Specify an xml file to process.',
                     required=False)
    prs.add_argument('-o', '--out',
                     help='Specify output directory. If not specified \
                     current dir is used.',
                     required=False)
    prs.add_argument('-qs', '--qsub', action='store_true',
                     help='Use this option if you run the script on forbin \
                       with qsub. The script will copy the required files \
                       under its /local instance dir and copy the results \
                       to /local/pavels',
                     required=False)
    prs.add_argument('-a', '--alg',
                     default='MALT,MALT_SPAN,LINEAR',
                     help='Choose a head searching strategy. Available \
                     strategies: MALT, MALT_SPAN, LINEAR. Usage example: \
                     ./head_searcher.py --alg MALT,MALT_SPAN.',
                     required=False)
    head_searcher(prs.parse_args())

    ### uncomment below to generate script execution .png graph
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'head_searcher.png'
    # with PyCallGraph(output=graphviz):
    #    head_searcher(arguments)
