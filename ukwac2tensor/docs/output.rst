Output description
==================

Here you can find the I/O description of the complete pipeline as well as the examples of **ukwac2tensor** **INPUT** and **OUTPUT** files.


INPUT (ukWaC corpus file) -> ukwac_converter.py 
---------------------------
**ukwac.fixed-nowiki.node0031.gz**::

    <s>
    I	I	PP	1	2	SBJ
    took	take	VVD	2	0	ROOT
    very	very	RB	3	4	AMOD
    little	little	JJ	4	5	NMOD
    part	part	NN	5	2	OBJ
    in	in	IN	6	2	ADV
    the	the	DT	7	8	NMOD
    conversation	conversation	NN	8	6	PMOD
    .	.	SENT	9	2	P
    </s>



ukwac_converter.py -> OUTPUT -> head_searcher.py
----------------------------
**ukwac.fixed-nowiki.node0031.converted.xml.gz**::

    <s>
       <predicate>
          <governor>take/VBD/2</governor>
          <dependencies>
             <dep type="A0">I/PRP/1</dep>
             <dep type="V">take/VBD/2</dep>
             <dep type="A2">very/RB/3 little/JJ/4</dep>
             <dep type="A1">part/NN/5</dep>
             <dep type="AM-LOC">in/IN/6 the/DT/7 conversation/NN/8</dep>
          </dependencies>
       </predicate>
    </s>


head_searcher.py -> OUTPUT -> tensor.py
------------------
**heads.ukwac.fixed-nowiki.node0031.converted.xml.gz**::

    <s>
       <predicate>
          <governor>take/VBD/2</governor>
          <dependencies>
             <dep algorithm="MALT" source="I/PRP/1" text="I" type="A0">i/prp/1</dep>
             <dep source="take/VBD/2" text="take" type="V">take/VBD/2</dep>
             <dep algorithm="FAILED" source="very/RB/3 little/JJ/4" text="very little" type="A2">very/rb/3 little/jj/4</dep>
             <dep algorithm="MALT" source="part/NN/5" text="part" type="A1">part/nn/5</dep>
             <dep algorithm="MALT" source="in/IN/6 the/DT/7 conversation/NN/8" text="in the conversation" type="AM-LOC">conversation/nn/8</dep>
          </dependencies>
       </predicate>
    </s>
  

OUTPUT XML tags description
---------------------------
| ``<sents>`` -- a root element of the xml file, contains <text> and <s>.
|
| ``<text>`` -- contains "id" attribute that is a url address of the sentences below until next <text> element.
|
| ``<s>`` -- an element that contains a sentence processed by head searching algorithms.
|
| ``<predicate>`` -- sentence predicate block.
|
| ``<governor>`` -- governor of the respective predicate block, e.g. examine/VBZ/12, where examine is a word, VBZ is a POS-tag and 12 is word index in a sentence <s> element.
|
| ``<dependencies>`` -- a block of dependencies for the current sentence governor, contains <dep> elements.
|
| ``<dep>`` -- a single dependency of a current governor, contains a head word extracted by one of the head searching algorithms, e.g. state/nn/16 where state is a head word, nn is a POS-tag, 16 is a word index. The element also contains the following important attributes:

   |
   | ``algorithm`` -- contains a name of a head searching algorithm used for head finding, e.g. MALT, MALT_SPAN, LINEAR or FAILED if no algorithm was able to find a head.
   | ``source`` -- contains tokenized, lemmatized and POS-tagged clause governed by the <governor>. This clause is used by head searching algorithms.
   | ``text`` -- contains tokenized, lemmatized clause governed by the <governor>. Similar to "source" but without POS-tags. 
   | ``type`` -- dependency type that follows `PropBank annotation <https://verbs.colorado.edu/~mpalmer/projects/ace/PBguidelines.pdf>`_.


tensor.py -> OUTPUT
-------------------
**tensor.MALT-MALT_SPAN-LINEAR.h5**::

    import pandas as pd
    tensor = pd.read_hdf('tensor_pmi.MALT-MALT_SPAN-LINEAR.h5', key='tensor')
