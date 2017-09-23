Introduction
============

The goal of **ukwac2tensor** project is to convert `ukWaC corpus <http://wacky.sslmit.unibo.it/doku.php?id=corpora>`_
into 3-d order tensor.

**ukwac2tensor** represents conversion pipeline which consists of three scripts **"ukwac_converter.py"**,
**"head_searcher.py"**, **"tensor.py"**.

Each script representes a part of the pipeline such that the output of the first script serves
as the input for the second and so on.

In order to convert **ukWaC** corpus you need to run **ukwac2tensor** scripts in the following order:
    1. :ref:`ukwac_converter`
    2. :ref:`head_searcher`
    3. :ref:`tensor`

Next see :ref:`howto` section.
