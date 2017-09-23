The goal of **ukwac2tensor** project is to convert **ukWaC corpus** into 3-d order tensor.

ukwac2tensor is a corpus conversion pipeline which consists of three scripts `ukwac_converter.py`, `head_searcher.py`, `tensor.py`.

Each script representes a part of the pipeline such that the output of the first script serves as the input for the second and so on.

In order to convert ukWaC corpus you need to run ukwac2tensor scripts in the following order:

        ukwac_converter.py
        head_searcher.py
        tensor.py

Next see `docs/html/index.html`.
