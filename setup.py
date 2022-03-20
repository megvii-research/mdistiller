"""MDistiller: a deep learning toolkit for knowledge distillation.
"""

import os.path
import sys
import setuptools


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mdistiller"))
    DISTNAME = "mdistiller"
    DESCRIPTION = "A Deep Learning Toolkit for Knowledge Distillation."
    AUTHOR = "zhaoborui"
    DOCLINES = __doc__

    setuptools.setup(
        name=DISTNAME,
        packages=setuptools.find_packages(),
        version="0.1",
        description=DESCRIPTION,
        long_description=DOCLINES,
        long_description_content_type="text/markdown",
        author=AUTHOR,
    )
