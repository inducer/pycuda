from __future__ import annotations


try:
    import mako.template  # ruff:ignore[unused-import]
except ImportError:
    raise ImportError(
        "Some of PyCUDA's facilities require the Mako templating engine.\n"
        "You or a piece of software you have used has tried to call such a\n"
        "part of PyCUDA, but there was a problem importing Mako.\n\n"
        "You may install mako now by typing one of:\n"
        "- easy_install Mako\n"
        "- pip install Mako\n"
        "- aptitude install python-mako\n"
        "\nor whatever else is appropriate for your system."
    )


from mako import *  # ruff:ignore[undefined-local-with-import-star]
