extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General substitutions.
project = "PyCUDA"
copyright = "2008-20, Andreas Kloeckner"

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
ver_dic = {}
exec(
    compile(
        open("../../pycuda/__init__.py").read(), "../../pycuda/__init__.py", "exec"
    ),
    ver_dic,
)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# Options for HTML output
# -----------------------

html_theme = "furo"


intersphinx_mapping = {
    "https://docs.python.org/3": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/codepy/": None,
}

autoclass_content = "class"
autodoc_typehints = "description"
