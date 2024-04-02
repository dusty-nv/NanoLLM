# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

import nano_llm

project = 'NanoLLM'
copyright = 'NVIDIA'
author = 'Dustin Franklin'
version = nano_llm.__version__
release = nano_llm.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser', 
    'sphinx.ext.autodoc', 
    'sphinx.ext.autosummary', 
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    #'sphinxcontrib_autodocgen'
]

DOC_SOURCE_DIR = os.path.dirname(__file__)
PROJECT_ROOT_DIR = os.path.abspath(DOC_SOURCE_DIR+'/..')

sys.path.append(PROJECT_ROOT_DIR)

'''
autodocgen_config = [{
        'modules':[example],
        'generated_source_dir': os.path.join(DOC_SOURCE_DIR, 'api/'), #DOC_SOURCE_DIR+'/autodocgen/',

        # if module matches this then it and any of its submodules will be skipped
        'skip_module_regex': '(.*[.]__|myskippedmodule)',

        # produce a text file containing a list of everything documented. you can use this in a test to notice when you've
        # intentionally added/removed/changed a documented API
        'write_documented_items_output_file': 'autodocgen_documented_items.txt',

        # customize autodoc on a per-module basis
        #'autodoc_options_decider': {
        #        'mymodule.FooBar':    { 'inherited-members':True },
        #},

        # choose a different title for specific modules, e.g. the toplevel one
        'module_title_decider': lambda modulename: 'API Reference' if modulename=='example' else modulename,
}]
'''

autodoc_default_options = {
	'show-inheritance': True, # show base classes
    #'members': 'var1, var2',
    'member-order': 'bysource',
    #'special-members': '__init__',
    #'undoc-members': True,
    # The supported options are 'members', 'member-order', 'undoc-members', 'private-members', 'special-members', 'inherited-members', 'show-inheritance', 'ignore-module-all', 'imported-members' and 'exclude-members'.
    #'exclude-members': '__weakref__'
}

#autoclass_content = 'both' # include __init__ params in doc strings for class
#autodoc_member_order = 'bysource'
#autosummary_generate_overwrite = False
autosummary_generate = True

add_module_names = False
add_function_parentheses = True

highlight_language = 'python3' # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#literal-blocks

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
	'display_version': False,
	#'prev_next_buttons_location': 'bottom',
	#'style_external_links': False,
	#'vcs_pageview_mode': '',
	#'style_nav_header_background': 'white',
	# Toc options
	'collapse_navigation': True,
	'sticky_navigation': True,
	#'navigation_depth': 4,
	'includehidden': False,
	#'titles_only': False,
    #'style_nav_header_background': '#5d7567', #'566c5f', '#4e5c54',
}

# https://github.com/pysys-test/pysys-test/blob/master/docs/conf.py#L163C1-L166C3
#html_context = {'css_files': [
#	# Workaround for RTD 0.4.3 bug https://github.com/readthedocs/sphinx_rtd_theme/issues/117
#	'_static/theme_overrides.css',  # override wide tables in RTD theme
#]}
