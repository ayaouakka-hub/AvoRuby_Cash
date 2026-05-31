import os, sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'AvoRuby Cash'
author = 'Aya Ouakka, Hiba Boutahir'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

# Fix: active les ancres de titres pour la table des matières
myst_heading_anchors = 3

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static', 'app_images']
html_extra_path = ['app_images']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

html_context = {
    'supervisor': 'Prof. Tawfik Masrour',
    'institution': 'ENSAM Meknes',
    'year': '2025–2026',
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}