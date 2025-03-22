# This file makes the components directory a Python package
# Import components to make them available at the package level
from components.document_uploader import render_document_uploader
from components.similarity_viewer import render_similarity_heatmap, render_similarity_network, render_document_comparison_view