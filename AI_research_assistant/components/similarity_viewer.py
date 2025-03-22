import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional

def render_similarity_heatmap(similarity_matrix: List[List[float]], 
                              doc_names: List[str],
                              threshold: float = 0.7,
                              title: str = "Document Similarity Heatmap"):
    """
    Render a heatmap visualization of document similarity.
    
    Args:
        similarity_matrix: 2D matrix of similarity scores
        doc_names: List of document names/labels
        threshold: Similarity threshold for highlighting
        title: Title for the heatmap
    """
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=doc_names,
        y=doc_names,
        colorscale='Blues',
        zmin=0, 
        zmax=1,
        hoverongaps=False,
        text=[[f"{val:.2f}" for val in row] for row in similarity_matrix],
        hoverinfo="text",
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        xaxis=dict(title="Documents"),
        yaxis=dict(title="Documents")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display a legend explaining the color scale
    st.markdown("""
    **Color Intensity Legend:**
    * **Dark blue**: High similarity (potential content overlap)
    * **Light blue**: Low similarity (different content)
    """)

def render_similarity_network(similarity_matrix: List[List[float]], 
                             doc_names: List[str],
                             threshold: float = 0.7,
                             title: str = "Document Similarity Network"):
    """
    Render a network visualization of document similarity relationships.
    
    Args:
        similarity_matrix: 2D matrix of similarity scores
        doc_names: List of document names/labels
        threshold: Similarity threshold for creating edges
        title: Title for the network diagram
    """
    # Create node positions in a circle layout
    n_docs = len(doc_names)
    angles = np.linspace(0, 2*np.pi, n_docs, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Create edges (connections) based on similarity threshold
    edge_x = []
    edge_y = []
    edge_text = []
    
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            if similarity_matrix[i][j] >= threshold:
                # Add edge between nodes i and j
                edge_x.extend([x_pos[i], x_pos[j], None])
                edge_y.extend([y_pos[i], y_pos[j], None])
                edge_text.append(f"Similarity: {similarity_matrix[i][j]:.2f}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.8)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=x_pos, 
        y=y_pos,
        mode='markers+text',
        text=doc_names,
        textposition="bottom center",
        marker=dict(
            size=20,
            color='skyblue',
            line=dict(width=1, color='darkblue')
        ),
        hoverinfo='text'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600
                   ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display legend
    st.markdown("""
    **Network Diagram Legend:**
    * **Nodes**: Individual documents
    * **Edges**: Connections between similar documents (similarity ≥ threshold)
    """)

def render_document_comparison_view(doc1_chunks: List[Dict], 
                                   doc2_chunks: List[Dict], 
                                   similarity_scores: List[float],
                                   threshold: float = 0.7):
    """
    Render a side-by-side comparison view of two documents with highlighted similar chunks.
    
    Args:
        doc1_chunks: List of chunks from document 1
        doc2_chunks: List of chunks from document 2
        similarity_scores: List of similarity scores between chunk pairs
        threshold: Similarity threshold for highlighting
    """
    st.subheader("Document Comparison View")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Document 1")
        for i, chunk in enumerate(doc1_chunks):
            chunk_text = chunk.get("text", "")
            
            # Determine if this chunk has a similar match
            has_similar = any(score >= threshold for score in similarity_scores[i::len(doc1_chunks)])
            
            if has_similar:
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #FF5733; margin-bottom: 10px;">
                    <p>{chunk_text}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #3498DB; margin-bottom: 10px;">
                    <p>{chunk_text}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Document 2")
        for i, chunk in enumerate(doc2_chunks):
            chunk_text = chunk.get("text", "")
            
            # Determine if this chunk has a similar match
            chunk_scores = [similarity_scores[j*len(doc2_chunks) + i] for j in range(len(doc1_chunks))]
            has_similar = any(score >= threshold for score in chunk_scores)
            
            if has_similar:
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #FF5733; margin-bottom: 10px;">
                    <p>{chunk_text}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #3498DB; margin-bottom: 10px;">
                    <p>{chunk_text}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Add legend
    st.markdown("""
    **Color Legend:**
    * <span style="color:#FF5733">■</span> **Orange border**: Chunk has similar content in the other document
    * <span style="color:#3498DB">■</span> **Blue border**: Chunk is unique to this document
    """, unsafe_allow_html=True)
    
def show_similarity_stats(similarity_matrix: List[List[float]], 
                         doc_names: List[str],
                         threshold: float = 0.7):
    """
    Display similarity statistics between documents.
    
    Args:
        similarity_matrix: 2D matrix of similarity scores
        doc_names: List of document names/labels
        threshold: Similarity threshold for highlighting
    """
    st.subheader("Similarity Statistics")
    
    n_docs = len(doc_names)
    
    # Calculate average similarity for each document
    avg_similarities = [
        sum(similarity_matrix[i][j] for j in range(n_docs) if j != i) / (n_docs - 1)
        for i in range(n_docs)
    ]
    
    # Calculate number of similar documents for each document
    similar_counts = [
        sum(1 for j in range(n_docs) if j != i and similarity_matrix[i][j] >= threshold)
        for i in range(n_docs)
    ]
    
    # Create dataframe for display
    data = {
        "Document": doc_names,
        "Avg. Similarity": [f"{avg:.2f}" for avg in avg_similarities],
        "Similar Documents": similar_counts
    }
    
    # Display as table
    st.table(data)