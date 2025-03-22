import streamlit as st
import os
import asyncio
import json
import pandas as pd
import time
import uuid
import sys
import plotly.graph_objects as go

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.document_processor import DocumentProcessor
from utils.config import load_config
from components.document_uploader import render_document_uploader
from utils.text_analysis import TextAnalyzer

# Set page config
st.set_page_config(
    page_title="Contextual Quality Assessment",
    page_icon="📊",
    layout="wide"
)

# Load config
config = load_config()

# Page title
st.title("Contextual Quality Assessment")
st.markdown("Evaluate argument strength, logical coherence, and research validity with AI-powered analysis.")

# Initialize session state
if 'quality_results' not in st.session_state:
    st.session_state.quality_results = None
if 'doc_for_analysis' not in st.session_state:
    st.session_state.doc_for_analysis = None
if 'in_progress' not in st.session_state:
    st.session_state.in_progress = False

# Get services from session state
llm_service = st.session_state.llm_service if 'llm_service' in st.session_state else None
embedding_service = st.session_state.embedding_service if 'embedding_service' in st.session_state else None

# Initialize document processor
doc_processor = DocumentProcessor(
    chunk_size=config.get("CHUNK_SIZE", 1000),
    chunk_overlap=config.get("CHUNK_OVERLAP", 200)
)

# Initialize text analyzer
text_analyzer = TextAnalyzer()

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Document Upload")
    
    # Upload document for analysis
    uploaded_file = st.file_uploader(
        "Upload Research Document",
        type=["pdf", "docx", "txt"],
        key="doc_uploader_quality"
    )
    
    if uploaded_file:
        if st.button("Process Document for Analysis"):
            with st.spinner("Processing document..."):
                # Process the document
                doc_result = doc_processor.process_uploaded_file(uploaded_file)
                
                if "error" in doc_result:
                    st.error(doc_result["error"])
                else:
                    st.session_state.doc_for_analysis = doc_result
                    st.success(f"Processed {doc_result['filename']} ({len(doc_result['full_text'])} characters, {doc_result['chunk_count']} chunks)")
                    
                    # Extract and show abstract if available
                    abstract = doc_processor.extract_abstract(doc_result["full_text"])
                    if abstract:
                        st.markdown("### Abstract")
                        st.markdown(abstract)

# Main content area
if 'doc_for_analysis' not in st.session_state or not st.session_state.doc_for_analysis:
    st.info("Please upload and process a document for quality assessment.")
else:
    doc = st.session_state.doc_for_analysis
    
    # Document info
    st.markdown(f"## Document: {doc['filename']}")
    
    # Start analysis button
    if st.button("Start Quality Assessment") and not st.session_state.in_progress:
        st.session_state.in_progress = True
        
        with st.spinner("Analyzing research quality..."):
            try:
                # Function to run async code in a blocking way for Streamlit
                def run_async(coro):
                    return asyncio.run(coro)
                
                # Get quality analysis from LLM
                preview_text = doc_processor.get_preview_text(doc["full_text"], max_chars=10000)
                quality_analysis = run_async(llm_service.analyze_research_quality(preview_text))
                
                # Perform additional text analysis
                text_analysis_results = text_analyzer.analyze_text(doc["full_text"])
                citation_analysis = text_analyzer.analyze_citations(doc["full_text"])
                key_sentences = text_analyzer.extract_key_sentences(doc["full_text"], n=5)
                
                # Extract additional document sections
                sections = doc_processor.extract_sections(doc["full_text"])
                references = doc_processor.extract_references(doc["full_text"])
                
                # Store results in session state
                st.session_state.quality_results = {
                    "quality_analysis": quality_analysis,
                    "text_analysis": text_analysis_results,
                    "citation_analysis": citation_analysis,
                    "key_sentences": key_sentences,
                    "document_sections": sections,
                    "references": references,
                    "document": doc,
                    "timestamp": time.time()
                }
                
                st.success("Analysis complete!")
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            finally:
                st.session_state.in_progress = False
    
    # Display quality assessment results if available
    if 'quality_results' in st.session_state and st.session_state.quality_results:
        results = st.session_state.quality_results
        
        if "quality_analysis" in results:
            analysis = results["quality_analysis"]
            
            # Create dashboard layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.header("Quality Assessment Overview")
                
                # Create radar chart of scores
                categories = [
                    'Overall Quality', 
                    'Argument Strength', 
                    'Reference Quality', 
                    'Methodology', 
                    'Structure'
                ]
                
                scores = [
                    analysis.get('overall_quality_score', 0),
                    analysis.get('argument_strength', {}).get('score', 0),
                    analysis.get('reference_quality', {}).get('score', 0),
                    analysis.get('methodology_assessment', {}).get('score', 0),
                    analysis.get('structural_consistency', {}).get('score', 0)
                ]
                
                # Add the first score again to close the loop
                categories.append(categories[0])
                scores.append(scores[0])
                
                # Create radar chart with plotly
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    name='Research Quality',
                    line_color='rgba(75, 192, 192, 0.8)',
                    fillcolor='rgba(75, 192, 192, 0.2)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary
                if "summary" in analysis:
                    st.subheader("Assessment Summary")
                    st.markdown(analysis["summary"])
            
            with col2:
                st.header("Quality Metrics")
                
                # Display scores as metrics
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.metric("Overall Quality", f"{analysis.get('overall_quality_score', 0)}/10")
                    st.metric("Argument Strength", f"{analysis.get('argument_strength', {}).get('score', 0)}/10")
                    st.metric("Reference Quality", f"{analysis.get('reference_quality', {}).get('score', 0)}/10")
                
                with col2b:
                    st.metric("Methodology", f"{analysis.get('methodology_assessment', {}).get('score', 0)}/10")
                    st.metric("Structure", f"{analysis.get('structural_consistency', {}).get('score', 0)}/10")
            
            # Detailed analysis sections
            st.header("Detailed Analysis")
            
            # Create tabs for each analysis category
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Argument Strength", 
                "References", 
                "Methodology", 
                "Structure",
                "Text Analytics"
            ])
            
            with tab1:
                if "argument_strength" in analysis and "issues" in analysis["argument_strength"]:
                    issues = analysis["argument_strength"]["issues"]
                    
                    if issues:
                        for issue in issues:
                            severity = issue.get("severity", "medium")
                            
                            # Set color based on severity
                            if severity == "high":
                                color = "red"
                            elif severity == "medium":
                                color = "orange"
                            else:
                                color = "gray"
                            
                            st.markdown(f"<div style='margin-bottom:10px; padding:10px; border-left:4px solid {color};'>"
                                      f"<span style='color:{color};font-weight:bold;'>{severity.upper()}</span>: "
                                      f"{issue.get('description', '')}"
                                      f"</div>", unsafe_allow_html=True)
                    else:
                        st.success("No significant issues found with argument strength.")
            
            with tab2:
                if "reference_quality" in analysis and "issues" in analysis["reference_quality"]:
                    issues = analysis["reference_quality"]["issues"]
                    
                    if issues:
                        for issue in issues:
                            severity = issue.get("severity", "medium")
                            
                            # Set color based on severity
                            if severity == "high":
                                color = "red"
                            elif severity == "medium":
                                color = "orange"
                            else:
                                color = "gray"
                            
                            st.markdown(f"<div style='margin-bottom:10px; padding:10px; border-left:4px solid {color};'>"
                                      f"<span style='color:{color};font-weight:bold;'>{severity.upper()}</span>: "
                                      f"{issue.get('description', '')}"
                                      f"</div>", unsafe_allow_html=True)
                    else:
                        st.success("No significant issues found with references.")
                
                # Display extracted references
                if "references" in results and results["references"]:
                    with st.expander("Extracted References"):
                        for i, ref in enumerate(results["references"]):
                            st.markdown(f"{i+1}. {ref}")
                
                # Display citation analysis
                if "citation_analysis" in results:
                    with st.expander("Citation Analysis"):
                        citation = results["citation_analysis"]
                        st.metric("Total Citations", citation.get("total_citations", 0))
                        st.metric("Citation Style", citation.get("dominant_style", "Unknown").upper())
                        
                        if "avg_citation_age" in citation:
                            st.metric("Average Citation Age", f"{citation['avg_citation_age']:.1f} years")
                        
                        if "recent_citations_percentage" in citation:
                            st.metric("Recent Citations (≤5 years)", f"{citation['recent_citations_percentage']*100:.1f}%")
            
            with tab3:
                if "methodology_assessment" in analysis and "issues" in analysis["methodology_assessment"]:
                    issues = analysis["methodology_assessment"]["issues"]
                    
                    if issues:
                        for issue in issues:
                            severity = issue.get("severity", "medium")
                            
                            # Set color based on severity
                            if severity == "high":
                                color = "red"
                            elif severity == "medium":
                                color = "orange"
                            else:
                                color = "gray"
                            
                            st.markdown(f"<div style='margin-bottom:10px; padding:10px; border-left:4px solid {color};'>"
                                      f"<span style='color:{color};font-weight:bold;'>{severity.upper()}</span>: "
                                      f"{issue.get('description', '')}"
                                      f"</div>", unsafe_allow_html=True)
                    else:
                        st.success("No significant issues found with methodology.")
            
            with tab4:
                if "structural_consistency" in analysis and "issues" in analysis["structural_consistency"]:
                    issues = analysis["structural_consistency"]["issues"]
                    
                    if issues:
                        for issue in issues:
                            severity = issue.get("severity", "medium")
                            
                            # Set color based on severity
                            if severity == "high":
                                color = "red"
                            elif severity == "medium":
                                color = "orange"
                            else:
                                color = "gray"
                            
                            st.markdown(f"<div style='margin-bottom:10px; padding:10px; border-left:4px solid {color};'>"
                                      f"<span style='color:{color};font-weight:bold;'>{severity.upper()}</span>: "
                                      f"{issue.get('description', '')}"
                                      f"</div>", unsafe_allow_html=True)
                    else:
                        st.success("No significant issues found with structural consistency.")
            
            with tab5:
                if "text_analysis" in results:
                    text_analysis = results["text_analysis"]
                    
                    # Basic metrics
                    st.subheader("Document Stats")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Word Count", text_analysis["basic_metrics"]["word_count"])
                        st.metric("Vocabulary Size", text_analysis["vocabulary_metrics"]["vocabulary_size"])
                    
                    with metrics_col2:
                        st.metric("Sentence Count", text_analysis["basic_metrics"]["sentence_count"])
                        st.metric("Lexical Diversity", f"{text_analysis['vocabulary_metrics']['lexical_diversity']:.2f}")
                    
                    with metrics_col3:
                        st.metric("Avg. Sentence Length", f"{text_analysis['basic_metrics']['avg_sentence_length']:.1f} words")
                        st.metric("Content Word Density", f"{text_analysis['vocabulary_metrics']['content_word_density']:.2f}")
                    
                    # Readability
                    st.subheader("Readability")
                    read_col1, read_col2 = st.columns(2)
                    
                    with read_col1:
                        flesch = text_analysis["readability"]["flesch_reading_ease"]
                        st.metric("Flesch Reading Ease", f"{flesch:.1f}/100")
                        
                        # Interpret Flesch score
                        if flesch > 80:
                            st.markdown("📚 **Easy to read** - 6th grade level")
                        elif flesch > 60:
                            st.markdown("📚 **Standard/Plain English** - 8-9th grade level")
                        elif flesch > 50:
                            st.markdown("📚 **Fairly difficult** - 10-12th grade level")
                        else:
                            st.markdown("📚 **Difficult** - College level")
                    
                    with read_col2:
                        fog = text_analysis["readability"]["gunning_fog_index"]
                        st.metric("Gunning Fog Index", f"{fog:.1f}")
                        
                        # Interpret Fog index
                        if fog < 8:
                            st.markdown("🎯 **Very readable** - Middle school level")
                        elif fog < 12:
                            st.markdown("🎯 **Readable** - High school level")
                        elif fog < 17:
                            st.markdown("🎯 **Challenging** - College level")
                        else:
                            st.markdown("🎯 **Difficult** - Graduate level")
                    
                    # Keywords
                    st.subheader("Top Keywords")
                    
                    keywords = text_analysis["keywords"]
                    if keywords:
                        # Create a dataframe for keywords
                        keywords_df = pd.DataFrame(keywords)
                        
                        # Show bar chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[kw["word"] for kw in keywords[:10]],
                                y=[kw["count"] for kw in keywords[:10]],
                                marker_color='lightblue'
                            )
                        ])
                        
                        fig.update_layout(
                            title="Top 10 Keywords by Frequency",
                            xaxis_title="Keywords",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Key sentences
                    if "key_sentences" in results:
                        st.subheader("Key Sentences")
                        
                        for i, sentence in enumerate(results["key_sentences"]):
                            st.markdown(f"**{i+1}.** {sentence}")
            
            # Download results button
            results_json = json.dumps(st.session_state.quality_results, default=str, indent=2)
            
            st.download_button(
                label="Download Quality Assessment Results",
                data=results_json,
                file_name=f"quality_assessment_{doc['document_id']}.json",
                mime="application/json"
            )