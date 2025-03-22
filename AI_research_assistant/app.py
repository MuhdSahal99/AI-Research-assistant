import streamlit as st
import os
import json
import asyncio
import time
import pandas as pd
from utils.service_manager import initialize_services, run_async_task

# Set page config
st.set_page_config(
    page_title="Research LLM Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def load_css():
    css = """
    .main-title {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
    
if 'reference_docs' not in st.session_state:
    st.session_state.reference_docs = []
    
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Initialize services
if 'service_manager' not in st.session_state:
    with st.spinner("Initializing services..."):
        st.session_state.service_manager = initialize_services()
        st.session_state.initialized = True

# Page header
st.markdown("<h1 class='main-title'>Research LLM Platform</h1>", unsafe_allow_html=True)
st.markdown("AI-powered research analysis for similarity detection, quality assessment, summarization, and compliance.")

# Sidebar for document upload and system status
with st.sidebar:
    st.header("System Status")
    
    if st.session_state.initialized:
        services_status = st.session_state.service_manager.get_services_status()
        
        # Display service status
        if services_status["llm"]:
            st.success("✅ LLM Service: Connected (Groq)")
        else:
            st.error("❌ LLM Service: Not configured")
            
        if services_status["embedding"]:
            st.success("✅ Embedding Service: Connected (Mistral)")
        else:
            st.error("❌ Embedding Service: Not configured")
            
        if services_status["vector_db"]:
            st.success("✅ Vector Database: Connected (Pinecone)")
        else:
            st.error("❌ Vector Database: Not configured")
    
    st.header("Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Research Document",
        type=["pdf", "docx", "txt"],
        key="main_uploader"
    )
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            st.session_state.processing = True
            
            # Process the document
            doc_result = st.session_state.service_manager.process_document(uploaded_file)
            
            if "error" in doc_result:
                st.error(doc_result["error"])
            else:
                st.session_state.current_doc = doc_result
                
                # Generate and store embedding
                with st.spinner("Generating embedding..."):
                    embedding_success = run_async_task(
                        st.session_state.service_manager.store_document_embedding(doc_result)
                    )
                    
                    if embedding_success:
                        st.success("Document processed and embedding stored!")
                    else:
                        st.warning("Document processed but embedding storage failed.")
            
            st.session_state.processing = False

    # Reference document upload
    st.header("Reference Documents")
    
    ref_file = st.file_uploader(
        "Upload Reference Document",
        type=["pdf", "docx", "txt"],
        key="ref_uploader",
        accept_multiple_files=True  
    )
    
    if ref_file and st.button("Process Reference Document"):
        with st.spinner("Processing reference documents..."):
            st.session_state.processing = True
            
            reference_docs = []
            for file in ref_file:
                # Process each reference document
                ref_result = st.session_state.service_manager.process_document(file)
                
                if "error" in ref_result:
                    st.error(f"Error processing {file.name}: {ref_result['error']}")
                else:
                    reference_docs.append(ref_result)
                    
                    # Generate and store embedding
                    embedding_success = run_async_task(
                        st.session_state.service_manager.store_document_embedding(ref_result)
                    )
                    
                    if embedding_success:
                        st.success(f"Processed {file.name}")
                    else:
                        st.warning(f"Processed {file.name} but embedding storage failed.")
            
            st.session_state.reference_docs = reference_docs
            st.session_state.processing = False

# Main content area - create tabs for different features
if st.session_state.current_doc:
    doc = st.session_state.current_doc
    
    # Document info
    st.markdown("<h2 class='sub-title'>Current Document</h2>", unsafe_allow_html=True)
    st.markdown(f"**Filename:** {doc['filename']}")
    st.markdown(f"**Size:** {doc['text_length']} characters")
    
    if "metadata" in doc and doc["metadata"]:
        with st.expander("Document Metadata"):
            for key, value in doc["metadata"].items():
                st.markdown(f"**{key}:** {value}")
    
    # Create tabs for different analysis features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Quality Assessment", 
        "Summarization", 
        "Similarity Detection",
        "Compliance Check"
    ])
    
    # Tab 1: Quality Assessment
    with tab1:
        st.header("Research Quality Assessment")
        
        if st.button("Analyze Research Quality"):
            with st.spinner("Analyzing research quality..."):
                # Get quality analysis
                quality_analysis = run_async_task(
                    st.session_state.service_manager.analyze_research_quality(doc)
                )
                
                if "error" in quality_analysis:
                    st.error(quality_analysis["error"])
                else:
                    # Display quality scores
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Overall Quality", f"{quality_analysis.get('overall_quality_score', 0)}/10")
                    
                    with col2:
                        st.metric("Argument Strength", f"{quality_analysis.get('argument_strength', {}).get('score', 0)}/10")
                    
                    with col3:
                        st.metric("Reference Quality", f"{quality_analysis.get('reference_quality', {}).get('score', 0)}/10")
                    
                    with col4:
                        st.metric("Methodology", f"{quality_analysis.get('methodology_assessment', {}).get('score', 0)}/10")
                    
                    # Display summary
                    if "summary" in quality_analysis:
                        st.subheader("Assessment Summary")
                        st.markdown(quality_analysis["summary"])
                    
                    # Display detailed issues
                    st.subheader("Detailed Analysis")
                    
                    # Create subtabs for different categories
                    subtab1, subtab2, subtab3, subtab4 = st.tabs([
                        "Argument Strength", 
                        "References", 
                        "Methodology", 
                        "Structure"
                    ])
                    
                    with subtab1:
                        if "argument_strength" in quality_analysis and "issues" in quality_analysis["argument_strength"]:
                            issues = quality_analysis["argument_strength"]["issues"]
                            
                            if issues:
                                for issue in issues:
                                    st.markdown(f"**{issue.get('severity', 'medium').upper()}:** {issue.get('description', '')}")
                            else:
                                st.success("No significant issues found.")
                    
                    with subtab2:
                        if "reference_quality" in quality_analysis and "issues" in quality_analysis["reference_quality"]:
                            issues = quality_analysis["reference_quality"]["issues"]
                            
                            if issues:
                                for issue in issues:
                                    st.markdown(f"**{issue.get('severity', 'medium').upper()}:** {issue.get('description', '')}")
                            else:
                                st.success("No significant issues found.")
                    
                    with subtab3:
                        if "methodology_assessment" in quality_analysis and "issues" in quality_analysis["methodology_assessment"]:
                            issues = quality_analysis["methodology_assessment"]["issues"]
                            
                            if issues:
                                for issue in issues:
                                    st.markdown(f"**{issue.get('severity', 'medium').upper()}:** {issue.get('description', '')}")
                            else:
                                st.success("No significant issues found.")
                    
                    with subtab4:
                        if "structural_consistency" in quality_analysis and "issues" in quality_analysis["structural_consistency"]:
                            issues = quality_analysis["structural_consistency"]["issues"]
                            
                            if issues:
                                for issue in issues:
                                    st.markdown(f"**{issue.get('severity', 'medium').upper()}:** {issue.get('description', '')}")
                            else:
                                st.success("No significant issues found.")
                    
                    # Download results button
                    st.download_button(
                        label="Download Quality Assessment Results",
                        data=json.dumps(quality_analysis, indent=2),
                        file_name=f"quality_assessment_{doc['document_id']}.json",
                        mime="application/json"
                    )
    
    # Tab 2: Summarization
    with tab2:
        st.header("Research Summarization")
        
        # Summarization options
        summary_type = st.radio(
            "Summary Type",
            options=["General", "Reviewer-Focused", "Editor-Focused"],
            horizontal=True
        )
        
        if st.button("Generate Summary"):
            with st.spinner("Generating research summary..."):
                # Convert to API parameter format
                summary_type_param = summary_type.lower().replace("-", "_").replace(" ", "_")
                
                # Get summary
                summary_result = run_async_task(
                    st.session_state.service_manager.generate_summary(doc, summary_type_param)
                )
                
                if "error" in summary_result:
                    st.error(summary_result["error"])
                else:
                    # Display title and abstract
                    if "title" in summary_result:
                        st.subheader(summary_result["title"])
                    
                    if "abstract" in summary_result:
                        st.markdown(f"**Abstract:** {summary_result['abstract']}")
                    
                    # Create sections
                    st.subheader("Key Points")
                    if "key_points" in summary_result:
                        for i, point in enumerate(summary_result["key_points"]):
                            st.markdown(f"{i+1}. {point}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Methodology")
                        if "methodology_summary" in summary_result:
                            st.markdown(summary_result["methodology_summary"])
                    
                    with col2:
                        st.subheader("Findings")
                        if "findings_summary" in summary_result:
                            st.markdown(summary_result["findings_summary"])
                    
                    st.subheader("Limitations & Future Work")
                    if "limitations" in summary_result:
                        st.markdown("**Limitations:**")
                        for limitation in summary_result["limitations"]:
                            st.markdown(f"- {limitation}")
                    
                    if "future_work" in summary_result:
                        st.markdown("**Future Work:**")
                        st.markdown(summary_result["future_work"])
                    
                    if "significance" in summary_result:
                        st.markdown("**Significance:**")
                        st.markdown(summary_result["significance"])
                    
                    # Download results button
                    st.download_button(
                        label="Download Summary",
                        data=json.dumps(summary_result, indent=2),
                        file_name=f"summary_{doc['document_id']}.json",
                        mime="application/json"
                    )
    
    # Tab 3: Similarity Detection
    with tab3:
        st.header("Similarity & Novelty Detection")
        
        if not st.session_state.reference_docs:
            st.info("Please upload reference documents to compare against.")
        else:
            st.markdown(f"Comparing **{doc['filename']}** against {len(st.session_state.reference_docs)} reference documents.")
            
            similarity_threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.5, 
                max_value=1.0, 
                value=0.7,
                step=0.05,
                help="Minimum similarity score to highlight"
            )
            
            if st.button("Analyze Similarity"):
                with st.spinner("Analyzing document similarity..."):
                    # Get similarity analysis
                    similarity_result = run_async_task(
                        st.session_state.service_manager.compare_documents(
                            doc, 
                            st.session_state.reference_docs
                        )
                    )
                    
                    if "error" in similarity_result:
                        st.error(similarity_result["error"])
                    else:
                        # Display novelty score
                        if "novelty_analysis" in similarity_result and "novelty_score" in similarity_result["novelty_analysis"]:
                            novelty_score = similarity_result["novelty_analysis"]["novelty_score"]
                            
                            # Determine color based on score
                            if novelty_score >= 7:
                                color = "green"
                                verdict = "Highly Novel"
                            elif novelty_score >= 4:
                                color = "orange"
                                verdict = "Moderately Novel"
                            else:
                                color = "red"
                                verdict = "Low Novelty"
                                
                            st.markdown(f"<h3>Novelty Score: <span style='color:{color};'>{novelty_score}/10</span></h3>", unsafe_allow_html=True)
                            st.markdown(f"<h4 style='color:{color};'>{verdict}</h4>", unsafe_allow_html=True)
                        
                        # Display similarity scores
                        if "similarity_scores" in similarity_result and similarity_result["similarity_scores"]:
                            st.subheader("Similar Documents")
                            
                            # Create table of similarity scores
                            similarity_data = []
                            for result in similarity_result["similarity_scores"]:
                                if result["similarity_score"] >= similarity_threshold:
                                    similarity_data.append({
                                        "Document": result["filename"],
                                        "Similarity Score": f"{result['similarity_score']:.4f}"
                                    })
                            
                            if similarity_data:
                                st.table(pd.DataFrame(similarity_data))
                            else:
                                st.success("No documents above similarity threshold found.")
                        
                        # Show novel contributions
                        if ("novelty_analysis" in similarity_result and 
                            "novel_contributions" in similarity_result["novelty_analysis"]):
                            
                            st.subheader("Novel Contributions")
                            contributions = similarity_result["novelty_analysis"]["novel_contributions"]
                            
                            for contrib in contributions:
                                significance = contrib.get("significance", "medium")
                                
                                # Set color based on significance
                                if significance == "high":
                                    sig_color = "green"
                                elif significance == "medium":
                                    sig_color = "orange"
                                else:
                                    sig_color = "gray"
                                
                                st.markdown(f"<span style='color:{sig_color};font-weight:bold;'>{significance.upper()}</span>: {contrib.get('description', '')}", unsafe_allow_html=True)
                        
                        # Show overlapping sections
                        if ("novelty_analysis" in similarity_result and 
                            "overlapping_sections" in similarity_result["novelty_analysis"]):
                            
                            st.subheader("Overlapping Content")
                            overlaps = similarity_result["novelty_analysis"]["overlapping_sections"]
                            
                            for overlap in overlaps:
                                severity = overlap.get("severity", "medium")
                                
                                # Set color based on severity
                                if severity == "high":
                                    sev_color = "red"
                                elif severity == "medium":
                                    sev_color = "orange"
                                else:
                                    sev_color = "gray"
                                
                                with st.expander(f"{severity.upper()}: {overlap.get('similarity_description', '')}"):
                                    if "content" in overlap:
                                        st.code(overlap["content"])
                        
                        # Display recommendation
                        if "novelty_analysis" in similarity_result and "recommendation" in similarity_result["novelty_analysis"]:
                            st.subheader("Recommendation")
                            st.markdown(similarity_result["novelty_analysis"]["recommendation"])
                        
                        # Download results button
                        st.download_button(
                            label="Download Similarity Analysis",
                            data=json.dumps(similarity_result, indent=2),
                            file_name=f"similarity_analysis_{doc['document_id']}.json",
                            mime="application/json"
                        )
    
    # Tab 4: Compliance Check
    with tab4:
        st.header("Research Compliance Check")
        
        # Compliance guidelines selection
        guideline_type = st.radio(
            "Guideline Type",
            options=["Common Guidelines", "Custom Guidelines"],
            horizontal=True
        )
        
        if guideline_type == "Common Guidelines":
            guideline_options = {
                "APA": "APA Style (American Psychological Association)",
                "MLA": "MLA Style (Modern Language Association)",
                "IEEE": "IEEE Style (Institute of Electrical and Electronics Engineers)",
                "Chicago": "Chicago Style"
            }
            
            selected_guideline = st.selectbox(
                "Select Guideline",
                options=list(guideline_options.keys()),
                format_func=lambda x: guideline_options[x]
            )
            
            # Load guideline text based on selection
            guidelines = {
                "APA": """
                APA Style Guidelines:
                1. Double-spaced text on standard-sized paper (8.5" x 11") with 1" margins
                2. Times New Roman 12-point font
                3. Page header with title and page number
                4. Title page with title, author, institution
                5. Citations: (Author, Year) or (Author, Year, p. #) for direct quotes
                6. References alphabetical by author's last name
                7. References format: Author, A. A. (Year). Title. Source. DOI
                """,
                
                "MLA": """
                MLA Style Guidelines:
                1. Double-spaced text on standard-sized paper with 1" margins
                2. Times New Roman 12-point font
                3. Header with last name and page number
                4. First page: name, instructor, course, date
                5. Citations: (Author Page)
                6. Works Cited alphabetical by author's last name
                7. Works Cited format: Author. "Title." Container, Other contributors, Version, Number, Publisher, Date, Location.
                """,
                
                "IEEE": """
                IEEE Style Guidelines:
                1. Double-column format on US letter-sized paper
                2. Times New Roman 10-point font
                3. First page: title, author names, affiliations, abstract
                4. Citations: numbered references in square brackets [1]
                5. References numbered in order of appearance
                6. References format: [1] A. Author, "Title," Journal, vol. x, no. x, pp. xxx-xxx, Month Year.
                7. Section headings numbered (I. INTRODUCTION)
                """,
                
                "Chicago": """
                Chicago Style Guidelines:
                1. Double-spaced text with 1" margins
                2. Times New Roman 12-point font
                3. Page numbers in header
                4. Notes-Bibliography system: footnotes/endnotes for citations
                5. Note format: Firstname Lastname, Title (Place: Publisher, Year), page.
                6. Bibliography: Lastname, Firstname. Title. Place: Publisher, Year.
                7. Block quotes (>100 words) indented 0.5"
                """
            }
            
            guidelines_text = guidelines[selected_guideline]
            
            with st.expander("View Guidelines"):
                st.markdown(guidelines_text)
        else:
            # Custom guidelines input
            guidelines_text = st.text_area(
                "Enter Custom Guidelines",
                height=200
            )
        
        if st.button("Check Compliance"):
            if not guidelines_text.strip():
                st.warning("Please select or enter compliance guidelines.")
            else:
                with st.spinner("Checking compliance..."):
                    # Get compliance results
                    compliance_result = run_async_task(
                        st.session_state.service_manager.check_compliance(doc, guidelines_text)
                    )
                    
                    if "error" in compliance_result:
                        st.error(compliance_result["error"])
                    else:
                        # Display compliance score
                        if "overall_compliance_score" in compliance_result:
                            score = compliance_result["overall_compliance_score"]
                            
                            # Determine color based on score
                            if score >= 8:
                                color = "green"
                                verdict = "Highly Compliant"
                            elif score >= 6:
                                color = "orange"
                                verdict = "Moderately Compliant"
                            else:
                                color = "red"
                                verdict = "Low Compliance"
                            
                            st.markdown(f"<h3>Overall Compliance Score: <span style='color:{color};'>{score}/10</span></h3>", unsafe_allow_html=True)
                            st.markdown(f"<h4 style='color:{color};'>{verdict}</h4>", unsafe_allow_html=True)
                        
                        # Display summary
                        if "summary" in compliance_result:
                            st.subheader("Summary")
                            st.markdown(compliance_result["summary"])
                        
                        # Display compliance metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if "citation_style_compliance" in compliance_result and "score" in compliance_result["citation_style_compliance"]:
                                st.metric("Citation Style", f"{compliance_result['citation_style_compliance']['score']}/10")
                        
                        with col2:
                            if "formatting_compliance" in compliance_result and "score" in compliance_result["formatting_compliance"]:
                                st.metric("Formatting", f"{compliance_result['formatting_compliance']['score']}/10")
                        
                        with col3:
                            if "structural_compliance" in compliance_result and "score" in compliance_result["structural_compliance"]:
                                st.metric("Structure", f"{compliance_result['structural_compliance']['score']}/10")
                        
                        # Display detailed issues
                        st.subheader("Detailed Compliance Issues")
                        
                        if "compliance_issues" in compliance_result:
                            issues = compliance_result["compliance_issues"]
                            
                            if issues:
                                for i, issue in enumerate(issues):
                                    section = issue.get("section", "General")
                                    severity = issue.get("severity", "medium")
                                    
                                    # Set color based on severity
                                    if severity == "high":
                                        color = "red"
                                    elif severity == "medium":
                                        color = "orange"
                                    else:
                                        color = "gray"
                                    
                                    with st.expander(f"{section}: {issue.get('issue', '')}"):
                                        st.markdown(f"**Severity:** <span style='color:{color};'>{severity.upper()}</span>", unsafe_allow_html=True)
                                        if "recommendation" in issue:
                                            st.markdown(f"**Recommendation:** {issue['recommendation']}")
                            else:
                                st.success("No compliance issues found.")
                        
                        # Download results button
                        st.download_button(
                            label="Download Compliance Report",
                            data=json.dumps(compliance_result, indent=2),
                            file_name=f"compliance_report_{doc['document_id']}.json",
                            mime="application/json"
                        )
else:
    # No document uploaded yet
    st.info("👈 Please upload a research document to get started.")
    
    # Show platform features
    st.markdown("### Key Features:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quality Assessment")
        st.markdown("Evaluate argument strength, logical coherence, and research validity.")
        
        st.markdown("#### Summarization")
        st.markdown("Generate concise summaries and reviewer briefs to streamline the review process.")
    
    with col2:
        st.markdown("#### Similarity Detection")
        st.markdown("Identify overlapping content, paraphrased text, and novel contributions.")
        
        st.markdown("#### Compliance Checking")
        st.markdown("Ensure papers meet formatting guidelines, citation standards, and other requirements.")

# Footer
