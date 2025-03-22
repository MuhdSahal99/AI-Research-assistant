import streamlit as st
import os
import asyncio
import json
import time
import uuid
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.document_processor import DocumentProcessor
from utils.config import load_config
from components.document_uploader import render_document_uploader

# Set page config
st.set_page_config(
    page_title="Compliance Checks",
    page_icon="✓",
    layout="wide"
)

# Load config
config = load_config()

# Page title
st.title("Research Compliance Checks")
st.markdown("Ensure research papers meet formatting guidelines, citation standards, and other requirements.")

# Initialize session state
if 'compliance_results' not in st.session_state:
    st.session_state.compliance_results = None
if 'doc_for_compliance' not in st.session_state:
    st.session_state.doc_for_compliance = None
if 'in_progress' not in st.session_state:
    st.session_state.in_progress = False
if 'selected_guideline' not in st.session_state:
    st.session_state.selected_guideline = "APA"  # Default
if 'custom_guidelines' not in st.session_state:
    st.session_state.custom_guidelines = ""

# Get services from session state
llm_service = st.session_state.llm_service if 'llm_service' in st.session_state else None

# Initialize document processor
doc_processor = DocumentProcessor(
    chunk_size=config.get("CHUNK_SIZE", 1000),
    chunk_overlap=config.get("CHUNK_OVERLAP", 200)
)

# Common journal guidelines
COMMON_GUIDELINES = {
    "APA": """
APA Style Guidelines:
1. Paper should be double-spaced on standard-sized paper (8.5" x 11") with 1" margins on all sides.
2. Use a clear font (e.g., Times New Roman) with 12-point size.
3. Include a page header (title) at the top of each page. For title page, include title, author(s), institutional affiliation.
4. Citations format: 
   - In-text citations should include the author's last name and year of publication, e.g., (Smith, 2020).
   - For direct quotes, include page number: (Smith, 2020, p. 123).
   - References list should be alphabetical by first author's last name.
   - Journal article format: Author, A. A., & Author, B. B. (Year). Title of article. Title of Journal, volume(issue), page range. DOI
5. Tables and figures should be numbered consecutively and include descriptive captions.
6. Headings: Use 5 levels of heading with specific formatting for each level.
7. Use active voice and write in third person. Avoid biased language.
8. Use past tense for literature review and methodology, present tense for results and conclusions.
    """,
    
    "MLA": """
MLA Style Guidelines:
1. Paper should be typed on standard 8.5 x 11-inch paper, double-spaced, with 1-inch margins.
2. Use a legible font (e.g., Times New Roman) with 12-point size.
3. Header should include last name and page number in upper right corner of each page.
4. First page should include your name, instructor's name, course, and date in upper left corner.
5. Citations format:
   - In-text citations should include author's last name and page number in parentheses: (Smith 123).
   - Works Cited page should list sources alphabetically by author's last name.
   - Journal article format: Author(s). "Title of Article." Title of Journal, Volume, Issue, Year, pages.
6. Block quotes (quotes longer than 4 lines) should be indented 1 inch from the left margin.
7. Titles of longer works (books, journals) should be italicized; titles of shorter works (articles, chapters) should be in quotation marks.
    """,
    
    "IEEE": """
IEEE Style Guidelines:
1. Paper should be in a two-column format on US letter-sized (8.5 x 11 inch) paper.
2. Use Times New Roman or similar serif font with 10-point size.
3. First page should include title, author names, affiliations, and abstract.
4. Citations format:
   - In-text citations use numbered references in square brackets: [1].
   - References are numbered in the order they appear in the text.
   - Journal article format: [1] A. Author, B. Author, and C. Author, "Title of article," Title of Journal, vol. x, no. x, pp. xxx-xxx, Month year.
5. Tables and figures should be numbered consecutively with captions.
6. Section headings should be numbered (e.g., I. INTRODUCTION, II. METHODOLOGY).
7. Use 1-inch margins on all sides.
    """,
    
    "Chicago": """
Chicago Style Guidelines:
1. Paper should be double-spaced with 1-inch margins on all sides.
2. Use a readable font (e.g., Times New Roman) with 12-point size.
3. Page numbers should be placed in the upper right corner, starting from the first page of text.
4. Citations format (Notes and Bibliography system):
   - Use footnotes or endnotes for citations.
   - Note format for books: Firstname Lastname, Title of Book (Place of publication: Publisher, Year), page number.
   - Bibliography entry for books: Lastname, Firstname. Title of Book. Place of publication: Publisher, Year.
   - Journal article note: Firstname Lastname, "Title of Article," Title of Journal Volume, Issue (Year): page number.
5. Block quotes (quotes longer than 100 words) should be indented 0.5 inches from the left margin.
6. Titles of longer works (books, journals) should be italicized; titles of shorter works (articles, chapters) should be in quotation marks.
    """
}

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Document Upload")
    
    # Upload document for compliance check
    uploaded_file = st.file_uploader(
        "Upload Research Document",
        type=["pdf", "docx", "txt"],
        key="doc_uploader_compliance"
    )
    
    if uploaded_file:
        if st.button("Process Document for Compliance Check"):
            with st.spinner("Processing document..."):
                # Process the document
                doc_result = doc_processor.process_uploaded_file(uploaded_file)
                
                if "error" in doc_result:
                    st.error(doc_result["error"])
                else:
                    st.session_state.doc_for_compliance = doc_result
                    st.success(f"Processed {doc_result['filename']} ({len(doc_result['full_text'])} characters, {doc_result['chunk_count']} chunks)")
    
    # Compliance guidelines selection
    if 'doc_for_compliance' in st.session_state and st.session_state.doc_for_compliance:
        st.header("Compliance Guidelines")
        
        guideline_type = st.radio(
            "Guideline Type",
            options=["Common Guidelines", "Custom Guidelines"],
            index=0
        )
        
        if guideline_type == "Common Guidelines":
            selected_guideline = st.selectbox(
                "Select Guideline",
                options=list(COMMON_GUIDELINES.keys()),
                index=list(COMMON_GUIDELINES.keys()).index(st.session_state.selected_guideline)
            )
            st.session_state.selected_guideline = selected_guideline
            
            with st.expander("View Guidelines"):
                st.markdown(COMMON_GUIDELINES[selected_guideline])
        else:
            st.markdown("Enter custom compliance guidelines:")
            custom_guidelines = st.text_area(
                "Custom Guidelines",
                value=st.session_state.custom_guidelines,
                height=300
            )
            st.session_state.custom_guidelines = custom_guidelines

# Main content area
if 'doc_for_compliance' not in st.session_state or not st.session_state.doc_for_compliance:
    st.info("Please upload and process a document for compliance check.")
else:
    doc = st.session_state.doc_for_compliance
    
    # Document info
    st.markdown(f"## Document: {doc['filename']}")
    
    # Get selected guidelines
    if st.session_state.selected_guideline in COMMON_GUIDELINES and st.session_state.custom_guidelines == "":
        # Use selected common guideline
        guidelines = COMMON_GUIDELINES[st.session_state.selected_guideline]
        guideline_name = st.session_state.selected_guideline
    else:
        # Use custom guidelines
        guidelines = st.session_state.custom_guidelines
        guideline_name = "Custom"
    
    st.markdown(f"**Selected Guidelines:** {guideline_name}")
    
    # Start compliance check button
    if st.button("Run Compliance Check") and not st.session_state.in_progress:
        st.session_state.in_progress = True
        
        with st.spinner("Checking document compliance..."):
            try:
                # Function to run async code in a blocking way for Streamlit
                def run_async(coro):
                    return asyncio.run(coro)
                
                # Get compliance check from LLM
                compliance_check = run_async(llm_service.check_compliance(
                    doc["full_text"][:15000],  # Limit size for LLM
                    guidelines
                ))
                
                # Store results in session state
                st.session_state.compliance_results = {
                    "compliance_check": compliance_check,
                    "guideline_name": guideline_name,
                    "guidelines": guidelines,
                    "document": doc,
                    "timestamp": time.time()
                }
                
                st.success("Compliance check complete!")
                
            except Exception as e:
                st.error(f"Error during compliance check: {str(e)}")
            finally:
                st.session_state.in_progress = False
    
    # Display compliance check results if available
    if 'compliance_results' in st.session_state and st.session_state.compliance_results:
        results = st.session_state.compliance_results
        
        if "compliance_check" in results:
            check = results["compliance_check"]
            
            # Create layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("Compliance Report")
                
                # Overall compliance score
                if "overall_compliance_score" in check:
                    score = check["overall_compliance_score"]
                    
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
                
                # Summary
                if "summary" in check:
                    st.subheader("Summary")
                    st.markdown(check["summary"])
            
            with col2:
                # Compliance scores
                st.header("Compliance Metrics")
                
                if "citation_style_compliance" in check and "score" in check["citation_style_compliance"]:
                    st.metric("Citation Style", f"{check['citation_style_compliance']['score']}/10")
                
                if "formatting_compliance" in check and "score" in check["formatting_compliance"]:
                    st.metric("Formatting", f"{check['formatting_compliance']['score']}/10")
                
                if "structural_compliance" in check and "score" in check["structural_compliance"]:
                    st.metric("Structure", f"{check['structural_compliance']['score']}/10")
            
            # Detailed compliance issues
            st.header("Detailed Compliance Issues")
            
            # Create tabs for different compliance categories
            tab1, tab2, tab3 = st.tabs([
                "Citation Style", 
                "Formatting", 
                "Structure"
            ])
            
            with tab1:
                if "citation_style_compliance" in check and "issues" in check["citation_style_compliance"]:
                    issues = check["citation_style_compliance"]["issues"]
                    
                    if issues:
                        st.subheader("Citation Style Issues")
                        
                        for i, issue in enumerate(issues):
                            with st.expander(f"Issue {i+1}: {issue.get('description', '')}"):
                                st.markdown("**Example:**")
                                st.markdown(f"```\n{issue.get('example', '')}\n```")
                                
                                st.markdown("**Correction:**")
                                st.markdown(issue.get('correction', ''))
                    else:
                        st.success("No citation style issues found.")
            
            with tab2:
                if "formatting_compliance" in check and "issues" in check["formatting_compliance"]:
                    issues = check["formatting_compliance"]["issues"]
                    
                    if issues:
                        st.subheader("Formatting Issues")
                        
                        for i, issue in enumerate(issues):
                            with st.expander(f"Issue {i+1}: {issue.get('description', '')}"):
                                if "location" in issue:
                                    st.markdown(f"**Location:** {issue['location']}")
                                
                                if "correction" in issue:
                                    st.markdown("**Correction:**")
                                    st.markdown(issue['correction'])
                    else:
                        st.success("No formatting issues found.")
            
            with tab3:
                if "structural_compliance" in check and "issues" in check["structural_compliance"]:
                    issues = check["structural_compliance"]["issues"]
                    
                    if issues:
                        st.subheader("Structural Issues")
                        
                        for i, issue in enumerate(issues):
                            with st.expander(f"Issue {i+1}: {issue.get('description', '')}"):
                                if "recommendation" in issue:
                                    st.markdown("**Recommendation:**")
                                    st.markdown(issue['recommendation'])
                    else:
                        st.success("No structural issues found.")
            
            # All compliance issues in a single view
            with st.expander("View All Compliance Issues"):
                if "compliance_issues" in check:
                    issues = check["compliance_issues"]
                    
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
                            
                            st.markdown(f"<div style='margin-bottom:10px; padding:10px; border-left:4px solid {color};'>"
                                      f"<strong>{section}</strong>: <span style='color:{color};font-weight:bold;'>{severity.upper()}</span> - "
                                      f"{issue.get('issue', '')}"
                                      f"<br><small>Recommendation: {issue.get('recommendation', '')}</small>"
                                      f"</div>", unsafe_allow_html=True)
                    else:
                        st.success("No compliance issues found.")
            
            # Download results button
            st.header("Export Report")
            
            results_json = json.dumps(results, default=str, indent=2)
            
            st.download_button(
                label="Download Compliance Report",
                data=results_json,
                file_name=f"compliance_report_{doc['document_id']}.json",
                mime="application/json"
            )