import os 
from pathlib import Path 


project_name="AI research assistant"

list_of_files={

    f"{project_name}/__init__.py",
    f"{project_name}/.env",
    f"{project_name}/.gitignore",
    f"{project_name}/.README.md",
    f"{project_name}/.requirements.txt",
    f"{project_name}/pages/similarity_detection.py",
    f"{project_name}/pages/quality_assesment.py",
    f"{project_name}/pages/summarization.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/config.py",
    f"{project_name}/utils/llm_service.py",
    f"{project_name}/utils/embedding_service.py",
    f"{project_name}/utils/vector_db.py",
    f"{project_name}/utils/document_processor.py",
    f"{project_name}/utils/text_analysis.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/document_uploader.py",
    f"{project_name}/components/similarity_viewer.py",

}

for path in list_of_files:
    filepath=Path(path)

    filedir,filename=os.path.split(path)
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass