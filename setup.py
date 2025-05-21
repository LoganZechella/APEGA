#!/usr/bin/env python
"""
Setup script for APEGA.
"""

from setuptools import setup, find_packages

setup(
    name="apega",
    version="0.1.0",
    description="Automated Practice Exam Generation Agent",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "PyMuPDF>=1.23.0",
        "pytesseract>=0.3.10",
        "Pillow>=10.0.0",
        "qdrant-client>=1.6.0",
        "openai>=1.0.0",
        "google-generativeai>=0.3.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langgraph>=0.0.20",
        "langchain-core>=0.1.0",
        "nltk>=3.8.1",
        "sentence-transformers>=2.2.2",
        "jinja2>=3.1.2",
        "weasyprint>=60.0",
        "markdown>=3.5",
        "rank-bm25>=0.2.2",
        "tqdm>=4.66.0",
        "tenacity>=8.2.3",
        "loguru>=0.7.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "apega-ingest=ingest:main",
            "apega-generate=generate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)