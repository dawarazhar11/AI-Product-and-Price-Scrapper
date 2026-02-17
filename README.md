# AI Product and Price Scrapper

An AI-powered web scraping and PDF processing toolkit built with Streamlit. Extracts product names, prices, and service information from websites and PDF documents using LLM-based analysis, browser automation, and vector similarity search.

## Features

### Web Scraper (`app.py`)
- **Multi-engine browser automation** - Selenium, Playwright, and undetected Chrome for bypassing anti-bot protections (Cloudflare, etc.)
- **LLM-powered extraction** - Uses Ollama models to intelligently extract product names, prices, descriptions, and categories from raw page content
- **Vector similarity search** - ChromaDB-based embeddings for semantic chunk tagging and deduplication
- **Fuzzy deduplication** - Combines RapidFuzz, FuzzyWuzzy, and textdistance for accurate duplicate detection
- **Specialized extractors** - Purpose-built logic for fuel prices, automotive services, and professional services
- **Multi-language support** - Automatic language detection and translation via `langdetect` and `deep-translator`
- **Proxy support** - Free proxy rotation and custom proxy configuration
- **Data persistence** - Save and reload scraped datasets locally

### PDF Processor (`pdf_app.py` / `pdf_handler.py`)
- **Multi-library extraction** - PyMuPDF, pdfplumber, and PyPDF2 for text, tables, images, and metadata
- **URL-based processing** - Fetch and process PDFs directly from URLs
- **Auto-discovery** - Automatically find PDF links on web pages
- **Menu extraction** - Specialized parsing for menu-style PDF documents

### Launcher Tools
- `simple_launcher.py` - Streamlit-based GUI to launch either tool
- `app_launcher.py` - Standalone launcher with automatic dependency checking

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) running locally or on a network host (for LLM features)
- Chrome/Chromium browser (for Selenium/undetected Chrome scraping)

## Installation

```bash
# Clone the repository
git clone https://github.com/dawarazhar11/AI-Product-and-Price-Scrapper.git
cd AI-Product-and-Price-Scrapper

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (if using Playwright scraper)
playwright install
```

## Configuration

Copy the example environment file and edit it with your settings:

```bash
cp .env.example .env
```

Key configuration options (set via `.env` or environment variables):

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server address | `http://localhost:11434` |
| `DEFAULT_TIMEOUT` | API request timeout (seconds) | `600` |
| `DEFAULT_SCRAPE_TIMEOUT` | Web scraping timeout (seconds) | `300` |
| `MAX_RETRIES` | Max retries for failed requests | `3` |

## Usage

### Web Scraper

```bash
streamlit run app.py
```

1. Enter a target URL
2. Configure scraping options (engine, proxy, pagination depth)
3. Select an Ollama model for extraction
4. Run the scraper and review extracted products/prices
5. Export results as CSV or save for later

### PDF Processor

```bash
streamlit run pdf_app.py
```

1. Upload a PDF or provide a URL
2. Choose extraction method (text, tables, images, or all)
3. Review and export extracted data

### Launcher

```bash
streamlit run simple_launcher.py
```

## Project Structure

```
app.py               # Main web scraper application
pdf_app.py           # PDF processor Streamlit UI
pdf_handler.py       # PDF extraction utilities
app_launcher.py      # Standalone launcher with dependency checks
simple_launcher.py   # Streamlit-based launcher
requirements.txt     # Python dependencies
.env.example         # Environment variable template
pyproject.toml       # Project metadata
```

## License

This project is provided as-is for educational and research purposes.
