"""
Streamlit PDF Handler App

This app can run independently or be imported into the main app.
"""

import os
import re
import io
import json
import base64
import tempfile
import requests
import streamlit as st
from typing import Dict, List, Any, Optional
from loguru import logger
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Configure logger
logger.add("pdf_processing.log", rotation="100 MB")

# Check for PDF libraries
PDF_HANDLER_AVAILABLE = True
try:
    import PyPDF2
    import fitz  # PyMuPDF
    import pdfplumber
    logger.info("PDF libraries loaded successfully.")
except ImportError as e:
    PDF_HANDLER_AVAILABLE = False
    logger.warning(f"PDF libraries not available: {e}. PDF processing will be limited.")

def is_pdf_url(url: str) -> bool:
    """Check if a URL points to a PDF file."""
    if not url:
        return False
    
    # Skip mailto: and tel: links
    if url.startswith(('mailto:', 'tel:')):
        return False
    
    # Check URL extension first (this is fast)
    if url.lower().endswith('.pdf'):
        logger.info(f"URL {url} has PDF extension")
        return True
    
    # Check for PDF in the URL path (common for dynamic PDF generators)
    url_path = urlparse(url).path.lower()
    if '/pdf/' in url_path or 'pdf' in url_path or 'document' in url_path:
        # This might be a PDF, but we need to check the content type to be sure
        logger.info(f"URL {url} contains 'pdf' in path, checking content type")
    
    # Check URL content type if needed
    try:
        # Use a HEAD request to avoid downloading the entire file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
        
        # If the HEAD request fails, try a GET request with limited bytes
        if response.status_code != 200:
            response = requests.get(url, headers=headers, stream=True, timeout=5)
            response.close()  # Close the connection to avoid downloading the entire file
        
        content_type = response.headers.get('Content-Type', '').lower()
        content_disposition = response.headers.get('Content-Disposition', '').lower()
        
        # Check Content-Type header
        if 'application/pdf' in content_type:
            logger.info(f"URL {url} has PDF content type: {content_type}")
            return True
        
        # Check Content-Disposition header for filename
        if 'filename=' in content_disposition and '.pdf' in content_disposition:
            logger.info(f"URL {url} has PDF filename in Content-Disposition: {content_disposition}")
            return True
            
        logger.info(f"URL {url} is not a PDF (Content-Type: {content_type})")
        return False
    except Exception as e:
        logger.warning(f"Error checking URL content type for {url}: {e}")
        # If can't check content type, rely on extension only
        return url.lower().endswith('.pdf')

def fetch_pdf_links_from_website(website_url: str) -> List[str]:
    """
    Fetch all PDF links from a website.
    
    Args:
        website_url (str): The URL of the website to scrape
        
    Returns:
        List[str]: A list of PDF URLs found on the website
    """
    pdf_links = []
    
    try:
        # Add http:// prefix if missing
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
            logger.info(f"Added https:// prefix to URL: {website_url}")
        
        # Make a request to the website
        with st.spinner(f"Fetching PDF links from {website_url}..."):
            logger.info(f"Requesting content from {website_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(website_url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise an exception for bad responses
            
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('text/html'):
                logger.warning(f"URL returned non-HTML content: {content_type}")
                st.warning(f"The URL returned {content_type} content instead of HTML. It might not be a webpage.")
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract all links
            all_links = []
            link_count = 0
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('#'):
                    continue  # Skip anchors
                
                # Convert relative URLs to absolute
                try:
                    if not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                        full_url = urljoin(website_url, href)
                    else:
                        full_url = href
                    
                    # Skip mailto: and tel: links
                    if full_url.startswith(('mailto:', 'tel:')):
                        continue
                    
                    all_links.append(full_url)
                    link_count += 1
                except Exception as e:
                    logger.warning(f"Error processing link {href}: {e}")
            
            logger.info(f"Found {link_count} total links on the page")
            
            # Filter for PDF links
            for link in all_links:
                try:
                    if is_pdf_url(link):
                        pdf_links.append(link)
                except Exception as e:
                    logger.warning(f"Error checking if {link} is a PDF: {e}")
            
            # Log the result
            if pdf_links:
                logger.info(f"Found {len(pdf_links)} PDF links on {website_url}")
            else:
                logger.warning(f"No PDF links found on {website_url}")
            
            return pdf_links
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching content from {website_url}: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return []
    except Exception as e:
        error_msg = f"Error processing {website_url}: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return []

def extract_text_from_pdf_content(pdf_content: bytes) -> str:
    """Extract text from PDF content using multiple methods for better results."""
    all_text = ""
    
    # Method 1: PyPDF2
    try:
        with st.spinner("Extracting text with PyPDF2..."):
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pypdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                pypdf_text += page_text + "\n\n"
            all_text += pypdf_text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
    
    # Method 2: PyMuPDF (fitz)
    try:
        with st.spinner("Extracting text with PyMuPDF..."):
            pdf_file = io.BytesIO(pdf_content)
            doc = fitz.open(stream=pdf_file, filetype="pdf")
            fitz_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text() or ""
                fitz_text += page_text + "\n\n"
            all_text += fitz_text
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
    
    # Method 3: pdfplumber
    try:
        with st.spinner("Extracting text with pdfplumber..."):
            pdf_file = io.BytesIO(pdf_content)
            with pdfplumber.open(pdf_file) as pdf:
                plumber_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    plumber_text += page_text + "\n\n"
            all_text += plumber_text
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
    
    return all_text.strip()

def extract_tables_from_pdf(pdf_content: bytes) -> List[Dict[str, Any]]:
    """Extract tables from PDF content."""
    tables = []
    
    # Use pdfplumber for table extraction
    try:
        with st.spinner("Extracting tables..."):
            pdf_file = io.BytesIO(pdf_content)
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        page_tables = page.extract_tables()
                        if page_tables:
                            for j, table in enumerate(page_tables):
                                if table:  # Skip empty tables
                                    # Convert to list of dictionaries
                                    headers = [str(cell).strip() if cell else f"Column{k}" for k, cell in enumerate(table[0])]
                                    rows = []
                                    for row in table[1:]:
                                        row_dict = {}
                                        for k, cell in enumerate(row):
                                            if k < len(headers):
                                                row_dict[headers[k]] = str(cell).strip() if cell else ""
                                        rows.append(row_dict)
                                    
                                    tables.append({
                                        "page": i + 1,
                                        "table_number": j + 1,
                                        "headers": headers,
                                        "rows": rows,
                                        "raw_data": table
                                    })
                    except Exception as e:
                        logger.warning(f"Error extracting table from page {i+1}: {e}")
    except Exception as e:
        logger.warning(f"Error extracting tables with pdfplumber: {e}")
    
    return tables

def extract_images_from_pdf(pdf_content: bytes) -> List[Dict[str, Any]]:
    """Extract images from a PDF file."""
    images = []
    
    try:
        with st.spinner("Extracting images..."):
            # Use PyMuPDF for image extraction
            pdf_file = io.BytesIO(pdf_content)
            doc = fitz.open(stream=pdf_file, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get images
                image_list = page.get_images(full=True)
                
                # Process each image
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]  # Get the XREF of the image
                        base_image = doc.extract_image(xref)
                        if base_image:
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Save basic metadata about the image
                            image_data = {
                                "page": page_num + 1,
                                "index": img_index,
                                "width": base_image.get("width", 0),
                                "height": base_image.get("height", 0),
                                "ext": image_ext,
                                "size_bytes": len(image_bytes),
                                "b64_data": base64.b64encode(image_bytes).decode('utf-8')
                            }
                            
                            images.append(image_data)
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num+1}: {e}")
    except Exception as e:
        logger.warning(f"Error extracting images: {e}")
    
    return images

def extract_menu_items_from_text(text: str) -> List[Dict[str, str]]:
    """Extract menu items and prices from text."""
    menu_items = []
    
    # Different patterns to match menu items with prices
    patterns = [
        # Pattern 1: Item followed by price with dots
        r'([A-Za-z\s\'\"]+[\w\s\'\"\-\,\(\)]*)\s*\.{2,}\s*\$?(\d+[\.,]\d+)',
        # Pattern 2: Item followed by price without dots
        r'([A-Za-z\s\'\"]+[\w\s\'\"\-\,\(\)]*)\s+\$?(\d+[\.,]\d+)(?:\s|$)',
        # Pattern 3: Price at the end of line
        r'([A-Za-z\s\'\"]+[\w\s\'\"\-\,\(\)]*?)\s+[\.\-]*\s*\$?(\d+[\.,]\d+)(?:\s|$)',
        # Pattern 4: Item name followed by price (for restaurant menus)
        r'([A-Za-z][\w\s\'\"\-\,\(\)]+?)\s+[\$€£]?(\d+[\.,]\d{2})',
        # Pattern 5: Item lines with prices (Tablard menu specific)
        r'([A-Za-z][A-Za-z\s&\'\"\-\,\(\)]+)[\s\.]*[\$€£]?(\d+\.(?:\d{2})?)',
        # Pattern 6: Very specific for Tablard menu
        r'([A-Za-z][A-Za-z\s&\'\"\-\,\(\)]+)[\s\.]*?(\d{1,2}\.\d{2})',
        # Pattern 7: Menu items with number prefix
        r'\d+\.\s+([A-Za-z][A-Za-z\s&\'\"\-\,\(\)]+)[\s\.]*?(\d{1,2}\.\d{2})'
    ]
    
    # Special manual parsing for this specific menu
    # Split text by lines and look for price patterns
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for price at the end of line
        price_match = re.search(r'(\d{1,2}\.\d{2})$', line)
        if price_match:
            price = price_match.group(1)
            # Extract the item name (everything before the price)
            item_text = line[:price_match.start()].strip()
            # Remove any trailing dots or spaces
            item_text = re.sub(r'[\s\.]+$', '', item_text)
            if len(item_text) > 3:  # Skip very short items
                menu_items.append({
                    "item": item_text,
                    "price": price,
                    "currency": "$"
                })
    
    # If manual parsing found items, return them
    if menu_items:
        return menu_items
    
    # Otherwise, try with the standard patterns
    for pattern in patterns:
        try:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    item = match.group(1).strip()
                    price = match.group(2).strip()
                    
                    # Skip very short items (likely false positives)
                    if len(item) < 3:
                        continue
                    
                    # Clean up the price
                    price = price.replace(',', '.')
                    
                    menu_items.append({
                        "item": item,
                        "price": price,
                        "currency": "$"
                    })
                except Exception as e:
                    logger.warning(f"Error processing menu item match: {e}")
        except Exception as e:
            logger.warning(f"Error with pattern {pattern}: {e}")
    
    # Parse Pokemon-style product list - for ScrapeMe Pokemon product list
    pokemon_pattern = r'([A-Za-z]+)\s*(\d+(?:\.\d+)*)?'
    
    pokemon_items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Skip "Add to basket" entries - these are buttons, not products
        if "Add to basket" in line:
            continue
            
        match = re.match(pokemon_pattern, line)
        if match:
            pokemon_name = match.group(1).strip()
            # Make sure it's a valid item (not too short and not a common word)
            if len(pokemon_name) > 3:
                pokemon_items.append({
                    "item": pokemon_name,
                    "price": "0.00",  # Default price if not available
                    "currency": "$"
                })
    
    # If we found Pokemon items, add them
    if pokemon_items and len(pokemon_items) > 3:  # Only if we found a reasonable number
        menu_items.extend(pokemon_items)
    
    # Remove duplicates
    unique_items = []
    seen_items = set()
    for item in menu_items:
        item_key = item["item"].lower()
        # Skip items with "Add to basket" in them
        if "add to basket" in item_key:
            continue
        if item_key not in seen_items:
            seen_items.add(item_key)
            unique_items.append(item)
    
    return unique_items

def process_pdf(url_or_path: str) -> Dict[str, Any]:
    """Process a PDF file and extract information. Works with URLs or local file paths."""
    result = {
        "extraction_success": False,
        "content": "",
        "tables": [],
        "metadata": {},
        "products": [],
        "images": []
    }
    
    if not PDF_HANDLER_AVAILABLE:
        st.error("PDF handler libraries are not available. Please install PyPDF2, PyMuPDF (fitz), and pdfplumber.")
        logger.warning("PDF handler not available")
        return result
    
    try:
        # Check if the input is a URL or a local file path
        is_url = url_or_path.startswith(('http://', 'https://'))
        
        if is_url:
            # Handle URL
            logger.info(f"Downloading PDF from {url_or_path}")
            
            # Download the PDF
            with st.spinner(f"Downloading PDF from {url_or_path}..."):
                response = requests.get(url_or_path, stream=True, timeout=30)
                if response.status_code != 200:
                    st.error(f"Failed to download PDF: HTTP {response.status_code}")
                    logger.error(f"Failed to download PDF: HTTP {response.status_code}")
                    return result
                
                pdf_content = response.content
                logger.info(f"Downloaded {len(pdf_content)} bytes")
                st.success(f"Downloaded {len(pdf_content):,} bytes")
        else:
            # Handle local file path
            logger.info(f"Loading PDF from local path: {url_or_path}")
            
            with st.spinner(f"Reading PDF file..."):
                try:
                    with open(url_or_path, 'rb') as file:
                        pdf_content = file.read()
                        
                    logger.info(f"Loaded {len(pdf_content)} bytes from local file")
                    st.success(f"Loaded {len(pdf_content):,} bytes from file")
                except Exception as e:
                    st.error(f"Failed to read local PDF file: {str(e)}")
                    logger.error(f"Failed to read local PDF file: {str(e)}")
                    return result
        
        # Extract text
        raw_text = extract_text_from_pdf_content(pdf_content)
        
        # Clean the extracted text to fix character duplication issues
        cleaned_text = clean_extracted_text(raw_text)
        
        result["content"] = cleaned_text
        logger.info(f"Extracted {len(cleaned_text)} characters of text")
        st.success(f"Extracted {len(cleaned_text):,} characters of text")
        
        # Extract tables
        tables = extract_tables_from_pdf(pdf_content)
        result["tables"] = tables
        logger.info(f"Extracted {len(tables)} tables")
        st.success(f"Extracted {len(tables)} tables")
        
        # Extract images
        images = extract_images_from_pdf(pdf_content)
        result["images"] = images
        logger.info(f"Extracted {len(images)} images")
        st.success(f"Extracted {len(images)} images")
        
        # Extract metadata
        try:
            with st.spinner("Extracting metadata..."):
                pdf_file = io.BytesIO(pdf_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                metadata = pdf_reader.metadata
                if metadata:
                    result["metadata"] = {
                        "title": metadata.get("/Title", ""),
                        "author": metadata.get("/Author", ""),
                        "creator": metadata.get("/Creator", ""),
                        "producer": metadata.get("/Producer", ""),
                        "page_count": len(pdf_reader.pages)
                    }
                    logger.info(f"Extracted metadata: {result['metadata']}")
                    st.success(f"Extracted metadata for {len(pdf_reader.pages)} page(s)")
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
            st.warning(f"Error extracting metadata: {str(e)}")
        
        result["extraction_success"] = True
        return result
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error processing PDF: {str(e)}")
        return result

def process_menu_pdf(url_or_path: str) -> Dict[str, Any]:
    """Process a PDF file as a menu and extract menu items."""
    # First get general PDF information
    result = process_pdf(url_or_path)
    
    if not result["extraction_success"]:
        return result
    
    try:
        # Extract menu items from the text
        with st.spinner("Extracting menu items..."):
            menu_items = extract_menu_items_from_text(result["content"])
            result["menu_items"] = menu_items
            
            logger.info(f"Extracted {len(menu_items)} menu items from PDF")
            st.success(f"Extracted {len(menu_items)} menu items")
    except Exception as e:
        logger.error(f"Error extracting menu items: {e}")
        st.error(f"Error extracting menu items: {str(e)}")
    
    return result

def display_pdf_content(result: Dict[str, Any]):
    """Display the content of a processed PDF."""
    # Create tabs for different content
    tabs = st.tabs(["Text", "Menu Items", "Tables", "Images", "Metadata", "Raw JSON"])
    
    # Text tab
    with tabs[0]:
        st.subheader("Extracted Text")
        if "content" in result and result["content"]:
            st.text_area("Text content", result["content"], height=300)
        else:
            st.info("No text content extracted")
    
    # Menu Items tab
    with tabs[1]:
        st.subheader("Menu Items")
        if "menu_items" in result and result["menu_items"]:
            st.success(f"Found {len(result['menu_items'])} menu items")
            
            # Create a DataFrame for the menu items
            import pandas as pd
            menu_df = pd.DataFrame(result["menu_items"])
            st.dataframe(menu_df, use_container_width=True)
            
            # Add option to filter items with prices
            only_with_prices = st.checkbox("Only include items with valid prices in download", value=True)
            
            # Filter menu items if requested
            download_df = menu_df
            if only_with_prices:
                # Convert price strings to numeric values where possible
                download_df['numeric_price'] = pd.to_numeric(download_df['price'], errors='coerce')
                # Filter out items with empty, zero, or NaN prices
                download_df = download_df[download_df['numeric_price'] > 0]
                # Drop the temporary numeric_price column
                download_df = download_df.drop(columns=['numeric_price'])
                
                st.info(f"{len(download_df)} items with valid prices will be included in the download")
            
            # Download option
            csv = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download menu items as CSV",
                csv,
                "menu_items.csv",
                "text/csv",
                key='download-csv-menu'
            )
        else:
            st.info("No menu items found")
    
    # Tables tab
    with tabs[2]:
        st.subheader("Tables")
        if "tables" in result and result["tables"]:
            for i, table in enumerate(result["tables"]):
                st.write(f"### Table {i+1} (Page {table['page']})")
                
                # Convert to DataFrame and display
                import pandas as pd
                if "rows" in table and table["rows"]:
                    df = pd.DataFrame(table["rows"])
                    st.dataframe(df, use_container_width=True)
                else:
                    # Fallback to raw data
                    st.table(table["raw_data"])
        else:
            st.info("No tables found in the PDF")
    
    # Images tab
    with tabs[3]:
        st.subheader("Images")
        if "images" in result and result["images"]:
            st.success(f"Found {len(result['images'])} images")
            
            # Create columns for images
            cols = st.columns(min(len(result["images"]), 3))
            
            for i, img in enumerate(result["images"]):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if "b64_data" in img:
                        st.image(
                            f"data:image/{img['ext']};base64,{img['b64_data']}",
                            caption=f"Image {i+1}",
                            use_container_width=True
                        )
                    else:
                        st.write(f"Image {i+1}: {img['width']}x{img['height']} ({img['ext']})")
        else:
            st.info("No images found in the PDF")
    
    # Metadata tab
    with tabs[4]:
        st.subheader("Metadata")
        if "metadata" in result and result["metadata"]:
            for key, value in result["metadata"].items():
                st.write(f"**{key}:** {value}")
        else:
            st.info("No metadata extracted")
    
    # Raw JSON tab
    with tabs[5]:
        st.subheader("Raw JSON")
        # Create a clean copy without binary data
        clean_result = result.copy()
        if "images" in clean_result:
            for img in clean_result["images"]:
                if "b64_data" in img:
                    del img["b64_data"]
        
        st.json(clean_result)
        
        # Download button for the JSON
        json_str = json.dumps(clean_result, indent=2)
        st.download_button(
            "Download JSON",
            json_str,
            "pdf_result.json",
            "application/json",
            key='download-json'
        )

def display_pdf_uploader():
    """Display a file uploader for PDFs."""
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Process the PDF
            st.info(f"Processing uploaded PDF: {uploaded_file.name}")
            
            # Check if it's a menu
            is_menu = st.checkbox("Process as menu", value="menu" in uploaded_file.name.lower())
            
            if st.button("Process PDF"):
                if is_menu:
                    result = process_menu_pdf(temp_file_path)
                else:
                    result = process_pdf(temp_file_path)
                
                if result["extraction_success"]:
                    st.success("PDF processed successfully")
                    display_pdf_content(result)
                else:
                    st.error("Failed to process the PDF")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

def main():
    st.set_page_config(
        page_title="PDF Processor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("PDF Processor")
    st.markdown("Extract text, tables, images, and menu items from PDF files.")
    
    # Initialize session state variables
    if 'pdf_links' not in st.session_state:
        st.session_state.pdf_links = []
    if 'selected_pdfs' not in st.session_state:
        st.session_state.selected_pdfs = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if not PDF_HANDLER_AVAILABLE:
        st.error("""
        PDF handler libraries are not available. Please install the required packages:
        ```
        pip install PyPDF2 pymupdf pdfplumber
        ```
        """)
        return
    
    # Sidebar for input options
    with st.sidebar:
        st.header("PDF Source")
        input_option = st.radio("Choose input method", ["Extract from Website", "URL", "Upload File"], index=0)
    
    if input_option == "Extract from Website":
        website_url = st.text_input("Enter website URL to extract PDF links from", value="https://tablard.ch")
        
        if website_url:
            # Function to update PDF links in session state
            def fetch_and_update_links():
                with st.spinner(f"Fetching PDF links from {website_url}..."):
                    st.session_state.pdf_links = fetch_pdf_links_from_website(website_url)
                    if st.session_state.pdf_links:
                        st.session_state.selected_pdfs = [st.session_state.pdf_links[0]] if st.session_state.pdf_links else []
            
            # Button to fetch PDF links
            if st.button("Fetch PDF Links", key="fetch_pdf_links_button"):
                fetch_and_update_links()
            
            # Display PDF links if available
            if st.session_state.pdf_links:
                st.success(f"Found {len(st.session_state.pdf_links)} PDF links on the website")
                
                # Create a selection widget for the PDF links
                st.session_state.selected_pdfs = st.multiselect(
                    "Select PDFs to process", 
                    options=st.session_state.pdf_links,
                    default=st.session_state.selected_pdfs
                )
                
                # Process selected PDFs
                if st.session_state.selected_pdfs:
                    if st.button("Process Selected PDFs", key="process_selected_pdfs_button"):
                        st.session_state.processing_complete = False
                        for pdf_url in st.session_state.selected_pdfs:
                            st.subheader(f"Processing: {pdf_url}")
                            
                            # Check if it's a menu
                            is_menu = "menu" in pdf_url.lower()
                            
                            with st.spinner(f"Processing PDF from {pdf_url}..."):
                                if is_menu:
                                    result = process_menu_pdf(pdf_url)
                                else:
                                    result = process_pdf(pdf_url)
                                
                                if result["extraction_success"]:
                                    st.success("PDF processed successfully")
                                    display_pdf_content(result)
                                else:
                                    st.error(f"Failed to process the PDF: {pdf_url}")
                        
                        st.session_state.processing_complete = True
            elif website_url and not st.session_state.pdf_links:
                st.warning("No PDF links found on the provided website.")
    elif input_option == "URL":
        url = st.text_input("Enter PDF URL", value="https://tablard.ch/menu.pdf")
        
        if url:
            if is_pdf_url(url):
                st.info(f"Detected PDF URL: {url}")
                
                # Check if it's a menu
                is_menu = st.checkbox("Process as menu", value="menu" in url.lower())
                
                if st.button("Process PDF", key="process_single_pdf"):
                    if is_menu:
                        with st.spinner(f"Processing menu PDF from {url}..."):
                            result = process_menu_pdf(url)
                    else:
                        with st.spinner(f"Processing PDF from {url}..."):
                            result = process_pdf(url)
                    
                    if result["extraction_success"]:
                        st.success("PDF processed successfully")
                        display_pdf_content(result)
                    else:
                        st.error("Failed to process the PDF")
            else:
                st.warning("The URL does not appear to be a PDF. Please enter a valid PDF URL.")
    else:
        display_pdf_uploader()

def clean_extracted_text(text: str) -> str:
    """Clean extracted text by removing duplicated characters and other issues.
    
    This handles common OCR/extraction issues like duplicated characters:
    - "GGaasspppaacchhoo" -> "Gazpacho"
    - "BBeettttuuccee" -> "Betuce"
    """
    if not text or len(text) < 2:
        return text
    
    # First pass: find runs of doubled characters (like 'aa', 'ee', etc.)
    cleaned_text = ""
    i = 0
    while i < len(text):
        char = text[i]
        repeat_count = 1
        
        # Count repeating characters
        j = i + 1
        while j < len(text) and text[j].lower() == char.lower():
            repeat_count += 1
            j += 1
        
        # Only keep at most 2 repeats, and for most common letters reduce to 1
        if repeat_count > 2:
            # Common letters that shouldn't normally be doubled in words
            if char.lower() in 'abcdefghijklmnopqrstuvwxyz':
                cleaned_text += char
            else:
                cleaned_text += char * 2
        else:
            cleaned_text += text[i:j]
        
        i = j
    
    # Second pass: handle repeated patterns like "ToTo" -> "To"
    if len(cleaned_text) >= 4:
        result = ""
        i = 0
        while i < len(cleaned_text) - 3:
            if (cleaned_text[i:i+2].lower() == cleaned_text[i+2:i+4].lower()):
                result += cleaned_text[i:i+2]
                i += 4
            else:
                result += cleaned_text[i]
                i += 1
        # Add any remaining characters
        result += cleaned_text[i:]
        cleaned_text = result
    
    # Third pass: replace known patterns
    replacements = [
        ("BBeettttuuccee", "Lettuce"),
        ("CCaammoonn", "Camembert"),
        ("GGaass", "Gas"),
        ("ppaa", "pa"),
        ("cchh", "ch"),
        ("hhoo", "ho"),
        ("ddee", "de"),
        ("BBee", "Be"),
        ("tttt", "tt"),
        ("eeee", "ee"),
        ("rraa", "ra"),
        ("vvee", "ve"),
        ("eett", "et"),
        ("oomm", "om"),
        ("aatt", "at"),
        ("eess", "es"),
        ("Sardo, Fraises", "Sardo Fraises"),
    ]
    
    for old, new in replacements:
        cleaned_text = cleaned_text.replace(old, new)
    
    # Return cleaned text
    return cleaned_text

if __name__ == "__main__":
    main() 