import io
import requests
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from loguru import logger
import tempfile
import os

# Try to import the translation handler, but provide a fallback if not available
try:
    from translation_handler import detect_language, translate_text
except ImportError:
    # Define fallback functions
    def detect_language(text):
        logger.warning("Translation handler not available - using fallback language detection")
        # Simple fallback detection - not reliable but better than nothing
        # Check for common French words
        french_words = ['le', 'la', 'du', 'de', 'un', 'une', 'et', 'est', 'au', 'aux', 'avec', 'ce', 'ces', 'cette']
        french_count = sum(1 for word in text.lower().split() if word in french_words)
        
        # Check for common German words
        german_words = ['der', 'die', 'das', 'ein', 'eine', 'und', 'ist', 'von', 'mit', 'zu', 'für', 'in', 'auf']
        german_count = sum(1 for word in text.lower().split() if word in german_words)
        
        if french_count > german_count and french_count > 5:
            return 'fr'
        elif german_count > french_count and german_count > 5:
            return 'de'
        return 'en'  # Default to English
    
    def translate_text(text, source_lang=None, target_lang='en'):
        logger.warning("Translation handler not available - returning original text")
        return text

def is_pdf_url(url: str) -> bool:
    """
    Check if a URL points to a PDF file.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if URL points to a PDF, False otherwise
    """
    if not url:
        return False
         
    # Check file extension
    if url.lower().endswith('.pdf'):
        return True
         
    # Check URL path
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    if path.endswith('.pdf'):
        return True
         
    # Check common PDF URL patterns
    if '/pdf/' in path or 'download=pdf' in parsed_url.query:
        return True
         
    # Try to get content type without downloading the whole file
    try:
        response = requests.head(url, timeout=5)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type:
            return True
    except Exception as e:
        logger.warning(f"Error checking content type for {url}: {e}")
         
    return False

def extract_from_pdf(pdf_url: str, target_lang: str = 'en') -> Optional[Dict[str, Any]]:
    """
    Extract text and structured data from a PDF URL with translation support.
    
    Args:
        pdf_url: URL of the PDF file
        target_lang: Target language for translation (default: 'en' for English)
        
    Returns:
        dict: Extracted content and metadata, or None if extraction fails
    """
    logger.info(f"Extracting content from PDF: {pdf_url}")
    
    try:
        # Download the PDF
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        pdf_content = io.BytesIO(response.content)
        extracted_data = {
            'text': '',
            'tables': [],
            'metadata': {},
            'pages': 0,
            'source_url': pdf_url,
            'products': [],  # Will store extracted products/prices
            'detected_language': None,
            'is_translated': False
        }
        
        # Direct extraction using BytesIO instead of temporary file
        try:
            # Extract text with PyMuPDF directly from memory
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            all_text = ""
            
            # Extract metadata
            metadata = {
                'title': doc.metadata.get("title", ""),
                'author': doc.metadata.get("author", ""),
                'subject': doc.metadata.get("subject", ""),
                'keywords': doc.metadata.get("keywords", ""),
                'creator': doc.metadata.get("creator", ""),
                'producer': doc.metadata.get("producer", ""),
                'creation_date': doc.metadata.get("creationDate", ""),
                'modification_date': doc.metadata.get("modDate", ""),
            }
            extracted_data['metadata'] = metadata
            extracted_data['pages'] = len(doc)
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                all_text += text + "\n\n"
            
            extracted_data['text'] = all_text.strip()
            
            # Detect language
            detected_lang = detect_language(all_text)
            extracted_data['detected_language'] = detected_lang
            
            # Translate text if needed
            if detected_lang and detected_lang != target_lang and detected_lang != 'en':
                try:
                    translated_text = translate_text(all_text, detected_lang, target_lang)
                    extracted_data['original_text'] = all_text
                    extracted_data['text'] = translated_text
                    extracted_data['is_translated'] = True
                    logger.info(f"Translated PDF content from {detected_lang} to {target_lang}")
                except Exception as e:
                    logger.error(f"Translation failed: {e}")
            
            # For table extraction we'll still need a temp file
            # Use a more robust approach with context manager
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract tables using the temp file
                extracted_data['tables'] = extract_tables_from_pdf(tmp_file_path)
            finally:
                # Make sure to clean up the temp file
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")
            
            # Extract products/menu items
            extracted_data['products'] = extract_products_from_pdf_text(
                extracted_data['text'], pdf_url
            )
            
            # Process menus specifically
            if is_likely_menu(extracted_data['text']):
                menu_items = process_pdf_menu(extracted_data)
                if menu_items:
                    extracted_data['menu_items'] = menu_items
                    # Add menu items to products if they're not already there
                    existing_names = {p.get('product_name', '') for p in extracted_data['products']}
                    for item in menu_items:
                        if item.get('product_name') not in existing_names:
                            extracted_data['products'].append(item)
            
            logger.info(f"Successfully extracted PDF content: {len(extracted_data['text'])} chars of text, " +
                   f"{len(extracted_data['tables'])} tables, {len(extracted_data['products'])} products")
            
            if extracted_data['is_translated']:
                logger.info(f"Translated from {extracted_data['detected_language']} to {target_lang}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error during PDF extraction: {e}")
            # Try to extract text directly using our text extraction function
            pdf_content.seek(0)
            text = extract_text_from_pdf_content(pdf_content)
            if text:
                extracted_data['text'] = text
                extracted_data['products'] = extract_products_from_pdf_text(text, pdf_url)
                return extracted_data
            raise  # Re-raise if we couldn't extract any text
        
    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        return None

def extract_tables_from_pdf(pdf_path):
    """
    Extract tables from a PDF file using both PyMuPDF and pdfplumber for better coverage.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        list: List of extracted tables
    """
    tables = []
    
    # Try pdfplumber first (better for well-structured tables)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        # Filter out empty tables
                        if table and any(any(cell for cell in row) for row in table):
                            # Convert to DataFrame for easier manipulation
                            df = pd.DataFrame(table)
                            # Use first row as header if it seems like a header
                            if df.iloc[0].notna().all():
                                df.columns = df.iloc[0]
                                df = df.iloc[1:]
                            # Reset index for clean dataframe
                            df = df.reset_index(drop=True)
                            tables.append(df)
    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed: {e}")
    
    # If no tables found with pdfplumber, try PyMuPDF
    if not tables:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks that might represent tables
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        lines = block.get("lines", [])
                        if len(lines) > 2:  # At least 3 lines to be a table
                            # Check if lines seem to form a table (aligned columns)
                            spans_per_line = [len(line.get("spans", [])) for line in lines]
                            if len(set(spans_per_line)) == 1 and spans_per_line[0] > 1:
                                # Likely a table with consistent columns
                                # Extract the text
                                table_data = []
                                for line in lines:
                                    row = [span.get("text", "") for span in line.get("spans", [])]
                                    table_data.append(row)
                                
                                if table_data:
                                    df = pd.DataFrame(table_data)
                                    # Use first row as header if it seems like a header
                                    if df.iloc[0].notna().all():
                                        df.columns = df.iloc[0]
                                        df = df.iloc[1:]
                                    # Reset index
                                    df = df.reset_index(drop=True)
                                    tables.append(df)
        except Exception as e:
            logger.warning(f"PyMuPDF table extraction failed: {e}")
    
    return tables

def extract_products_from_pdf_text(text, source_url):
    """
    Extract product information from PDF text using pattern matching.
    
    Args:
        text (str): The extracted text from the PDF
        source_url (str): The URL source of the PDF
        
    Returns:
        list: List of dictionaries containing product information
    """
    products = []
    
    # Common price patterns in different currencies
    price_patterns = [
        r'(\$\d+(?:\.\d{2})?)',  # USD, CAD, AUD
        r'(\€\d+(?:\.\d{2})?)',  # EUR
        r'(\£\d+(?:\.\d{2})?)',  # GBP
        r'(\d+\s?(?:USD|EUR|CHF|CAD|AUD))',  # Currency codes
        r'(\d+[.,]\d{2}\s?(?:USD|EUR|CHF|CAD|AUD|€|\$|\£))',  # Price with currency
        r'(\d+[.,]\d{2}[-€])',   # European style
        r'((?:USD|EUR|CHF|CAD|AUD|€|\$|\£)\s?\d+[.,]\d{2})',  # Currency first
        r'(CHF\s?\d+(?:[.,]\d{2})?)',  # Swiss francs
        r'(\d+[.,]\d{2}(?:\s?[Ff]r\.?))',  # French/Swiss francs abbreviated
    ]
    
    # Look for product name patterns near prices
    lines = text.split('\n')
    
    # First pass: Look for line items with prices
    for i, line in enumerate(lines):
        for pattern in price_patterns:
            price_matches = list(re.finditer(pattern, line))
            if price_matches:
                for match in price_matches:
                    price = match.group(1)
                    
                    # Get product name before the price
                    pre_price_text = line[:match.start()].strip()
                    if not pre_price_text and i > 0:
                        # If no text before price, check previous line
                        pre_price_text = lines[i-1].strip()
                    
                    # Get product name after the price if none found before
                    if not pre_price_text:
                        post_price_text = line[match.end():].strip()
                        if not post_price_text and i < len(lines)-1:
                            # If no text after price, check next line
                            post_price_text = lines[i+1].strip()
                        
                        product_name = post_price_text
                    else:
                        product_name = pre_price_text
                    
                    # Clean up product name (remove excess punctuation, numbers at beginning)
                    product_name = re.sub(r'^[\d\.\s]+', '', product_name)
                    product_name = product_name.strip()
                    
                    # Only add if we found a product name and it's reasonably long
                    if product_name and len(product_name) > 3:
                        products.append({
                            "product_name": product_name,
                            "price": price,
                            "category": "Extracted from PDF",
                            "source_url": source_url,
                            "extracted_by": "PDF Extractor"
                        })
    
    # Second pass: Look for menu-like structures (title followed by price on same line)
    # This is common in menus where item and price are on the same line
    menu_pattern = r'([A-Za-z\s\'\"\-]+)\s+(' + '|'.join(p.strip('()') for p in price_patterns) + ')'
    for line in lines:
        menu_matches = list(re.finditer(menu_pattern, line))
        for match in menu_matches:
            if len(match.groups()) >= 2:
                product_name = match.group(1).strip()
                price = match.group(2)
                
                # Check if this is already added
                if not any(p['product_name'] == product_name and p['price'] == price for p in products):
                    products.append({
                        "product_name": product_name,
                        "price": price,
                        "category": "Menu Item",
                        "source_url": source_url,
                        "extracted_by": "PDF Menu Extractor"
                    })
    
    # Third pass: Look for sections that might contain product descriptions
    in_product_section = False
    current_product = None
    current_description = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # Empty line might end a product description
            if current_product and current_description:
                current_product["description"] = " ".join(current_description)
                current_description = []
                current_product = None
            continue
        
        # Check if line contains a price
        has_price = any(re.search(pattern, line) for pattern in price_patterns)
        
        if has_price:
            # This might be a new product
            in_product_section = True
            
            # If we have a pending product, finalize it
            if current_product and current_description:
                current_product["description"] = " ".join(current_description)
            
            # Create new product
            for pattern in price_patterns:
                price_match = re.search(pattern, line)
                if price_match:
                    price = price_match.group(1)
                    
                    # Get the product name (everything before the price)
                    product_name = line[:price_match.start()].strip()
                    
                    # Clean up product name
                    product_name = re.sub(r'^[\d\.\s]+', '', product_name)
                    product_name = product_name.strip()
                    
                    if product_name and len(product_name) > 3:
                        # Check if this is already in our list
                        existing = next((p for p in products if p['product_name'] == product_name), None)
                        if existing:
                            current_product = existing
                        else:
                            new_product = {
                                "product_name": product_name,
                                "price": price,
                                "category": "Menu Item",
                                "source_url": source_url,
                                "extracted_by": "PDF Menu Extractor"
                            }
                            products.append(new_product)
                            current_product = new_product
                        
                        current_description = []
                        break
        elif in_product_section and current_product:
            # This might be part of the product description
            current_description.append(line)
    
    # Finalize any pending product description
    if current_product and current_description:
        current_product["description"] = " ".join(current_description)
    
    return products

def is_likely_menu(text):
    """
    Determine if the PDF text is likely to be a menu.
    
    Args:
        text: The extracted text content
        
    Returns:
        bool: True if the text appears to be a menu
    """
    # Common menu section indicators
    menu_sections = [
        'appetizer', 'starter', 'entrée', 'main course', 'dessert', 'beverage',
        'drink', 'cocktail', 'wine', 'beer', 'spirit', 'breakfast', 'lunch',
        'dinner', 'brunch', 'special', 'menu du jour', 'prix fixe', 'à la carte',
        'salad', 'soup', 'sandwich', 'pasta', 'pizza', 'seafood', 'meat', 'vegetarian',
        'vegan', 'gluten-free', 'children', 'kids', 'sides', 'extras'
    ]
    
    # Multiple price indicators in the text
    price_count = sum(1 for p in [r'\$\d+', r'\€\d+', r'\£\d+', r'\d+\s?(?:USD|EUR|CHF)'] if re.search(p, text))
    
    # Check for menu sections
    section_count = sum(1 for section in menu_sections if re.search(r'\b' + section + r'\b', text.lower()))
    
    # Check for typical menu formatting (items followed by prices)
    menu_item_patterns = [
        r'[\w\s]+\.\.\.\.\s*\$\d+',  # Item.... $XX
        r'[\w\s]+\s+\$\d+',          # Item $XX
        r'[\w\s]+\s+\d+\s*\€',       # Item XX€
    ]
    
    format_count = sum(len(re.findall(pattern, text)) for pattern in menu_item_patterns)
    
    # It's likely a menu if:
    # 1. It has multiple prices (at least 5)
    # 2. It has at least 2 menu section indicators
    # 3. It has several items in menu-like formatting
    return (price_count >= 5) or (section_count >= 2) or (format_count >= 3)

def process_pdf_menu(pdf_data_or_url):
    """
    Process PDF data for menu items, handling both direct PDF data objects 
    and URLs that need to be downloaded first.
    
    Args:
        pdf_data_or_url: Either a dictionary of already-extracted PDF data,
                        or a URL string pointing to a PDF to process
        
    Returns:
        list: Extracted menu items with prices and categories
    """
    # Handle URL input
    if isinstance(pdf_data_or_url, str):
        pdf_data = extract_from_pdf(pdf_data_or_url)
        if not pdf_data:
            logger.error(f"Failed to extract PDF from URL: {pdf_data_or_url}")
            return []
    else:
        # Assume it's already extracted PDF data
        pdf_data = pdf_data_or_url
    
    # If PDF data contains pre-extracted products, return those
    if 'products' in pdf_data and pdf_data['products']:
        return pdf_data['products']
    
    # Otherwise, perform more targeted menu extraction
    menu_items = []
    
    # Extract menu sections and items using pattern recognition
    text = pdf_data.get("text", "")
    
    # Look for menu sections (capitalized headers often indicate sections)
    section_pattern = r"\n([A-Z][A-Z\s]{2,30})\n"
    sections = re.findall(section_pattern, text)
    
    current_section = "General Menu"
    lines = text.split('\n')
    
    for line in lines:
        # Check if this is a section header
        if any(section in line for section in sections):
            current_section = line.strip()
            continue
            
        # Look for menu items (name + price)
        price_match = re.search(r"(\$\d+(?:\.\d{2})?|\€\d+(?:\.\d{2})?|\£\d+(?:\.\d{2})?|\d+\s?(?:USD|EUR|CHF))", line)
        if price_match:
            price = price_match.group(1)
            # Get the item name (text before the price)
            name_part = line.split(price)[0].strip()
            if name_part:
                # Clean up the name (remove leading numbers, dots)
                name_part = re.sub(r'^[\d\.\s]+', '', name_part)
                
                menu_items.append({
                    "product_name": name_part,
                    "price": price,
                    "category": current_section,
                    "source_url": pdf_data.get('source_url', '')
                })
    
    return menu_items

def download_pdf(url, save_path=None):
    """
    Download a PDF file from a URL and optionally save it to disk.
    
    Args:
        url: URL of the PDF to download
        save_path: Optional path to save the PDF to disk
        
    Returns:
        BytesIO or None: PDF content as a file-like object, or None if download failed
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to disk if a path is provided
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        
        # Return as file-like object
        return io.BytesIO(response.content)
    
    except Exception as e:
        logger.error(f"Error downloading PDF from {url}: {e}")
        return None

# Enhanced version with better text extraction
def extract_text_from_pdf_content(pdf_content):
    """
    Extract text from PDF content using multiple methods
    
    Args:
        pdf_content: BytesIO object containing PDF data
        
    Returns:
        str: Extracted text
    """
    extracted_text = ""
    
    # Method 1: Try with PyMuPDF (usually best for complex PDFs)
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())
        extracted_text = "\n".join(text_parts)
        if extracted_text.strip():
            logger.info("Successfully extracted text using PyMuPDF")
            return extracted_text
        pdf_content.seek(0)  # Reset for next attempt
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
    
    # Method 2: Try with pdfplumber
    try:
        with pdfplumber.open(pdf_content) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3)
                if text:
                    text_parts.append(text)
            extracted_text = "\n".join(text_parts)
        if extracted_text.strip():
            logger.info("Successfully extracted text using pdfplumber")
            return extracted_text
        pdf_content.seek(0)  # Reset for next attempt
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
    
    # Method 3: Try with PyPDF2
    try:
        reader = PdfReader(pdf_content)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        extracted_text = "\n".join(text_parts)
        if extracted_text.strip():
            logger.info("Successfully extracted text using PyPDF2")
            return extracted_text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
    
    # If we got here, all methods failed or returned empty text
    if not extracted_text.strip():
        logger.error("All PDF extraction methods failed")
    
    return extracted_text

def process_pdf(url):
    """
    Main function to process a PDF URL
    
    Args:
        url: URL of the PDF file
        
    Returns:
        dict: Dictionary with PDF content and metadata
    """
    logger.info(f"Processing PDF: {url}")
    
    # Download the PDF
    pdf_content = download_pdf(url)
    if not pdf_content:
        return {
            "url": url,
            "content": "",
            "content_type": "pdf",
            "extraction_success": False
        }
    
    # Extract text
    pdf_content.seek(0)
    text = extract_text_from_pdf_content(pdf_content)
    
    # Extract tables
    pdf_content.seek(0)
    result = extract_from_pdf(url)
    
    if not result:
        return {
            "url": url,
            "content": text,
            "content_type": "pdf",
            "extraction_success": text != "",
            "tables": [],
            "num_tables": 0,
            "metadata": {
                "title": "",
                "word_count": len(text.split()) if text else 0,
                "character_count": len(text) if text else 0,
                "num_tables": 0
            }
        }
    
    return {
        "url": url,
        "content": result.get("text", text),
        "content_type": "pdf",
        "extraction_success": True,
        "tables": result.get("tables", []),
        "num_tables": len(result.get("tables", [])),
        "products": result.get("products", []),
        "metadata": result.get("metadata", {})
    }

def process_menu_pdf(url):
    """
    Special function for processing menu PDFs, with extra heuristics
    for menu items, prices, etc.
    
    Args:
        url: URL of the menu PDF
        
    Returns:
        dict: Dictionary with processed menu data
    """
    result = process_pdf(url)
    if not result["extraction_success"]:
        return result
    
    # Try to get menu items from extracted_pdf result
    pdf_data = None
    try:
        pdf_data = extract_from_pdf(url)
    except Exception as e:
        logger.warning(f"Could not use extract_from_pdf: {e}")
    
    if pdf_data and "products" in pdf_data:
        result["menu_items"] = pdf_data["products"]
        result["item_count"] = len(result["menu_items"])
    else:
        # Add menu-specific processing using regex
        import re
        menu_items = []
        
        # Common patterns for menu items and prices
        price_patterns = [
            r"[€$£]?\s*(\d+[\.,]\d{2})",  # €12.95, $ 12.95
            r"(\d+[\.,]\d{2})[€$£]",  # 12.95€
            r"[€$£]?\s*(\d+)",  # €12, $ 12
        ]
        
        # Look for menu items
        lines = result["content"].splitlines()
        for line in lines:
            # Skip very short or empty lines
            if len(line.strip()) < 5:
                continue
                
            # Check for price patterns
            for pattern in price_patterns:
                try:
                    matches = list(re.finditer(pattern, line))
                    for match in matches:
                        price = match.group(1)
                        # Get text before price as item name
                        item_name = line[:match.start()].strip()
                        if item_name and len(item_name) > 3:
                            menu_items.append({
                                "item": item_name,
                                "price": price,
                                "currency": "€" if "€" in line else "$" if "$" in line else "£" if "£" in line else ""
                            })
                except Exception as e:
                    logger.warning(f"Regex error on line '{line}': {e}")
                    continue
        
        result["menu_items"] = menu_items
        result["item_count"] = len(menu_items)
    
    return result
