"""
Utility Functions for Document Processing
=========================================
This module provides utility functions for campaign date extraction, PDF text processing,
file path management, and schema building for the document comparison pipeline.

Functions:
    get_campaign_dates: Extract campaign dates from folder names
    get_campaign_folder_path: Get campaign folder paths and extract ZIP files
    extract_text_from_pdf_plumber: Extract text from PDF using pdfplumber
    extract_text_from_pdf_pypdf2: Extract text from PDF using PyPDF2
    chunk_text_by_ordered_markers: Split text into chunks based on markers
    get_commodity: Determine commodity type from filename
    build_schema: Build Spark DataFrame schema from configuration
    filter_tables_pdfplumber_extraction: Filter out tables from PDF pages
    not_within_bboxes: Check if object is within table bounding boxes
"""

import os
import re
import zipfile
import pdfplumber
from PyPDF2 import PdfReader
from functools import partial
from databricks.sdk.runtime import dbutils
from datetime import datetime
import pyspark.sql.types as T
import time
import random


def get_campaign_dates(path):
    """
    Extract Campaign Dates from Folder Names
    ========================================

    Scans folder names in the given path, extracts dates in DDMMYYYY format,
    and returns the current and past campaign month-year strings.

    Args:
        path (str): The DBFS path to scan for campaign folders

    Returns:
        tuple: (current_campaign_mmYYYY, past_campaign_mmYYYY) strings
               or None if no valid date found
    """
    # Get list of files/folders from DBFS path
    files = dbutils.fs.ls(path)
    date_candidates = []

    # Process each folder to extract date information
    for f in files:
        folder_name = f.name.rstrip("/")  # Clean trailing slash from folder name

        # Find all 8-digit numeric substrings that could be dates
        date_matches = re.findall(r"(\d{8})", folder_name)

        # Validate each potential date string
        for date_match in date_matches:
            try:
                # Parse as DDMMYYYY format (e.g., "01012024" = January 1, 2024)
                parsed_date = datetime.strptime(date_match, "%d%m%Y")
                date_candidates.append((parsed_date, folder_name))
            except ValueError:
                continue  # Skip invalid date formats

    # Return None if no valid dates found
    if not date_candidates:
        print("No valid dates found in folder names.")
        return None

    # Find the most recent campaign date
    latest_campaign = max(date_candidates, key=lambda x: x[0])
    current_date = latest_campaign[0]
    currentcampaign_year = current_date.year
    currentcampaign_month = current_date.month

    # Calculate past campaign period (previous month)
    if currentcampaign_month == 1:
        # Handle January case - previous month is December of previous year
        pastcampaign_month = 12
        pastcampaign_year = currentcampaign_year - 1
    else:
        # Standard case - previous month of same year
        pastcampaign_month = currentcampaign_month - 1
        pastcampaign_year = currentcampaign_year

    # Format as MMYYYY strings for consistent naming
    currentcampaign_month_year = f"{currentcampaign_month:02}{currentcampaign_year:04}"
    pastcampaign_month_year = f"{pastcampaign_month:02}{pastcampaign_year:04}"

    return currentcampaign_month_year, pastcampaign_month_year


def get_campaign_folder_path(volume_directory: str, campaign_month_year: str):
    """
    Get Campaign Folder Paths and Extract ZIP Files
    ===============================================

    Finds campaign folders matching the given month-year pattern,
    extracts any ZIP files found, and returns paths to extracted folders.

    Args:
        volume_directory (str): Base directory path containing campaign folders
        campaign_month_year (str): Campaign period in MMYYYY format

    Returns:
        list: List of extracted folder paths, or error message string
    """
    # Create regex pattern to match campaign folders (e.g., "Campagna_01012024")
    campaign_pattern = re.compile(
        rf"Campagna_(0[1-9]|[12][0-9]|3[01]){campaign_month_year}"
    )

    # Find all folders matching the campaign pattern
    campaign_folders = [
        folder
        for folder in os.listdir(volume_directory)
        if campaign_pattern.match(folder)
    ]

    # Return error message if no matching folders found
    if not campaign_folders:
        return "No campaign folder found"

    # Identify ZIP files from campaign folders
    zip_files = [
        folder for folder in campaign_folders if folder.lower().endswith(".zip")
    ]
    extracted_paths = []

    # Extract ZIP files if found
    if zip_files:
        for zip_file in zip_files:
            zip_file_path = os.path.join(volume_directory, zip_file)

            # Extract ZIP file contents to volume directory
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(volume_directory)

            # Add extracted folder path (without .zip extension)
            extracted_folder_path = os.path.splitext(zip_file_path)[0]
            extracted_paths.append(extracted_folder_path)

            # Remove ZIP file after extraction
            os.remove(zip_file_path)

    # Add non-ZIP folders to extracted paths
    for folder in campaign_folders:
        if "." not in folder:  # Skip files with extensions
            folder_path = os.path.join(volume_directory, folder)
            extracted_paths.append(folder_path)

    return extracted_paths


def filter_tables_pdfplumber_extraction(
    page: pdfplumber.page.Page,
) -> pdfplumber.page.Page:
    """
    Filter Out Tables from PDF Page
    ===============================

    Removes table content from a PDF page to focus on text extraction,
    useful when tables interfere with text flow analysis.

    Args:
        page (pdfplumber.page.Page): PDF page object to filter

    Returns:
        pdfplumber.page.Page: Filtered page object without table content
    """
    # Check if page contains any tables
    page_tables = page.find_tables()

    if page_tables != []:
        # Get bounding boxes of all tables on the page
        # Adapted from: https://github.com/jsvine/pdfplumber/issues/242#issuecomment-668448246
        table_bboxes = [table.bbox for table in page_tables]

        # Create filter function to exclude objects within table bounding boxes
        bbox_filter_function = partial(not_within_bboxes, bboxes=table_bboxes)

        # Apply filter to remove table content from page
        filtered_page = page.filter(bbox_filter_function)
        return filtered_page

    return page


def not_within_bboxes(obj, bboxes):
    """
    Check if Object is Within Table Bounding Boxes
    ==============================================

    Determines whether a PDF object (text, line, etc.) is located
    within any of the specified table bounding boxes.

    Args:
        obj (dict): PDF object with position coordinates
        bboxes (list): List of table bounding boxes

    Returns:
        bool: True if object is NOT within any bounding box, False otherwise
    """

    def obj_in_bbox(bbox_coordinates):
        """
        Check if Object is Within Specific Bounding Box
        ==============================================

        Determines if an object's center point falls within a table's bounding box.

        Reference: https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404
        """
        # Calculate object center coordinates
        vertical_midpoint = (obj["top"] + obj["bottom"]) / 2
        horizontal_midpoint = (obj["x0"] + obj["x1"]) / 2

        # Extract bounding box coordinates
        x0, top, x1, bottom = bbox_coordinates

        # Check if center point is within bounding box
        within_horizontal_bounds = (horizontal_midpoint >= x0) and (
            horizontal_midpoint < x1
        )
        within_vertical_bounds = (vertical_midpoint >= top) and (
            vertical_midpoint < bottom
        )

        return within_horizontal_bounds and within_vertical_bounds

    # Return True if object is NOT in any bounding box
    return not any(obj_in_bbox(bbox) for bbox in bboxes)


def get_commodity(file_name: str) -> str:
    """
    Determine Commodity Type from Filename
    ====================================

    Analyzes the filename structure to determine the energy commodity type
    (Electricity, Gas, Dual, or Other) based on naming conventions.

    Args:
        file_name (str): PDF filename to analyze

    Returns:
        str: Commodity type ("Luce", "Gas", "Dual", "-", or "ERROR")
    """
    # Remove file extension and split filename into parts
    base_filename = file_name.removesuffix(".pdf")
    filename_parts = base_filename.split("_")

    # Return error if filename has no parts
    if not filename_parts:
        return "ERROR"

    # Analyze filename based on document type prefix
    document_type_prefix = filename_parts[0]

    match document_type_prefix:
        case "CE":  # Condizioni Economiche (Economic Conditions)
            if len(filename_parts) >= 4:
                return "Dual"  # CE files with 4+ parts are dual commodity

            if len(filename_parts) < 2:
                return "-"  # Insufficient parts to determine commodity

            # Analyze second part of filename
            commodity_indicator = filename_parts[1]

            match commodity_indicator:
                case "TRVULN":  # Special gas tariff indicator
                    return "Gas"
                case indicator if indicator.startswith("G") or (
                    len(indicator) > 4 and indicator[4] == "G"
                ):
                    return "Gas"  # Gas indicators
                case indicator if indicator.startswith("L") or (
                    len(indicator) > 4 and indicator[4] == "L"
                ):
                    return "Luce"  # Electricity indicators (Luce = Light/Electricity)
                case _:
                    return "-"  # Unknown commodity indicator

        case "CGC":  # Condizioni Generali di Contratto (General Contract Conditions)
            return "-"  # CGC documents are commodity-agnostic

        case "SR":  # Scheda di Confrontabilità (Comparison Sheet)
            return "Luce"  # SR documents are electricity-only

        case _:
            return "ERROR"  # Unknown document type


def extract_text_from_pdf_plumber(pdf_path, table_filter=False):
    """
    Extract Text from PDF using PDFPlumber
    ====================================

    Extracts and cleans text from PDF files, with optional table filtering
    and automatic removal of page number lines.

    Args:
        pdf_path (str): Path to the PDF file
        table_filter (bool): Whether to filter out table content

    Returns:
        str: Extracted and cleaned text content, or error message
    """

    def is_page_number_line(text_line):
        """
        Check if Text Line is a Page Number
        =================================

        Identifies lines that contain only page numbers in format "X/Y".
        """
        return bool(re.search(r"\b\d+\s*/\s*\d+\b", text_line.strip()))

    try:
        # Open PDF file using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            all_page_texts = []

            # Process each page in the PDF
            for page in pdf.pages:
                if table_filter:
                    # Apply table filtering if requested
                    filtered_page = filter_tables_pdfplumber_extraction(page)
                    page_text = (
                        filtered_page.extract_text(
                            keep_blank_chars=True,
                            layout=True,
                            x_tolerance=1,
                            y_tolerance=3,
                        )
                        if filtered_page
                        else ""
                    )
                else:
                    # Extract text without table filtering
                    page_text = (
                        page.extract_text(
                            keep_blank_chars=True,
                            layout=True,
                            x_tolerance=1,
                            y_tolerance=3,
                        )
                        or ""
                    )

                # Split page text into lines for cleaning
                text_lines = page_text.splitlines()

                # Remove second-to-last line if it's a page number line
                last_five_indices = range(max(0, len(text_lines) - 5), len(text_lines))
                cleaned_lines = [
                    line
                    for i, line in enumerate(text_lines)
                    if i not in last_five_indices or not is_page_number_line(line)
                ]

                # Rejoin cleaned lines and add to collection
                cleaned_page_text = "\n".join(cleaned_lines)
                all_page_texts.append(cleaned_page_text)

            # Combine all page texts into single document text
            return "\n".join(all_page_texts)

    except Exception as extraction_error:
        return f"Error reading {pdf_path}: {extraction_error}"


def extract_pc_from_plichi_plumber(pdf_path):
    """
    Extract Text from PDF using PDFPlumber
    ====================================

    Extracts and cleans text from PDF files, with optional table filtering
    and automatic removal of page number lines.

    Args:
        pdf_path (str): Path to the PDF file
        table_filter (bool): Whether to filter out table content

    Returns:
        str: Extracted and cleaned text content, or error message
    """

    def is_page_number_line(text_line):
        """
        Detect simple page number pattern like '1/3' anywhere in the line.
        """
        return bool(re.search(r"\b\d+\s*/\s*\d+\b", text_line.strip()))

    try:
        keywords = [
            "Copia Cliente",
            "Copia Fornitore",
            "CONDIZIONI CONTRATTUALI DELL'ATTIVAZIONE",
        ]
        pages_text = []

        # Open PDF file using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # Process each page in the PDF
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text(
                    keep_blank_chars=True,
                    layout=True,
                    x_tolerance=1,
                    y_tolerance=3,
                )

                if page_text:
                    lines = page_text.splitlines()
                    first_lines = "\n".join(
                        lines[:10]
                    )  # Check first 10 lines for keywords

                    if any(keyword in first_lines for keyword in keywords):
                        # Remove page numbers in the last 5 lines
                        last_five_indices = range(max(0, len(lines) - 5), len(lines))
                        cleaned_lines = [
                            line
                            for i, line in enumerate(lines)
                            if i not in last_five_indices
                            or not is_page_number_line(line)
                        ]
                        cleaned_text = "\n".join(cleaned_lines)
                        pages_text.append(cleaned_text)

            return "\n".join(pages_text) if pages_text else "No matching content found."

    except Exception as extraction_error:
        return f"Error reading {pdf_path}: {extraction_error}"


def extract_text_from_pdf_pypdf2(pdf_path):
    """
    Extract Text from PDF using PyPDF2
    =================================

    Alternative PDF text extraction method using PyPDF2 library.
    Note: PyPDF2 import is commented out in the current implementation.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content, or error message
    """
    try:
        # Open PDF file in binary mode
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)

            # Extract text from all pages and combine
            extracted_text = "".join(
                page.extract_text() or "" for page in pdf_reader.pages
            )

        return extracted_text

    except Exception as extraction_error:
        return f"Error reading {pdf_path}: {extraction_error}"


def chunk_text_by_ordered_markers(text, markers):
    """
    Split Text into Chunks Based on Ordered Markers
    ==============================================

    Divides document text into logical sections based on predefined markers,
    maintaining the order of markers as they appear in the text.

    Args:
        text (str): Full document text to be chunked
        markers (list): List of marker strings or marker lists to search for

    Returns:
        list: List of [marker_label, chunk_text] pairs
    """
    marker_positions = []
    search_start_position = 0

    # Find positions of all markers in the text
    for marker in markers:
        if isinstance(marker, list):
            # Handle multiple marker variants (alternative phrasings)
            variant_pattern = "|".join(
                re.escape(normalize_apostrophes(variant)).replace(r"\ ", r"\s*")
                for variant in marker
            )
            regex_pattern = rf"(?:{variant_pattern})"
            regex_pattern = (
                rf"\s+{regex_pattern}\s+"  # Allow whitespace around the marker
            )
            marker_label = marker[0]  # Use first variant as label
        else:
            marker = normalize_apostrophes(marker)
            # Handle single marker string
            marker_label = marker
            # Adjust the regex to handle exact word matching without lookbehind
            regex_pattern = re.escape(marker).replace(r"\ ", r"\s*")
            # Match exact marker words, ensuring it's a full match between spaces
            regex_pattern = rf"(?:\s|^){regex_pattern}(?:\s|$)"

        # Search for marker starting from current position
        marker_match = re.search(regex_pattern, text[search_start_position:])
        if marker_match:
            # Calculate absolute position in full text
            absolute_position = search_start_position + marker_match.start()
            marker_positions.append((absolute_position, marker_label))

            # Update search start to avoid overlapping matches
            search_start_position = absolute_position + len(marker_match.group(0))

    # Return entire text as single chunk if no markers found
    if not marker_positions:
        return [[None, text]]
    # Add end position for final chunk
    marker_positions.append((len(text), None))

    # Create chunks between consecutive markers
    text_chunks = []
    for i in range(len(marker_positions) - 1):
        chunk_start_pos = marker_positions[i][0]
        chunk_end_pos = marker_positions[i + 1][0]

        # Extract text between markers
        chunk_text = text[chunk_start_pos:chunk_end_pos]
        chunk_marker = marker_positions[i][1]

        text_chunks.append([chunk_marker, chunk_text])

    return text_chunks


def chunk_text_by_ordered_markers_2(chunked_text_and_title: list, markers) -> list:
    """
    Return list of [marker_with_number, chunk_text] pairs.
    """
    if chunked_text_and_title is None:
        return [[None, ""]]
    compiled = []
    for marker in markers:
        if isinstance(marker, list):
            normalized_variants = [
                re.escape(normalize_apostrophes(variant)).replace(r"\ ", r"\s+")
                for variant in marker
            ]
            variant_pattern = "|".join(normalized_variants)
            regex_pattern = rf"(?:{variant_pattern})"
            label = normalize_apostrophes(marker[0])
        else:
            base = normalize_apostrophes(marker)
            regex_pattern = re.escape(base).replace(r"\ ", r"\s+")
            label = base

        compiled.append((re.compile(regex_pattern), label))
    compiled_patterns = compiled
    chunks = []
    for i in chunked_text_and_title:
        title, text = i
        # Find all occurrences: store (start, end, label)
        positions = []
        for patt, label in compiled_patterns:
            for m in patt.finditer(text):
                positions.append((m.start(), m.end(), label))

        positions.sort(key=lambda x: x[0])

        if not positions:
            # No markers: return whole text once
            return [[None, text]]

        # Sentinel at end of text for final slice
        positions.append((len(text), len(text), None))

        counts = {}

        for i in range(len(positions) - 1):
            start_i, end_i, label_i = positions[i]
            start_next, _, _ = positions[i + 1]

            counts[label_i] = counts.get(label_i, 0) + 1
            if counts[label_i] == 1:
                numbered_label = f"{title} - {label_i}"
            else:
                numbered_label = f"{title} - {label_i} - {counts[label_i]}"

                # Slice after marker up to next marker
            chunk_text = text[end_i:start_next]
            chunks.append([numbered_label, chunk_text])

    return chunks


def build_schema(col_dict, extra_col=None):
    """
    Build Spark DataFrame Schema from Configuration
    =============================================

    Creates a Spark StructType schema from a dictionary of column definitions,
    with support for special data types and optional extra columns.

    Args:
        col_dict (dict): Dictionary mapping column names to type strings
        extra_col (tuple, optional): Additional column as (name, type) tuple

    Returns:
        pyspark.sql.types.StructType: Complete schema for DataFrame creation
    """
    # Standard type mapping for basic data types
    type_mapping = {
        "string": T.StringType(),
        "integer": T.IntegerType(),
        "boolean": T.BooleanType(),
        "date": T.DateType(),
        "map": T.ArrayType(T.StringType()),
    }

    # Special type definitions for complex fields
    special_types = {
        "corresponding_contractual_paragraphs": T.ArrayType(
            T.ArrayType(T.StringType())
        ),  # Array of arrays of strings
        "comparison_lists": T.ArrayType(T.IntegerType()),  # Array of integers
        "num_paragraph": T.ArrayType(T.IntegerType()),  # Array of integers
        "list_of_differences": T.ArrayType(T.StringType()),  # Array of strings
    }

    # Build schema fields from column dictionary
    schema_fields = []
    for column_name, type_string in col_dict.items():
        # Use special type if defined, otherwise use standard mapping
        data_type = special_types.get(
            column_name, type_mapping.get(type_string, T.StringType())
        )
        schema_fields.append(T.StructField(column_name, data_type))

    # Add extra column if provided
    if extra_col is not None:
        extra_column_name, extra_type = extra_col

        # Handle both string type names and DataType objects
        if isinstance(extra_type, str):
            extra_data_type = special_types.get(
                extra_column_name, type_mapping.get(extra_type, T.StringType())
            )
        else:
            extra_data_type = extra_type

        schema_fields.append(T.StructField(extra_column_name, extra_data_type))

    return T.StructType(schema_fields)


def normalize_apostrophes(text: str) -> str:
    """
    Replace curly/typographic apostrophes with a straight ASCII apostrophe.
    """
    if not text:
        return text

    replacements = {
        "ʼ": "'",  # modifier letter apostrophe (U+02BC)
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark
        "‚": "'",  # single low-9 quotation mark
        "‛": "'",  # reversed single quotation mark
    }

    for unicode_char, ASCII in replacements.items():
        text = text.replace(unicode_char, ASCII)
    return text


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 300,
    jitter: bool = False,
    max_retries: int = 2,
):
    """Retry a function with exponential backoff."""

    num_retries = 0
    delay = initial_delay

    while num_retries < max_retries:
        try:
            return func()
        except Exception as e:
            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                num_retries += 1
                print(num_retries)

                if jitter:  # # nosec
                    delay *= exponential_base * (  # nosec
                        1 + random.random()  # nosec
                    )  # # nosec
                else:
                    delay *= exponential_base  # nosec

                time.sleep(delay)
            else:
                raise e

    raise Exception(f"Maximum retries {max_retries} exceeded")
