"""
Master Ingestion Module for Document Correspondence
==================================================
This module provides functionality to find and match corresponding documents
between current and past marketing campaigns across different market segments.

Functions:
    normalize_name: Normalize filename by removing numeric parts
    is_target_file: Check if filename matches target pattern
    get_corrisponding_files: Find corresponding files between campaigns
    get_commodity: Determine commodity type from filename
"""

import os
import re


def normalize_name(filename):
    """
    Normalize Filename by Removing Numeric Parts
    ===========================================

    Removes file extension and numeric characters from filename to enable
    matching between files that may have different version numbers or dates.

    Args:
        filename (str): Original filename to normalize

    Returns:
        str: Normalized filename without extension and numeric characters

    Example:
        normalize_name("CE_LUCE_123.pdf") -> "CE_LUCE_"
    """
    # Remove file extension (.pdf, .docx, etc.)
    name, _ = os.path.splitext(filename)

    # Remove all numeric characters to allow version matching
    return re.sub(r"\d+", "", name)


def is_target_file(filename):
    """
    Check if Filename Matches CE Document Pattern
    ============================================

    Validates if a filename follows the expected pattern for CE
    (Condizioni Economiche) documents.

    Args:
        filename (str): Filename to validate

    Returns:
        bool: True if filename matches CE pattern, False otherwise

    Pattern:
        CE_[IDENTIFIER]_[OPTIONAL_PARTS].pdf
    """
    # Pattern for CE documents: starts with "CE_", contains word characters and underscores, ends with ".pdf"
    return re.match(r"^CE_[\w_]+\.pdf$", filename)


def get_corrisponding_plichi(
    macrosegmenti: list,
    file_type: str,
    currentcampaign_paths: list,
    pastcampaign_paths: list,
    debug: bool = False,
):
    rows = []

    # Create regex pattern for folder matching based on document type
    foldername_pattern = re.compile(
        r"^PLICHI_(?:ATTCART_)?(?:CPCART_)?(?:SWICART_)?(?:SWIVLTCART_)?(?:VLTCART_)?(AUP|BUS|RES)$"
    )

    def log(*args):
        """
        Internal Debug Logging Function
        ==============================

        Prints debug messages only when debug mode is enabled.
        """
        if debug:
            print(*args)

    # Process each current campaign path
    for current_path in currentcampaign_paths:
        log(f"Current campaign path: {current_path}")

        # Process each market segment
        for segmento in macrosegmenti:
            # Build path to segment folder within campaign
            current_segment_path = os.path.join(current_path, "PLICHI", segmento)
            log(f"Checking segment path: {current_segment_path}")

            # Skip if segment directory doesn't exist
            if not os.path.isdir(current_segment_path):
                continue

            # Find folders matching the document type pattern
            foldernames = [
                f
                for f in os.listdir(current_segment_path)
                if foldername_pattern.match(f)
            ]
            log(f"Matched folders: {foldernames}")

            # Process each matching folder
            for folder in foldernames:
                current_folder = os.path.join(current_segment_path, folder)

                # Skip if not a directory
                if not os.path.isdir(current_folder):
                    continue

                files = os.listdir(current_folder)

                gas_file = next(
                    (f for f in files if "_GAS_" in f and f.endswith(".pdf")), None
                )
                log(f"Gas file: {gas_file}")
                power_file = next(
                    (f for f in files if "_POWER_" in f and f.endswith(".pdf")), None
                )
                log(f"Power file: {power_file}")
                dual_file = next(
                    (f for f in files if "_DUAL_" in f and f.endswith(".pdf")), None
                )
                log(f"Dual file: {dual_file}")
                aup_file = next(
                    (f for f in files if "_AUP_SMALL_" in f and f.endswith(".pdf")),
                    None,
                )
                log(f"AUP file: {aup_file}")

                current_files = {}

                if gas_file:
                    current_files[normalize_name(gas_file)] = gas_file
                if power_file:
                    current_files[normalize_name(power_file)] = power_file
                if dual_file:
                    current_files[normalize_name(dual_file)] = dual_file
                if aup_file:
                    current_files[normalize_name(aup_file)] = aup_file

                # Process each past campaign path
                for past_path in pastcampaign_paths:
                    log(f"Past campaign path: {past_path}")

                    # Build corresponding past campaign folder path
                    past_folder = os.path.join(past_path, "PLICHI", segmento, folder)

                    # Skip if past folder doesn't exist
                    if not os.path.isdir(past_folder):
                        continue

                    # Create normalized filename mapping for past campaign files
                    past_files = {normalize_name(f): f for f in os.listdir(past_folder)}

                    # Find files that exist in both current and past campaigns
                    # Create correspondence records for matching files
                    for base_name in sorted(current_files.keys() & past_files.keys()):
                        current_filename = current_files[base_name]
                        past_filename = past_files[base_name]

                        # Determine commodity type based on segment and filename
                        commodity_type = (
                            "-"
                            if "_AUP_SMALL_" in current_files.keys()
                            else "Gas"
                            if "_GAS_" in current_filename
                            else "Luce"
                            if "_POWER_" in current_filename
                            else "Dual"
                            if "_DUAL_" in current_filename
                            else "-"
                        )

                        # Create correspondence record
                        correspondence_record = {
                            "macrosegmento": segmento,
                            "document_type": file_type,
                            "current_campaign": os.path.basename(current_path),
                            "past_campaign": os.path.basename(past_path),
                            "current_file_name": current_filename,
                            "past_file_name": past_filename,
                            "commodity": commodity_type,
                        }

                        rows.append(correspondence_record)

    return rows


def get_corrisponding_files(
    macrosegmenti: list,
    file_type: str,
    currentcampaign_paths: list,
    pastcampaign_paths: list,
    debug: bool = False,
):
    """
    Find Corresponding Files Between Current and Past Campaigns
    =========================================================

    Searches through campaign folders to find matching documents between
    current and past campaigns across different market segments and document types.

    Args:
        macrosegmenti (list): List of market segments to process
                             (e.g., ["RESIDENZIALI", "MICROBUSINESS", "AUP"])
        file_type (str): Type of documents to match (CE, CGC, SR)
        currentcampaign_paths (list): List of paths to current campaign folders
        pastcampaign_paths (list): List of paths to past campaign folders
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        list: List of dictionaries containing file correspondence information:
            - macrosegmento: Market segment
            - document_type: Type of document (CE, CGC, SR, SS)
            - current_campaign: Current campaign folder name
            - past_campaign: Past campaign folder name
            - current_file_name: Current campaign file name
            - past_file_name: Past campaign file name
            - commodity: Energy commodity type (Luce, Gas, Dual, -)
    """
    # Initialize list to store correspondence results
    rows = []

    # Create regex pattern for folder matching based on document type
    foldername_pattern = re.compile(
        rf"^{re.escape(file_type)}_(?:STANDARD_)?(?:LUCE_)?(?:GAS_)?(AUP|BUS|RES)$"
    )

    def log(*args):
        """
        Internal Debug Logging Function
        ==============================

        Prints debug messages only when debug mode is enabled.
        """
        if debug:
            print(*args)

    # Process each current campaign path
    for current_path in currentcampaign_paths:
        log(f"Current campaign path: {current_path}")

        # Process each market segment
        for segmento in macrosegmenti:
            # Build path to segment folder within campaign
            current_segment_path = os.path.join(current_path, "PLICHI", segmento)
            log(f"Checking segment path: {current_segment_path}")

            # Skip if segment directory doesn't exist
            if not os.path.isdir(current_segment_path):
                continue

            # Find folders matching the document type pattern
            foldernames = [
                f
                for f in os.listdir(current_segment_path)
                if foldername_pattern.match(f)
            ]
            log(f"Matched folders: {foldernames}")

            # Process each matching folder
            for folder in foldernames:
                current_folder = os.path.join(current_segment_path, folder)

                # Skip if not a directory
                if not os.path.isdir(current_folder):
                    continue

                # Create normalized filename mapping for current campaign files
                current_files = {
                    normalize_name(f): f for f in os.listdir(current_folder)
                }

                # Process each past campaign path
                for past_path in pastcampaign_paths:
                    log(f"Past campaign path: {past_path}")

                    # Build corresponding past campaign folder path
                    past_folder = os.path.join(past_path, "PLICHI", segmento, folder)

                    # Skip if past folder doesn't exist
                    if not os.path.isdir(past_folder):
                        continue

                    # Create normalized filename mapping for past campaign files
                    past_files = {normalize_name(f): f for f in os.listdir(past_folder)}

                    # Find files that exist in both current and past campaigns
                    # Create correspondence records for matching files
                    for base_name in sorted(current_files.keys() & past_files.keys()):
                        current_filename = current_files[base_name]
                        past_filename = past_files[base_name]

                        # Determine commodity type based on segment and filename
                        commodity_type = (
                            "-"
                            if segmento == "AUP"
                            else get_commodity(current_filename)
                        )

                        # Create correspondence record
                        correspondence_record = {
                            "macrosegmento": segmento,
                            "document_type": file_type,
                            "current_campaign": os.path.basename(current_path),
                            "past_campaign": os.path.basename(past_path),
                            "current_file_name": current_filename,
                            "past_file_name": past_filename,
                            "commodity": commodity_type,
                        }

                        rows.append(correspondence_record)

    return rows


def get_commodity(file_name: str) -> str:
    """
    Determine Energy Commodity Type from Filename
    ============================================

    Analyzes the filename structure to determine the energy commodity type
    (Electricity, Gas, Dual commodity, or Other) based on standard naming conventions.

    Args:
        file_name (str): PDF filename to analyze

    Returns:
        str: Commodity type classification:
            - "Luce": Electricity documents
            - "Gas": Natural gas documents
            - "Dual": Documents covering both electricity and gas
            - "-": Commodity-agnostic documents
            - "ERROR": Invalid or unrecognized filename format

    Filename Patterns:
        CE documents:
            - CE_L*: Electricity (Luce)
            - CE_G*: Gas
            - CE_TRVULN: Special gas tariff
            - CE_*_*_*: Dual commodity (4+ parts)
        CGC documents: Always commodity-agnostic (-)
        SR documents: Always electricity (Luce)
        S documents: Special handling with 4th part analysis
    """
    # Remove file extension for processing
    name = file_name.removesuffix(".pdf")
    parts = name.split("_")

    # Return error if filename has no parts
    if not parts:
        return "ERROR"

    # Analyze filename based on document type prefix
    match parts[0]:
        case "CE":  # Condizioni Economiche (Economic Conditions)
            # Dual commodity detection: CE files with 4+ parts are dual commodity
            if len(parts) >= 4:
                return "Dual"

            # Insufficient parts to determine commodity
            if len(parts) < 2:
                return "-"

            # Analyze commodity indicator (second part of filename)
            part = parts[1]
            match part:
                case "TRVULN":  # Special vulnerable customer gas tariff
                    return "Gas"
                case p if p.startswith("G") or (len(p) > 4 and p[4] == "G"):
                    # Gas indicators: starts with "G" or has "G" at position 4 (TNDVG)
                    return "Gas"
                case p if p.startswith("L") or (len(p) > 4 and p[4] == "L"):
                    # Electricity indicators: starts with "L" or has "L" at position 4 (TNDVL)
                    return "Luce"
                case _:
                    # Unknown or unclassifiable commodity indicator
                    return "-"

        case "CGC":  # Condizioni Generali di Contratto (General Contract Conditions)
            # CGC documents are commodity-agnostic
            return "-"

        case "SR":  # Scheda di Confrontabilità (Comparison Sheet)
            # SR documents are specific to electricity only
            return "Luce"

        case "S":  # Special document type with 4th part analysis
            # Analyze commodity indicator from 4th part of filename
            if len(parts) < 4:
                return "-"

            part = parts[3]
            match part:
                case p if p[0:6] == "TRVULN":  # Special vulnerable customer gas tariff
                    return "Gas"
                case p if p.startswith("G") or ("G" in (p[3:5] if len(p) > 4 else "")):
                    # Gas indicators: starts with "G" or has "G" at position 4 (TNDVG)
                    return "Gas"
                case p if p.startswith("L") or ("L" in (p[3:5] if len(p) > 4 else "")):
                    # Electricity indicators: starts with "L" or has "L" at position 4 (TNDVL)
                    return "Luce"
                case _:
                    # Unknown or unclassifiable commodity indicator
                    return "-"

        case _:
            # Unknown or unsupported document type
            return "ERROR"
