# MAGIC %md
# MAGIC Text Extraction and Master Table Creation Module
# MAGIC ## This module provides functionality to extract text from PDF documents and create master tables with chunked content for document comparison.
from collections import Counter
import os
from pyspark.sql import SparkSession, Row
from src.utils.utils import (
    chunk_text_by_ordered_markers,
    chunk_text_by_ordered_markers_2,
    extract_text_from_pdf_plumber,
    extract_pc_from_plichi_plumber,
    normalize_apostrophes,
)

# Initialize Spark session
spark: SparkSession = SparkSession.builder.getOrCreate()


class TextExtractor:
    """
    Text Extraction and Master Table Creation Class
    ==============================================

    A class to handle text extraction from PDF documents and master table creation.
    This class processes document pairs (current vs past campaigns) and extracts
    text content organized by predefined chunk markers.

    Attributes:
        volume_directory (str): Base directory path for document volumes
    """

    def __init__(self, volume_directory):
        """
        Initialize TextExtractor with volume directory path.

        Args:
            volume_directory (str): Base directory path for document volumes
        """
        self.volume_directory = volume_directory

    def build_file_path(self, campaign, macrosegmento, document_type, file_name):
        """
        Build File Path Based on Campaign Structure
        ==========================================

        Constructs the complete file path based on campaign folder structure
        for different document types (CE, CGC, SR, SS, PLICHI) and market segments.
        """

        base_path = os.path.join(
            self.volume_directory,
            campaign,
            "PLICHI",
            macrosegmento,
        )

        # Common macrosegment mappings
        ce_map = {
            "RESIDENZIALI": "STANDARD_RES",
            "MICROBUSINESS": "STANDARD_BUS",
            "AUP": "AUP",
        }

        cgc_map = {
            "RESIDENZIALI": "RES",
            "MICROBUSINESS": "BUS",
            "AUP": "AUP",
        }

        ss_gas_map = {
            "RESIDENZIALI": "GAS_RES",
            "MICROBUSINESS": "GAS_BUS",
            "AUP": "AUP",
        }

        ss_luce_map = {
            "RESIDENZIALI": "LUCE_RES",
            "MICROBUSINESS": "LUCE_BUS",
            "AUP": "AUP",
        }

        if document_type == "CE":
            suffix = f"CE_{ce_map[macrosegmento]}"

        elif document_type == "CGC":
            suffix = f"CGC_{cgc_map[macrosegmento]}"

        elif document_type == "SR":
            # SR only for residential electricity
            suffix = "SR_LUCE_RES"

        elif document_type == "SS_gas":
            suffix = f"SS_{ss_gas_map[macrosegmento]}"

        elif document_type == "SS_luce":
            suffix = f"SS_{ss_luce_map[macrosegmento]}"

        elif document_type == "PLICHI":
            if macrosegmento == "AUP":
                suffix = "PLICHI_AUP"
            else:
                prefix = file_name.split("_")[0]
                segment = "RES" if macrosegmento == "RESIDENZIALI" else "BUS"
                suffix = f"PLICHI_{prefix}_{segment}"

        else:
            raise ValueError(f"Unknown document type: {document_type}")

        return os.path.join(base_path, suffix, file_name)

    def _prefix_dual_chunks(self, chunks):
        """
        For dual commodity documents, prefix duplicate chunk titles with their
        commodity section so they become unique keys.

        E.g. two "Sconti e/o Bonus" become:
          - "Energia Elettrica - Sconti e/o Bonus"
          - "Gas Naturale - Sconti e/o Bonus"
        """
        title_counts = Counter(title for title, _ in chunks)
        duplicates = {title for title, count in title_counts.items() if count > 1}

        if not duplicates:
            return chunks

        current_section = ""
        prefixed = []
        for title, text in chunks:
            if title.strip() == "Energia Elettrica":
                current_section = "Energia Elettrica"
            elif title.strip() == "Gas Naturale":
                current_section = "Gas Naturale"

            if title in duplicates and current_section:
                prefixed.append((f"{current_section} - {title}", text))
            else:
                prefixed.append((title, text))

        return prefixed

    def create_master(
        self,
        df,
        chunk_markers_lvl1,
        chunk_markers_lvl2=None,
        chunk_markers_dual_lvl1=None,
    ):
        """
        Create Master Table with Extracted and Chunked Text
        ==================================================

        Processes pairs of documents (current vs past campaign), extracts text content,
        and chunks it based on predefined markers for document comparison analysis.

        Args:
            df: Spark DataFrame containing document metadata with columns:
                - current_campaign: Current campaign identifier
                - past_campaign: Past campaign identifier
                - macrosegmento: Market segment
                - document_type: Type of document (CE, CGC, SR)
                - current_file_name: Current campaign file name
                - past_file_name: Past campaign file name
                - commodity: Energy commodity type (Luce, Gas, Dual)
            chunk_markers_lvl1 (list): List of markers to chunk regular documents
            chunk_markers_dual_lvl1 (list, optional): List of markers for dual commodity documents

        Returns:
            DataFrame: Spark DataFrame with chunked text data containing:
                - Original metadata columns
                - chunk: Chunk index number
                - chunk_title: Title/marker of the chunk
                - extracted_text: Combined text from both documents
                - extracted_chunk: Combined chunk text from both documents
                - chunk_check: Boolean indicating if chunk exists in both documents
        """
        all_rows = []

        # Process each document pair in the DataFrame
        # Handle SS subtype based on commodity
        for i in df.collect():
            document_type = i["document_type"]
            if document_type == "SS":
                if i["commodity"] == "Luce":
                    document_type = "SS_luce"
                elif i["commodity"] == "Gas":
                    document_type = "SS_gas"
                else:
                    raise ValueError(
                        f"Unknown commodity type for SS document: {i['commodity']}"
                    )
            # Build complete file paths for current and past documents
            current_file_path = self.build_file_path(
                i["current_campaign"],
                i["macrosegmento"],
                document_type,
                i["current_file_name"],
            )
            past_file_path = self.build_file_path(
                i["past_campaign"],
                i["macrosegmento"],
                document_type,
                i["past_file_name"],
            )
            if document_type == "PLICHI":
                document_type = "PC"
                # Chunk CE and CGC documents based on predefined markers
                past_text = extract_pc_from_plichi_plumber(past_file_path)
                current_text = extract_pc_from_plichi_plumber(current_file_path)
            else:
                # Extract text content from PDF files
                past_text = extract_text_from_pdf_plumber(past_file_path)

                current_text = extract_text_from_pdf_plumber(current_file_path)

            # Combine texts for comparison analysis
            combined_text = f"CURRENT_DOCUMENT: {normalize_apostrophes(current_text)} \n PAST_DOCUMENT: {normalize_apostrophes(past_text)}"

            # Handle PDF extraction errors
            if "Error reading" in current_text or "Error reading" in past_text:
                # Create error row when PDF extraction fails
                row_dict = i.asDict()
                all_rows.append(
                    Row(
                        **{
                            **{
                                k: str(v) if v is not None else ""
                                for k, v in row_dict.items()
                            },
                            "chunk": -1,  # Error indicator
                            "chunk_title": "ERROR",
                            "extracted_text": f"ERROR - CURRENT_DOCUMENT: {current_text} \n PAST_DOCUMENT: {past_text}",
                            "chunk_check": False,
                        }
                    )
                )
                continue

            # Choose appropriate chunk markers based on commodity type
            if chunk_markers_lvl2:
                current_chunks = chunk_text_by_ordered_markers_2(
                    chunk_text_by_ordered_markers(current_text, chunk_markers_lvl1),
                    chunk_markers_lvl2,
                )
                past_chunks = chunk_text_by_ordered_markers_2(
                    chunk_text_by_ordered_markers(past_text, chunk_markers_lvl1),
                    chunk_markers_lvl2,
                )

            elif i["commodity"] == "Dual" and i["document_type"] == "CE":
                # Use dual commodity markers for documents containing both electricity and gas
                current_chunks = chunk_text_by_ordered_markers(
                    normalize_apostrophes(current_text), chunk_markers_dual_lvl1
                )
                # Prefix duplicate titles with their commodity section
                current_chunks = self._prefix_dual_chunks(current_chunks)
                past_chunks = chunk_text_by_ordered_markers(
                    normalize_apostrophes(past_text), chunk_markers_dual_lvl1
                )
                past_chunks = self._prefix_dual_chunks(past_chunks)
            else:
                # Use standard markers for single commodity documents
                current_chunks = chunk_text_by_ordered_markers(
                    normalize_apostrophes(current_text), chunk_markers_lvl1
                )
                past_chunks = chunk_text_by_ordered_markers(
                    normalize_apostrophes(past_text), chunk_markers_lvl1
                )

            # Create dictionaries for efficient chunk mapping
            current_dict = {title: part for title, part in current_chunks}
            past_dict = {title: part for title, part in past_chunks}

            # Get all unique chunk titles from both documents
            all_titles = set(current_dict.keys()).union(past_dict.keys())

            # Process each chunk and create rows for master table
            for idx, title in enumerate(sorted(all_titles)):
                # Get chunk content from both documents (empty string if not found)
                current_part = current_dict.get(title, "")
                past_part = past_dict.get(title, "")

                # Check if chunk exists in both documents for validation
                chunk_check = title in current_dict and title in past_dict

                # Combine chunk content for comparison
                combined_chunk = (
                    f"CURRENT_DOCUMENT: {current_part} \n PAST_DOCUMENT: {past_part}"
                )

                # Create row with original metadata plus chunk information
                row_dict = i.asDict()
                all_rows.append(
                    Row(
                        **{
                            **{
                                k: str(v) if v is not None else ""
                                for k, v in row_dict.items()
                            },
                            "chunk": int(idx),  # Sequential chunk number
                            "chunk_title": str(title),  # Chunk marker/title
                            "extracted_text": str(combined_text),  # Full document text
                            "extracted_chunk": str(
                                combined_chunk
                            ),  # Specific chunk text
                            "chunk_check": bool(chunk_check),  # Validation flag
                        }
                    )
                )

        # Return Spark DataFrame with all processed rows
        return spark.createDataFrame(all_rows)
