import pytest
import pandas as pd
import numpy as np
import io
import tarfile
from fastapi import HTTPException
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.parse import parse_data_for_model, extract_csv_from_tar_gz_bytes


class TestParseDataForModel:
    """Tests for the parse_data_for_model function."""

    def test_basic_parsing(self):
        """Test basic parsing functionality with a small CSV"""
        csv_data = io.StringIO(
            "col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13\n"
            "1,2,3,4,5,6,7,8,9,10,11,12,13\n"
            "14,15,16,17,18,19,20,21,22,23,24,25,26\n"
        )

        result = parse_data_for_model(csv_data)

        assert len(result) == 2
        assert result[0].shape == (1, 13)  
        # First row (index 0): [1,2,3,4,5,6,7,8,9,10,11,12,13]
        assert np.array_equal(result[0], np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13]]))
        # Second row (index 1): [14,15,16,17,18,19,20,21,22,23,24,25,26]
        assert np.array_equal(result[1], np.array([[14,15,16,17,18,19,20,21,22,23,24,25,26]]))

    def test_parse_data_for_model_row_limit(self):
        """Test that processing is limited to 30 rows"""
        rows = ["col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13"]
        for i in range(40):  # Add 40 data rows
            rows.append(f"{i},{i+1},{i+2},{i+3},{i+4},{i+5},{i+6},{i+7},{i+8},{i+9},{i+10},{i+11},{i+12}")
        
        csv_data = io.StringIO("\n".join(rows))
        
        result = parse_data_for_model(csv_data)
        
        assert len(result) == 30

    def test_parse_data_for_model_column_selection(self):
        """Don't skip columns"""
        # Create a CSV with many columns
        csv_data = io.StringIO(
            "col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15\n"
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
            "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30\n"
        )
        
        # Call the function
        result = parse_data_for_model(csv_data)
        
        assert result[0].shape == (1, 15)  
        # First row (index 0): [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        assert np.array_equal(result[0], np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]))

    def test_parse_data_for_model_error_handling(self):
        """Test error handling with invalid CSV data"""
        # Invalid CSV with mismatched columns
        csv_data = io.StringIO("header1,header2\nvalue1\nvalue2,value3,extra")
        
        # Call should raise HTTPException
        with pytest.raises(HTTPException) as excinfo:
            parse_data_for_model(csv_data)
        
        # Verify exception details
        assert excinfo.value.status_code == 400
        assert "Error parsing data" in excinfo.value.detail


class TestExtractCsvFromTarGzBytes:
    """Tests for the extract_csv_from_tar_gz_bytes function."""

    @pytest.mark.asyncio
    async def test_valid_extraction(self):
        """Test successful extraction of a CSV from tar.gz bytes"""
        # Create a tar.gz file in memory with a CSV file
        csv_content = "col1,col2\n1,2\n3,4"
        tar_io = io.BytesIO()
        
        with tarfile.open(fileobj=tar_io, mode='w:gz') as tar:
            tarinfo = tarfile.TarInfo("test.csv")
            csv_bytes = csv_content.encode('utf-8')
            tarinfo.size = len(csv_bytes)
            tar.addfile(tarinfo, io.BytesIO(csv_bytes))
        
        tar_io.seek(0)
        tar_gz_bytes = tar_io.getvalue()
        
        # Call the function
        result = await extract_csv_from_tar_gz_bytes(tar_gz_bytes)
        
        # Verify result
        assert result == csv_content

    @pytest.mark.asyncio
    async def test_extract_csv_from_tar_gz_bytes_no_csv(self):
        """Test extraction when no CSV file is present"""
        # Create a tar.gz file with only non-CSV files
        tar_io = io.BytesIO()
        
        with tarfile.open(fileobj=tar_io, mode='w:gz') as tar:
            tarinfo = tarfile.TarInfo("test.txt")
            content = "Not a CSV file".encode('utf-8')
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))
        
        tar_io.seek(0)
        tar_gz_bytes = tar_io.getvalue()
        
        # Call the function
        result = await extract_csv_from_tar_gz_bytes(tar_gz_bytes)
        
        # Should return None when no CSV file is found
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_csv_from_tar_gz_bytes_no_data(self):
        """Test extraction with no input data"""
        result = await extract_csv_from_tar_gz_bytes(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_csv_from_tar_gz_bytes_invalid_tar(self):
        """Test extraction with invalid tar.gz data"""
        # Call with invalid bytes
        result = await extract_csv_from_tar_gz_bytes(b"not a valid tar.gz file")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_csv_from_tar_gz_bytes_encoding_error(self):
        """Test extraction with CSV that has encoding problems"""
        # Create a tar.gz with a CSV containing non-UTF-8 bytes
        tar_io = io.BytesIO()
        
        with tarfile.open(fileobj=tar_io, mode='w:gz') as tar:
            tarinfo = tarfile.TarInfo("problematic.csv")
            # Invalid UTF-8 sequence
            content = b'\xff\xfe\xfd'
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))
        
        tar_io.seek(0)
        tar_gz_bytes = tar_io.getvalue()
        
        # Call the function
        result = await extract_csv_from_tar_gz_bytes(tar_gz_bytes)
        
        # Should return None on encoding error
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_csv_first_in_multiple(self):
        """Test that the first CSV is extracted when multiple are present"""
        # Create a tar.gz with multiple CSV files
        tar_io = io.BytesIO()
        
        with tarfile.open(fileobj=tar_io, mode='w:gz') as tar:
            # First CSV
            first_csv = "header1,header2\n1,2"
            first_info = tarfile.TarInfo("first.csv")
            first_bytes = first_csv.encode('utf-8')
            first_info.size = len(first_bytes)
            tar.addfile(first_info, io.BytesIO(first_bytes))
            
            # Second CSV
            second_csv = "header3,header4\n3,4"
            second_info = tarfile.TarInfo("second.csv")
            second_bytes = second_csv.encode('utf-8')
            second_info.size = len(second_bytes)
            tar.addfile(second_info, io.BytesIO(second_bytes))
        
        tar_io.seek(0)
        tar_gz_bytes = tar_io.getvalue()
        
        # Call the function
        result = await extract_csv_from_tar_gz_bytes(tar_gz_bytes)
        
        # Should return the first CSV
        assert result == first_csv