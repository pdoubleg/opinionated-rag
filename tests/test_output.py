import os
from pathlib import Path
import pytest
from unittest.mock import patch
from src.utils.output import write_md_to_pdf

@pytest.mark.asyncio
async def test_write_md_to_pdf():
    # Mock the dependencies
    with patch('src.utils.output.write_to_file') as mock_write_to_file, \
         patch('src.utils.output.asyncio.to_thread', return_value=None) as mock_to_thread, \
         patch('src.utils.output.urllib.parse.quote', return_value='mocked_path') as mock_quote:
        
        # Call the function with test data
        file_name = "test"
        text = "## Test Markdown"
        output_dir = Path("./outputs_test")
        expected_file_path = output_dir / f"{file_name}.pdf"
        relative_expected_path = os.path.join('outputs_test', f'{file_name}.pdf').replace(os.sep, '/')
        
        # Ensure the output directory exists for this test
        if not expected_file_path.parent.exists():
            expected_file_path.parent.mkdir(parents=True, exist_ok=True)

        result_path = await write_md_to_pdf(text, file_name=file_name, output_dir=output_dir)
        
        # Verify the file was attempted to be written
        mock_write_to_file.assert_called_once()
        mock_to_thread.assert_called_once()
        mock_quote.assert_called_once_with(relative_expected_path)
        
        # Check the result
        assert result_path == 'mocked_path'
        
        # Cleanup test output directory
        if expected_file_path.exists():
            expected_file_path.unlink()
        if output_dir.exists():
            output_dir.rmdir()
