from pathlib import Path
from typing import Iterable
import json
import time
import os
import re
import click

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from utils.logger_config import setup_logger


logger = setup_logger(__name__)

def write_doc_json(conv_res, filename):
    with filename.open("w") as fp:
        fp.write(json.dumps(conv_res.document.export_to_dict(), indent=2))
    logger.info(f"Exported: {filename}")
    return filename


def write_doc_md(conv_res, filename):
    with filename.open("w") as fp:
        fp.write(conv_res.document.export_to_markdown())
    logger.info(f"Exported: {filename}")
    return filename


def process_directory(input_dir, output_dir):
    file_paths = list(input_dir.rglob("*.pdf"))
    doc_converter = DocumentConverter()
    start_time = time.time()
    conversion_results = doc_converter.convert_all(file_paths)

    success_count = 0
    failure_count = 0
    output_files = []

    for conv_res in conversion_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            logger.info(f"Exporting: {conv_res.input.file}")
            doc_filename = conv_res.input.file.stem
            doc_directory = conv_res.input.file.parent
            doc_output_dir = Path(re.sub(str(input_dir), str(output_dir), str(doc_directory)))
            doc_output_dir.mkdir(parents=True, exist_ok=True)

            # output_json = write_doc_json(conv_res, doc_output_dir / f"{doc_filename}.json")
            # output_files.append(output_json)
            output_md = write_doc_md(conv_res, doc_output_dir / f"{doc_filename}.md")
            output_files.append(output_md)

        else:
            logger.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    logger.info(
        f"Successfully processed {success_count} docs. "
        f"Failed to convert {failure_count} docs. "
        f"Elapsed time: {time.time() - start_time:.2f} seconds."
    )
    return success_count, failure_count

@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Directory containing the documents to convert",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save the converted documents",
    required=True,
)
def main(input_dir: Path, output_dir: Path):
    print(f"Processing directory {input_dir}.  Output will be saved to {output_dir}")
    success_count, failure_count = process_directory(input_dir, output_dir)
    print(f"Conversion complete. {success_count} documents from {input_dir} have been converted to markdown and saved to {output_dir}.")

if __name__ == "__main__":
    main()
