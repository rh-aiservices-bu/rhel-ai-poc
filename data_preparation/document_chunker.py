# Standard
from pathlib import Path
from typing import Iterable
import json
import time
import os
import re
import shutil

# Third Party
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from utils.logger_config import setup_logger
import click
import pandas as pd

# Local
from utils.docprocessor import DocProcessor

logger = setup_logger(__name__)


def export_documents(
        conversion_results: Iterable[ConversionResult],
        output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    for conv_res in conversion_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            # Export Deep Search document JSON format:
            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                fp.write(json.dumps(conv_res.legacy_document.export_to_dict()))

            # Export Markdown format:
            with (output_dir / f"{doc_filename}.md").open("w") as fp:
                fp.write(conv_res.legacy_document.export_to_markdown())
        else:
            logger.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    logger.info(
        f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
    )

    return doc_filename


def process_directory(input_dir, output_dir):
    file_paths = list(input_dir.rglob("*.pdf"))
    doc_converter = DocumentConverter()
    start_time = time.time()
    converted_docs = doc_converter.convert_all(file_paths)
    doc_filenames = export_documents(converted_docs, output_dir)
    end_time = time.time()

    logger.info(f"Parsing documents took {end_time - start_time:.2f} seconds")

    dp = DocProcessor(output_dir, user_config_path=f'{input_dir}/qna.yaml')
    seed_data = dp.get_processed_dataset()

    jsonl_file_path = f"{output_dir}/seed_data.jsonl"
    seed_data.to_json(jsonl_file_path, orient='records', lines=True)

    md_output_dir = f"{output_dir}/md"
    os.makedirs(md_output_dir, exist_ok=True)
    return jsonl_file_path


def get_leaf_directories(root_dir):
    leaf_dirs = []
    for dirpath, dirs, files in os.walk(root_dir):
        if not dirs:
            logger.info(dirpath, type(dirpath))
            leaf_dirs.append(dirpath)
    return leaf_dirs


def process_jsonl_file(jsonl_file_path):
    output_dir = os.path.dirname(jsonl_file_path)
    md_output_dir = f"{output_dir}/md"
    os.makedirs(md_output_dir, exist_ok=True)

    with open(jsonl_file_path, 'r') as f:
        saved_hashes = set()
        i = 0
        for line in f:
            entry = json.loads(line)
            document_text = entry.get('document', '')
            h = hash(document_text)
            if h not in saved_hashes:
                saved_hashes.add(h)
                i += 1
                file_path = os.path.join(md_output_dir, f"document_{i}.md")
                with open(file_path, 'w') as f:
                    f.write(document_text)


def copy_qna_files(input_dir, output_dir, overwrite=False):
    qna_file_paths = list(input_dir.rglob("qna.yaml"))
    for src_file in qna_file_paths:
        if src_file.is_file():
            rel_path = src_file.relative_to(input_dir)
            if str(rel_path).find("knowledge") != -1:
                dest_file = output_dir / "taxonomy" / rel_path
            else:
                dest_file = output_dir / "taxonomy/knowledge" / rel_path

            if overwrite or not dest_file.exists():
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {rel_path}")
            else:
                print(f"Skipped (already exists): {rel_path}")


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
    dirs = get_leaf_directories(input_dir)
    for dir in dirs:
        print(f"Processing directory: {dir}")
        process_directory(Path(dir), Path(re.sub(str(input_dir), str(output_dir), dir)))

    jsonl_file_paths = list(output_dir.rglob("seed_data.jsonl"))
    for jsonl_file_path in jsonl_file_paths:
        process_jsonl_file(jsonl_file_path)

    copy_qna_files(input_dir, output_dir)

    print("Chunking finished")

if __name__ == "__main__":
    main()
