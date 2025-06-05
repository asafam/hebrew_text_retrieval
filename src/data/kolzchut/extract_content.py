from bs4 import BeautifulSoup, Tag
import glob
import os
import re
import argparse
from markdownify import MarkdownConverter

class CustomConverter(MarkdownConverter):
    def convert_a(self, el, text, *args, **kwargs):
        return text

    def convert_table(self, el, text, *args, **kwargs):
        return "[טבלה]"

# BeautifulSoup is a very strong package that allows you to target and extract very specific parts of the page,
# using tag-names, ids, css-selectors, and more. It is worth reading its documentation. Here, we use it very basically:
# - we extract the "title" and the "main" tags in the document
# - we then pass "main" to a custom MarkdownConverter to convert its content to markdown, while skipping table formatting and formatting links as text only.
# - We then manually split the markdown string into sections.
# You can do much more with BeautifulSoup and its worth looking at its documentation.
# You probably also want to not print everything to the same unformatted string as we do here, but create some data-structure,
# or save to a structured file (say where each item you care about is a jsonl string) or to multiple files.

def extract_content(file_name: str):
    doc = BeautifulSoup(open(fname).read(), "html.parser")
    doc_id = file_name.split("/")[-1].replace(".html", "")
    title = doc.title.contents[0]
    main = doc.main
    as_md = CustomConverter(heading_style="ATX", bullets="*").convert_soup(doc.main)
    as_md = re.sub(r"\n\n+", "\n\n", as_md)
    sections = as_md.split("\n#")
    for section in sections:
        if not section.strip(): continue # skip the before-first section if empty.
        section = "#" + section # add back the "#" we split on.
        sec_title, sec_body = section.split("\n", 1)
    
    return {
        "title": title,
        "doc_id": doc_id,
        "text": 
    }

def foo(corpus_path):
    for i, fname in enumerate(glob.iglob(f"{corpus_path}/pages/*.html")):
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract content from Kol Zchut HTML files.")
    parser.add_argument("corpus_path", type=str, help="Path to the created Kol Zchut corpus directory.")
    args = parser.parse_args()
    extract_content(args.created_kol_zchut_corpus)
