from typing import List
from langchain_community.document_loaders import PyPDFLoader


async def load_pdf(file: str) -> List[str]:
    loader = PyPDFLoader(file)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


def prettify_docs(docs) -> str:
    return f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
