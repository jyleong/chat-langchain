"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import UnstructuredEPubLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

BOOKS = [
    "documents/Malcolm Collins - The Pragmatist's Guide to Relationships-Pragmatist Foundation Inc., The (2020).epub",
    "documents/Malcolm Collins_ Simone Collins - The Pragmatist's Guide to Sexuality-Pragmatist Foundation Inc., The (2020).epub",
    "documents/Nonviolent Communication_ A Language of Life -- Marshall B. Rosenberg [Rosenberg, Marshall B.] -- 5a7aff2c7cad8595c7a81aeb7924c03e -- Anna’s Archive.epub",
    "documents/orca_share_media1477764558633.epub",
    "documents/The Relationship Cure - A 5 Step Guide to Strengthening Your Marriage, Family, and Friendships -- John Gottman, Joan DeClaire -- 2017;0 -- Potter_TenSpeed_Harmony -- 9780609809532 -- 599ff3c2721dd4c6268e627e1404123c -- Anna’s Archive.epub",
    "documents/The Seven Principles for Making Marriage Work_ A Practical Guide from the Country’s Foremost -- John Gottman Ph.D., Nan Silver -- Revised ed., 2015 -- Harmony -- 9782014034165 -- 36c8e06afc0c18ec9f63ed175ecf2260 -- Anna’s Archive.epub"
]

def ingest_docs():
    """Get documents from web pages."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    for book in BOOKS:
        epub_loader = UnstructuredEPubLoader(book, mode="elements")
        docs = epub_loader.load_and_split(text_splitter)
        documents += docs
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
