import os

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage
)

from .log import logger

def get_pdf_index(documents, index_name):
    index = None
    if not os.path.exists(index_name):
        logger.info(f'Creating index {index_name}...')
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(index_name)
    else:
        logger.info(f'Loading index {index_name}...')
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        index = load_index_from_storage(storage_context)
        
    return index

if __name__ == '__main__':
    logger.info('Testing utils...')