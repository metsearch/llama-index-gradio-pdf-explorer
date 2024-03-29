from pathlib import Path

import click
import dotenv

from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from utilities.utils import *

dotenv.load_dotenv()

@click.command()
@click.option('--pdf_path', help='Path to PDF file', default='data/Resume.pdf')
def explorer(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'File not found: {pdf_path}')
    
    pdf_loader = PDFReader()
    resume_document = pdf_loader.load_data(file=Path(pdf_path))
    resume_index = get_pdf_index(resume_document, 'resume_index')
    resume_engine = resume_index.as_query_engine()
    
    query_engine_tools = [
        QueryEngineTool(
            query_engine=resume_engine,
            metadata=ToolMetadata(
                name='Resume',
                description=(
                    'This is a resume which gives a brief overview of the candidate\'s skills, formation and experience.'
                ),
            ),
        ),
    ]

    llm = OpenAI(model='gpt-3.5-turbo-0613')
    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
    )
    
    while (prompt := input('A question: ')) != 'q':
        response = agent.query(prompt)
        print(f'LLM: {response}')
    
    
if __name__ == '__main__':
    logger.info('Processing...')
    explorer( )