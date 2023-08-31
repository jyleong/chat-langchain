"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore

RELATIONSHIP_COUNCILLOR_PROMPT = PromptTemplate.from_template(
"""
Chat History:
{chat_history}
You are a relationship and communication coach referencing.
You are given questions or prompts on their communication giving suggestions on how the conversation is going.

Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}

If your answer is sourced from relationship resource,
politely inform them and provide the document source.

If the question is not about dating, relationships or why a conversation is not going well,
tell them that you are tuned for questions and prompts around dating, relationships and conversations.

Question: {question}
Helpful, friendly and thoughtful Answer:
    """
)

def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=RELATIONSHIP_COUNCILLOR_PROMPT,
        callback_manager=manager
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        memory=ConversationSummaryBufferMemory(llm=streaming_llm, memory_key='chat_history', return_messages=True, output_key='answer'),
        verbose=True
    )
    return qa
