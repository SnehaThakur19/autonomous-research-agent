from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with text generation. It allows a large language
model to use external information when answering questions.

Instead of relying only on information learned during training,
the system retrieves relevant documents and provides them to the
language model as additional context.

RAG is useful when information needs to be current, domain-specific,
or grounded in external sources. It can also help reduce hallucinations
by giving the model relevant evidence before generating an answer.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)

print(f"\nTotal chunks: {len(chunks)}")