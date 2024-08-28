import streamlit as st
from datetime import datetime

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from supabase import create_client, Client

from pydantic import BaseModel, Field
from typing import List, Tuple, Any

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize

import requests
from bs4 import BeautifulSoup

# Ensure that the necessary NLTK data is available
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# CONSTANTS
# The embedding model to use (vector size 1,536)
ENGINE = 'text-embedding-3-small'
# Define constants for the Pinecone index and namespace
INDEX_NAME = st.secrets["INDEX_NAME"]  # The name of the original Pinecone index
NAMESPACE = st.secrets["NAMESPACE"]  # The namespace to use within the index

print("Loading functions and classes...")
# FUNCTIONS
def extract_text_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    container = soup.find('div', class_='single_content')

    if container:
        resolution_text = container.get_text()

        # Remove the first 6 lines (For printing purposes)
        lines = resolution_text.split('\n')
        filtered_lines = lines[6:]

        # Join the remaining lines back into a single string
        filtered_text = '\n'.join(filtered_lines)
        return filtered_text
    else:
        return ""



def summarize_texts(texts, num_sentences=80):
    """
    Extracts title from texts. Summarizes the given texts using LexRank summarization and prints the number
    of tokens in the summary. 
    Args:
    - texts (list): The texts to be summarized.
    - num_sentences (int): The number of sentences to include in the summary.
    Returns:
    - str: The summary of the text with the title.
    """
    summary_texts = []
    for text in texts:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        # Initialize the LexRank summarizer
        summarizer = LexRankSummarizer()
        # Generate the summary
        summary = summarizer(parser.document, num_sentences)
        # Convert the summary to a single string
        summary_text = " ".join([str(sentence) for sentence in summary])
        summary_texts.append(summary_text)
    return summary_texts


def get_embedding(text, engine=ENGINE):
    """
    Function to get embedding for a single text using the OpenAI API
    """
    return get_embeddings([text], engine=engine)[0]


def get_embeddings(texts, engine=ENGINE):
    # Create embeddings for the input texts using the specified engine
    response = client.embeddings.create(
        input=summarize_texts(texts),
        model=engine
    )

    # Extract and return the list of embeddings from the response
    return [d.embedding for d in list(response.data)]


def query_from_pinecone(query, top_k=3, include_metadata=True):
    """
    Queries pinecone and returns the top results.
    """
    # get embedding from THE SAME embedder as the documents
    query_embedding = get_embedding(query, engine=ENGINE)

    return index.query(
      vector=query_embedding,
      top_k=top_k,
      namespace=NAMESPACE,
      include_metadata=include_metadata   # gets the metadata (dates, text, etc)
    ).get('matches')


# Define a class for the Chat Language Model
class OpenAIChatLLM(BaseModel):
    model: str = 'gpt-4o'  # Default model to use
    temperature: float = 0.0  # Default temperature for generating responses

    # Method to generate a response from the model based on the provided prompt
    def generate(self, prompt: str, stop: List[str] = None):
        # Create a completion request to the OpenAI API with the given parameters
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop
        )

        # Insert the details of the prompt and response into the 'cost_projecting' table in Supabase
        supabase.table('cost_projecting').insert({
            'prompt': prompt,
            'response': response.choices[0].message.content,
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens,
            'model': self.model,
            'inference_params': {
                'temperature': self.temperature,
                'stop': stop
            },
            'is_openai': True,
            'app': 'RAG'
        }).execute()

        # Return the generated response content
        return response.choices[0].message.content
    
FINAL_ANSWER_TOKEN = "Legal Assistant Response:"
STOP = '[END]'
PROMPT_TEMPLATE = """Today is {today} and you can retrieve information from a database. Respond to the user's input as best as you can.

Here is an example of the conversation format:

[START]
User Input: the input question you must answer
Context: retrieved context from the database
Context URL: context url
Context Score : a score from 0 - 1 of how strong the information is a match
Assistant Thought: This context has sufficient information to answer the question.
Assistant Response: your final answer to the original input question which could be I don't have sufficient information to answer the question.
[END]
[START]
User Input: another input question you must answer
Context: more retrieved context from the database
Context URL: context url
Context Score : another score from 0 - 1 of how strong the information is a match
Assistant Thought: This context does not have sufficient information to answer the question.
Assistant Response: your final answer to the second input question which could be I don't have sufficient information to answer the question.
[END]
[START]
User Input: another input question you must answer
Context: NO CONTEXT FOUND
Context URL: NONE
Context Score : 0
Assistant Thought: We either could not find something or we don't need to look something up
Assistant Response: Reason through the best way to answer
[END]

Begin:

{running_convo}
"""


class RagBot(BaseModel):
    llm: Any
    prompt_template: str = PROMPT_TEMPLATE
    stop_pattern: List[str] = [STOP]
    user_inputs: List[str] = []
    ai_responses: List[str] = []
    contexts: List[Tuple[str, float]] = []
    verbose: bool = False
    threshold: float = 0.40

    def query_from_pinecone(self, query, top_k=1, include_metadata=True):
        return query_from_pinecone(query, top_k, include_metadata)

    @property
    def running_convo(self):
        convo = ''
        for index in range(len(self.user_inputs)):
            convo += f'[START]\nUser Input: {self.user_inputs[index]}\n'
            convo += f'Context: {self.contexts[index][0]}\nContext URL: {self.contexts[index][1]}\nContext Score: {self.contexts[index][2]}\n'
            if len(self.ai_responses) > index:
                convo += self.ai_responses[index]
                convo += '\n[END]\n'
        return convo.strip()

    def expand_query(self, query):
        # Expand short queries to provide more context
        if len(query.split()) < 5:  # Adjust threshold as needed
            return f"Provide a detailed explanation of: {query}. Include its legal provisions, history, and key implications."
        return query
        
    def run(self, question: str):
        expanded_question = self.expand_query(question)
        self.user_inputs.append(expanded_question)
        top_response = self.query_from_pinecone(expanded_question)[0]
        print(top_response['score'])
        if top_response['score'] >= self.threshold:
            self.contexts.append((extract_text_data(top_response['metadata']['url']), top_response['metadata']['url'], top_response['score']))
        else:
            self.contexts.append(('NO CONTEXT FOUND', 'NONE', 0))

        prompt = self.prompt_template.format(
                today=datetime.now(),
                running_convo=self.running_convo
        )
        if self.verbose:
            print('--------')
            print('PROMPT')
            print('--------')
            print(prompt)
            print('--------')
            print('END PROMPT')
            print('--------')
        generated = self.llm.generate(prompt, stop=self.stop_pattern)
        if self.verbose:
            print('--------')
            print('GENERATED')
            print('--------')
            print(generated)
            print('--------')
            print('END GENERATED')
            print('--------')
        self.ai_responses.append(generated)
        if FINAL_ANSWER_TOKEN in generated:
            generated = generated.split(FINAL_ANSWER_TOKEN)[-1]
        return generated
    

print("Initializing Program")



# Initialize the OpenAI client with the API key from secrets
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# Initialize the Pinecone client with the retrieved API key
pc = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"]
)

# Initialize the Supabase client with the retrieved API key
SUPABASE_URL: str = st.secrets["SUPABASE_URL"]
SUPABASE_API_KEY: str = st.secrets["SUPABASE_API_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

index = pc.Index(name=INDEX_NAME)
print(index)
print(index.describe_index_stats())

query1 = """In the city of Maribajo, a dispute arose in 1995 when a local electronics company, Visatron, sought to expand its business by opening a repair center. The city’s mayor, Carlos Fernandez, granted a business permit but imposed several conditions: Visatron could not provide diagnostic services, only sell accessories; the repair services must be outsourced to independent technicians; and advertisements were limited to basic product listings. The Maribajo Technicians Guild filed a complaint, arguing that these conditions unfairly restricted the scope of Visatron's business. Following a brief investigation by the city’s legal office, the mayor revoked Visatron’s permit. Visatron, contesting the decision on grounds of due process and claiming the conditions were arbitrary, escalated the matter to the local trial court, initiating a legal battle that questioned the boundaries of municipal regulatory powers."""
# Should match G.R. No. 100152, March 31, 2000
matching_documents_query1 = query_from_pinecone(query1, top_k=1)
for matching_doc in matching_documents_query1:
    print(matching_doc['metadata']['url'], matching_doc['score'])
    print(matching_doc['id'])
print("End test")

# r = RagBot(llm=OpenAIChatLLM(temperature=0.0), stop_pattern=['[END]'])
# print(r.run(query1))
# match = query_from_pinecone(prompt, top_k=1)[0]
#         print(matching_doc['metadata']['url'], matching_doc['score'])