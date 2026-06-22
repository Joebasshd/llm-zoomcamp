INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

from google.genai import types


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gemini-2.5-flash'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        # For the lessons dataset our documents use `content` and `filename` fields.
        # Use a simple search over the index (minsearch) that was built with
        # `text_fields=['content']` and `keyword_fields=['filename']`.
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            # documents from the lessons index have `filename` and `content`
            filename = doc.get('filename') # or doc.get('meta', {}).get('source') or ''
            content = doc.get('content') # or doc.get('text') or ''
            lines.append(f"FILE: {filename}")
            lines.append(content)
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        # Return the full response object so callers can inspect usage metadata
        config = types.GenerateContentConfig(
            system_instruction=self.instructions,
        )

        message_history = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )
        ]

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=message_history,
            config=config,
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)

        # Extract answer text and usage (usage field names vary by client)
        answer_text = getattr(response, 'text', None) or getattr(response, 'output_text', None) or str(response)
        usage = getattr(response, 'usage', None) or getattr(response, 'usage_metadata', None) or getattr(response, 'usage_tokens', None)

        return answer_text, usage

class OllamaRAG(RAGBase):

    def llm(self, prompt):
        # same behavior as base but kept as override in case we need Ollama-specific tweaks
        config = types.GenerateContentConfig(
            system_instruction=self.instructions,
        )

        message_history = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )
        ]

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=message_history,
            config=config
        )

        return response
