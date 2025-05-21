import ollama
from typing import List, Dict, Any, Optional

class OllamaQuestionGenerator:
    def __init__(self, model: str = "llama3.2:latest", host: Optional[str] = None): # Use a default existing tag
        """
        Initializes the Ollama Question Generator.

        Args:
            model (str): The Ollama model to use (e.g., "llama3", "llama3:8b").
            host (Optional[str]): The host for the Ollama API. Defaults to Ollama's default.
        """
        self.model = model
        self.client_args = {}
        self.client: Optional[ollama.Client] = None

        if host:
            self.client_args['host'] = host
        
        try:
            # Check if client can connect and list models to verify setup
            self.client = ollama.Client(**self.client_args)
            self.client.list() # Throws an exception if Ollama server is not reachable
            print(f"Ollama client initialized successfully for model '{self.model}'. Host: {host or 'default'}")
        except Exception as e:
            self.client = None # Ensure client is None on failure
            print(f"Warning: Ollama client failed to initialize or connect. Host: {host or 'default'}. Error: {e}")
            print("Ensure Ollama server is running and the model is pulled (e.g., 'ollama pull llama3').")

    def _create_prompt_for_text_summary(self, text_summary: str, num_questions: int = 3) -> str:
        # Prompts might need slight adjustments for different LLMs
        prompt = f"""Given the following summary:
"{text_summary}"

Generate {num_questions} insightful and distinct questions about this summary.
Focus on "what", "why", "how", "compare", or "what if" type questions.
Avoid simple yes/no questions.

Questions:
"""
        return prompt

    def _create_prompt_for_community(self, community_texts: List[str], community_id: Optional[Any] = None, num_questions: int = 3) -> str:
        intro = f"The following text excerpts belong to a related community"
        if community_id is not None:
            intro += f" (ID: {community_id})"
        intro += ".\n\nExcerpts:\n"
        excerpts_str = ""
        for i, text in enumerate(community_texts[:5]): # Limit excerpts, test expects up to 5
            if len(text) > 150:
                excerpts_str += f"{i+1}. \"{text[:150]}...\"\n"
            else:
                excerpts_str += f"{i+1}. \"{text}\"\n"
        
        prompt = f"""{intro}{excerpts_str}
Based on these excerpts, generate {num_questions} insightful and distinct questions.
Aim for questions that explore themes, relationships, or implications.
Focus on "what", "why", "how", "compare", "correlate", or "what if".
Avoid simple yes/no questions.

Questions:
"""
        return prompt

    def _parse_ollama_response(self, response_content: str) -> List[str]:
        questions = [q.strip() for q in response_content.split('\n') if q.strip()]
        cleaned_questions = []
        for q_text in questions:
            # Llama models might also number questions
            if q_text and q_text[0].isdigit() and (q_text[1] == '.' or q_text[1:2] == '.)'):
                cleaned_questions.append(q_text[2:].lstrip())
            elif q_text and q_text.startswith("- "):
                 cleaned_questions.append(q_text[2:].lstrip())
            elif q_text:
                cleaned_questions.append(q_text)
        # If no questions were parsed, return the stripped original content as a single item list,
        # or [""] if the stripped content is empty.
        return cleaned_questions if cleaned_questions else [response_content.strip()]

    def generate_questions_from_summary(self, summary: str, num_questions: int = 3) -> List[str]:
        prompt = self._create_prompt_for_text_summary(summary, num_questions)
        if not self.client:
            error_msg = "Ollama client not initialized or connection failed during init."
            print(f"Error: {error_msg} Cannot generate questions for summary.")
            return [f"Error generating Ollama questions: {error_msg}"]
        try:
            response = self.client.chat( # Or use .generate for simpler completion
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                # options={'temperature': 0.7, 'num_predict': 150} # Options for generate
            )
            content = response['message']['content'].strip()
            return self._parse_ollama_response(content)
        except Exception as e:
            print(f"Error during Ollama API call for summary: {e}")
            return [f"Error generating Ollama questions: {e}"]

    def generate_questions_from_community_texts(self, community_texts: List[str], community_id: Optional[Any] = None, num_questions: int = 3) -> List[str]:
        if not community_texts:
            return []
        prompt = self._create_prompt_for_community(community_texts, community_id, num_questions)
        if not self.client:
            error_msg = "Ollama client not initialized or connection failed during init."
            print(f"Error: {error_msg} Cannot generate questions for community.")
            return [f"Error generating Ollama community questions: {error_msg}"]
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content'].strip()
            return self._parse_ollama_response(content)
        except Exception as e:
            print(f"Error during Ollama API call for community: {e}")
            return [f"Error generating Ollama community questions: {e}"]
