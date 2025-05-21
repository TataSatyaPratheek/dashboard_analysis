# src/core/question_generator.py

# import openai # Keep for synchronous version if needed, or type hints
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from openai import OpenAI, AsyncOpenAI # Import AsyncOpenAI
import asyncio # For concurrency

# Potential: from src.utils.config import load_config
# config = load_config("configs/base.yaml", "configs/m1_optim.yaml")
# openai.api_key = os.getenv("OPENAI_API_KEY", config.openai.api_key)
# DEFAULT_MODEL = config.openai.model
# DEFAULT_TEMPERATURE = config.openai.temperature
# DEFAULT_MAX_TOKENS = config.openai.max_tokens
# The module-level API key management below is removed in favor of instance-based client.
# if not os.getenv("OPENAI_API_KEY"):
#     print("Warning: OPENAI_API_KEY environment variable not set. OpenAI calls will fail.")
# openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAIQuestionGenerator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 150,
        api_key: Optional[str] = None,
        # New parameters for async batching
        request_timeout: float = 20.0, # Timeout per request in seconds
        max_concurrent_requests: int = 5 # Max parallel requests
    ):
        """
        Initializes the OpenAI Question Generator.

        Args:
            model (str): The OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4").
            temperature (float): Sampling temperature for the model.
            max_tokens (int): Maximum number of tokens to generate.
            api_key (Optional[str]): OpenAI API key. If None, attempts to use OPENAI_API_KEY env var.
            request_timeout (float): Timeout for individual API requests.
            max_concurrent_requests (int): Maximum number of concurrent API requests for async methods.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.semaphore = asyncio.Semaphore(max_concurrent_requests) # For limiting concurrency
        
        _effective_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not _effective_api_key:
            print(
                "Warning: OpenAI API key not provided or found in environment. Question generation will not work."
            )
            self.client: Optional[OpenAI] = None
            self.async_client: Optional[AsyncOpenAI] = None
        else:
            self.client = OpenAI(api_key=_effective_api_key, timeout=self.request_timeout)
            self.async_client = AsyncOpenAI(api_key=_effective_api_key, timeout=self.request_timeout)

    # --- Prompt Creation Methods (unchanged) ---
    def _create_prompt_for_text_summary(self, text_summary: str, num_questions: int = 3) -> str:
        """Creates a prompt to generate questions based on a text summary."""
        prompt = f"""Given the following summary of a document or a cluster of related text excerpts:

Summary:
"{text_summary}"

Please generate {num_questions} insightful and distinct questions that can be asked about this summary.
The questions should encourage deeper exploration of the topics mentioned.
Focus on "what", "why", "how", "compare", or "what if" type questions.
Avoid simple yes/no questions.

Questions:
"""
        return prompt

    def _create_prompt_for_community(self, community_texts: List[str], community_id: Optional[Any] = None, num_questions: int = 3) -> str:
        """Creates a prompt to generate questions based on a list of texts from a community."""
        intro = f"The following text excerpts belong to a detected community"
        if community_id is not None:
            intro += f" (ID: {community_id})"
        intro += ". They are related by some common themes or topics.\n\n"

        excerpts_str = ""
        # Use up to 5 excerpts for the prompt, and truncate each if too long
        for i, text in enumerate(community_texts[:5]): 
            if len(text) > 200: # Max length for an excerpt in the prompt
                excerpts_str += f"Excerpt {i+1}: \"{text[:200]}...\"\n"
            else:
                excerpts_str += f"Excerpt {i+1}: \"{text}\"\n" # No ellipsis if not truncated

        prompt = f"""{intro}
Text Excerpts:
{excerpts_str}

Based on these related text excerpts, please generate {num_questions} insightful and distinct questions.
The questions should aim to uncover the core themes of this community, relationships between the excerpts,
or implications of the information presented.
Focus on "what", "why", "how", "compare", "correlate", or "what if" type questions.
Avoid simple yes/no questions.

Questions:
"""
        return prompt

    def _create_prompt_for_community_summary(self, community_texts: List[str], community_id: Optional[Any] = None, source_pdfs: Optional[List[str]] = None) -> str:
        """Creates a prompt to generate a summary for a community of texts."""
        intro = f"The following text excerpts belong to a detected community (ID: {community_id if community_id else 'N/A'})."
        if source_pdfs:
            unique_sources = list(set(source_pdfs))
            if unique_sources: # Only add if there are actual source PDFs provided
                intro += f" These texts are derived from the following document(s): {', '.join(unique_sources)}."
        intro += "\n\nText Excerpts:\n"

        excerpts_str = ""
        # Use up to 5 excerpts for the prompt, and truncate each if too long
        for i, text in enumerate(community_texts[:5]):
            preview = text[:250] + "..." if len(text) > 250 else text # Max length for an excerpt in the prompt
            excerpts_str += f"Excerpt {i+1}: \"{preview}\"\n"

        prompt = f"""{intro}{excerpts_str}
Based on these related text excerpts, please provide a concise summary (around 100-150 words) covering:
- The primary themes or topics discussed in this community.
- The key insights or conclusions that can be drawn from these texts.
- The apparent rationale or purpose of this cluster of information.

Summary:""" # Removed the trailing newline from "Summary:\n" to "Summary:" for potentially better model adherence
        return prompt

    # Inside OpenAIQuestionGenerator class [5]
    def _create_prompt_for_cross_community_correlation(
        self,
        summary_A: str, community_A_id: Any, source_pdfs_A: List[str],
        summary_B: str, community_B_id: Any, source_pdfs_B: List[str],
        num_questions: int = 3
    ) -> str:
        unique_sources_A = list(set(source_pdfs_A))
        unique_sources_B = list(set(source_pdfs_B))

        prompt = f"""Consider two distinct communities of information:
Community {community_A_id} (Sources: {', '.join(unique_sources_A) if unique_sources_A else 'N/A'}):
Summary: "{summary_A}"

Community {community_B_id} (Sources: {', '.join(unique_sources_B) if unique_sources_B else 'N/A'}):
Summary: "{summary_B}"

Based on these summaries and their source document context, generate {num_questions} insightful questions that explore potential correlations, comparisons, or relationships between these two communities. Focus on:
- Shared themes or contrasting viewpoints.
- Potential causal links or interdependencies if applicable (e.g., if one discusses a problem and another a solution).
- How insights from one community might inform or impact the other, especially considering they may span different documents or sections of the same document(s).
- Questions that would require looking at data from both underlying sets of documents to answer.

Correlation Questions:
"""
        return prompt

    # --- Parsing Method (unchanged, but used by both sync and async) ---
    def _parse_openai_response(self, content: str) -> List[str]:
        questions = [q.strip() for q in content.split('\n') if q.strip() and not q.strip().isnumeric()]
        cleaned_questions = []
        for q_text in questions:
            if q_text and q_text[0].isdigit() and (q_text[1] == '.' or q_text[1:2] == '.)'):
                cleaned_questions.append(q_text[2:].lstrip())
            elif q_text:
                cleaned_questions.append(q_text)
        return cleaned_questions if cleaned_questions else [content.strip()] # Fallback if parsing fails or content is single line

    # --- Synchronous Methods (mostly unchanged, use self.client) ---
    def generate_questions_from_summary(self, summary: str, num_questions: int = 3) -> List[str]:
        """
        Generates questions from a given text summary using the OpenAI API.

        Args:
            summary (str): The text summary.
            num_questions (int): The number of questions to generate.

        Returns:
            List[str]: A list of generated questions, or an empty list if an error occurs.
        """
        if not self.client:
            # This print is covered by tests now if API key is None
            print("Error: OpenAI client not initialized. Cannot generate questions.")
            return ["Error: OpenAI API key not set."] # Or return empty list

        prompt = self._create_prompt_for_text_summary(summary, num_questions)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled at generating insightful questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1, # We want one completion that contains multiple questions based on prompt
                stop=None,
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_openai_response(content)

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return [f"Error generating questions: {e}"] # Or return empty list

    def generate_questions_from_community_texts(self, community_texts: List[str], community_id: Optional[Any] = None, num_questions: int = 3) -> List[str]:
        """
        Generates questions from a list of texts belonging to a community.

        Args:
            community_texts (List[str]): A list of text snippets from the community.
            community_id (Optional[Any]): An optional identifier for the community.
            num_questions (int): The number of questions to generate.

        Returns:
            List[str]: A list of generated questions, or an empty list if an error occurs.
        """
        if not self.client:
            print("Error: OpenAI client not initialized. Cannot generate questions.")
            return ["Error: OpenAI API key not set."]

        if not community_texts: # This path needs a test
            return []

        prompt = self._create_prompt_for_community(community_texts, community_id, num_questions)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled at generating insightful questions from related text excerpts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens * num_questions, # Allow more tokens if generating multiple questions
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content.strip()
            return self._parse_openai_response(content)

        except Exception as e:
            print(f"Error during OpenAI API call for community: {e}")
            return [f"Error generating community questions: {e}"]

    # Synchronous version
    def generate_summary_for_community_texts(
        self, community_texts: List[str], community_id: Optional[Any] = None, source_pdfs: Optional[List[str]] = None
    ) -> str:
        if not self.client:
            return "Error: OpenAI client not initialized."
        if not community_texts:
            return "Error: No texts provided for summarization."

        prompt = self._create_prompt_for_community_summary(community_texts, community_id, source_pdfs)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled at summarizing thematic text clusters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature, # Can be lower for summaries, e.g., 0.3-0.5
                max_tokens=self.max_tokens * 2 # Allow more tokens for a summary
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Error during OpenAI API call for community summary: {e}")
            return f"Error generating summary: {e}"

    # Synchronous version
    def generate_correlation_questions(
        self,
        summary_A: str, community_A_id: Any, source_pdfs_A: List[str],
        summary_B: str, community_B_id: Any, source_pdfs_B: List[str],
        num_questions: int = 3
    ) -> List[str]:
        if not self.client:
            return ["Error: OpenAI client not initialized."]

        prompt = self._create_prompt_for_cross_community_correlation(
            summary_A, community_A_id, source_pdfs_A,
            summary_B, community_B_id, source_pdfs_B,
            num_questions
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are skilled at generating analytical questions about relationships between different information sets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature, # Higher temperature might be good for creative correlation questions
                max_tokens=self.max_tokens * num_questions
            )
            content = response.choices[0].message.content.strip()
            return self._parse_openai_response(content) # Use existing parser
        except Exception as e:
            print(f"Error during OpenAI API call for correlation questions: {e}")
            return [f"Error generating correlation questions: {e}"]

    # --- New Asynchronous Methods ---
    async def _async_generate_for_one_item(
        self, item_id: Any, prompt: str, num_questions: int
    ) -> Tuple[Any, List[str]]:
        """Helper to generate questions for a single item asynchronously."""
        if not self.async_client:
            # This error message is more specific for async client initialization issues.
            return item_id, ["Error: Async OpenAI client not initialized. API key may be missing or invalid."]

        async with self.semaphore: # Limit concurrent requests
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant skilled at generating insightful questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens * num_questions, # Adjust if needed
                    n=1, stop=None,
                )
                content = response.choices[0].message.content.strip()
                return item_id, self._parse_openai_response(content)
            except Exception as e:
                print(f"Error during async OpenAI API call for item {item_id}: {e}")
                return item_id, [f"Error generating questions for item {item_id}: {e}"]

    # Asynchronous helper for batch summarization
    async def _async_generate_summary_for_one_community(
        self, community_id: Any, community_texts: List[str], source_pdfs: Optional[List[str]] = None
    ) -> Tuple[Any, str]:
        if not self.async_client:
            return community_id, "Error: Async OpenAI client not initialized."
        if not community_texts:
            return community_id, "Error: No texts provided."

        prompt = self._create_prompt_for_community_summary(community_texts, community_id, source_pdfs)
        async with self.semaphore: # Limit concurrent requests
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an assistant skilled at summarizing text clusters."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature, max_tokens=self.max_tokens * 2
                )
                summary = response.choices[0].message.content.strip()
                return community_id, summary
            except Exception as e:
                return community_id, f"Error generating summary for community {community_id}: {e}"

    async def async_generate_questions_from_summaries(
        self, summaries_with_ids: List[Tuple[Any, str]], num_questions_per_summary: int = 3
    ) -> Dict[Any, List[str]]:
        """
        Generates questions for a batch of summaries asynchronously.

        Args:
            summaries_with_ids (List[Tuple[Any, str]]): A list of tuples, where each tuple is (id, summary_text).
            num_questions_per_summary (int): Number of questions to generate for each summary.

        Returns:
            Dict[Any, List[str]]: A dictionary mapping each id to its list of generated questions.
        """
        if not self.async_client:
            return {item_id: ["Error: Async OpenAI client not initialized. API key may be missing or invalid."] for item_id, _ in summaries_with_ids}

        tasks = []
        for item_id, summary_text in summaries_with_ids:
            prompt = self._create_prompt_for_text_summary(summary_text, num_questions_per_summary)
            tasks.append(self._async_generate_for_one_item(item_id, prompt, num_questions_per_summary))
        
        results_with_ids = await asyncio.gather(*tasks)
        
        return {item_id: questions for item_id, questions in results_with_ids}

    async def async_generate_questions_from_community_texts_batch(
        self, communities_data: List[Dict[str, Any]], num_questions_per_community: int = 3
    ) -> Dict[Any, List[str]]:
        """
        Generates questions for a batch of communities asynchronously.

        Args:
            communities_data (List[Dict[str, Any]]): A list of dictionaries,
                each with "id" (community_id) and "texts" (List[str] of community texts).
            num_questions_per_community (int): Number of questions per community.

        Returns:
            Dict[Any, List[str]]: A dictionary mapping each community_id to its list of generated questions.
        """
        if not self.async_client:
            return {item_data["id"]: ["Error: Async OpenAI client not initialized. API key may be missing or invalid."] for item_data in communities_data}

        tasks = []
        for community_data in communities_data:
            community_id = community_data["id"]
            community_texts = community_data["texts"]
            if not community_texts: # Skip if a community has no texts
                tasks.append(asyncio.create_task(self._return_empty_for_item(community_id))) # Create a task that returns empty
                continue

            prompt = self._create_prompt_for_community(community_texts, community_id, num_questions_per_community)
            tasks.append(self._async_generate_for_one_item(community_id, prompt, num_questions_per_community))
            
        results_with_ids = await asyncio.gather(*tasks)
        
        return {item_id: questions for item_id, questions in results_with_ids}

    # Batch asynchronous summarization
    async def async_generate_summaries_for_communities(
        self, communities_data: List[Dict[str, Any]] # Each dict: {"id": Any, "texts": List[str], "source_pdfs": List[str]}
    ) -> Dict[Any, str]:
        if not self.async_client:
            return {data["id"]: "Error: Async OpenAI client not initialized." for data in communities_data}

        tasks = [
            self._async_generate_summary_for_one_community(
                data["id"], data["texts"], data.get("source_pdfs")
            ) for data in communities_data if data.get("texts")
        ]
        results_with_ids = await asyncio.gather(*tasks)
        return {item_id: summary for item_id, summary in results_with_ids}

    async def _return_empty_for_item(self, item_id: Any) -> Tuple[Any, List[str]]:
        """Helper for asyncio.gather to return an empty list for an item."""
        return item_id, []

    async def _return_error_for_item(self, item_id: Any, error_message: str) -> Tuple[Any, str]:
        """Helper for asyncio.gather to return an error string for an item."""
        return item_id, f"Error: {error_message}"
