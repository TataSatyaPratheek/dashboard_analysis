import pytest
from unittest.mock import patch, MagicMock
import os
import asyncio # Import asyncio

# Add src to Python path
import sys
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.llm import OpenAIQuestionGenerator

@pytest.fixture
def test_api_key():
    """Provides a consistent test API key."""
    return "test_key_for_unit_tests"

@pytest.fixture
def openai_generator_for_prompt_tests(test_api_key):
    """
    Provides an OpenAIQuestionGenerator instance initialized with a test API key,
    suitable for tests that do not mock the OpenAI client constructor itself,
    such as prompt generation logic tests.
    """
    gen = OpenAIQuestionGenerator(model="gpt-dummy", api_key=test_api_key)
    return gen

@pytest.fixture
def generator(test_api_key):
    """
    Provides a general OpenAIQuestionGenerator instance initialized with a test API key.
    Suitable for various tests, including async ones.
    """
    # Ensure OPENAI_API_KEY is set for async client initialization if it relies on env var by default
    return OpenAIQuestionGenerator(model="gpt-dummy", api_key=test_api_key, max_concurrent_requests=3)

@pytest.fixture
def generator_no_api_key(monkeypatch):
    """Provides an OpenAIQuestionGenerator instance without an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # Ensure env var is not set
    # Instantiate with api_key=None to test behavior when no key is available
    gen = OpenAIQuestionGenerator(model="gpt-dummy", api_key=None)
    return gen

def test_prompt_creation_for_summary(openai_generator_for_prompt_tests):
    summary = "This is a test summary."
    prompt = openai_generator_for_prompt_tests._create_prompt_for_text_summary(summary, num_questions=2)
    assert "Summary:" in prompt
    assert summary in prompt
    assert "generate 2 insightful" in prompt
    
def test_prompt_creation_for_community(openai_generator_for_prompt_tests):
    texts_short = ["text one", "text two also short"]
    prompt = openai_generator_for_prompt_tests._create_prompt_for_community(texts_short, community_id="A1", num_questions=1)
    assert "community (ID: A1)" in prompt
    assert "Excerpt 1: \"text one\"" in prompt  # No "..." for short text
    assert "Excerpt 2: \"text two also short\"" in prompt  # No "..." for short text
    assert "generate 1 insightful" in prompt

    # Test truncation explicitly
    long_text = "a" * 201  # Text longer than 200 chars
    texts_for_truncation = [long_text, "short text"]
    prompt_truncated = openai_generator_for_prompt_tests._create_prompt_for_community(texts_for_truncation, community_id="A2", num_questions=1)
    assert f"Excerpt 1: \"{long_text[:200]}...\"" in prompt_truncated  # Check truncated part + ellipsis
    assert "Excerpt 2: \"short text\"" in prompt_truncated  # Short text should remain as is

@patch('src.llm.openai_caller.OpenAI')  # Patch the OpenAI class constructor
def test_generate_questions_from_summary_success(mock_openai_constructor, test_api_key):
    mock_client_instance = mock_openai_constructor.return_value
    mock_choice = MagicMock()
    mock_choice.message.content = "1. What is test?\n2. Why is summary important?"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client_instance.chat.completions.create.return_value = mock_response

    # Instantiate OpenAIQuestionGenerator after OpenAI is patched
    generator = OpenAIQuestionGenerator(model="gpt-dummy", api_key=test_api_key, request_timeout=20.0)

    # Assert that OpenAI was constructed as expected
    mock_openai_constructor.assert_called_once_with(api_key=test_api_key, timeout=20.0)

    summary = "Test summary."
    questions = generator.generate_questions_from_summary(summary, num_questions=2)
    
    mock_client_instance.chat.completions.create.assert_called_once()
    args, kwargs = mock_client_instance.chat.completions.create.call_args
    assert kwargs['model'] == "gpt-dummy"
    assert "Test summary." in kwargs['messages'][1]['content']
    
    assert len(questions) == 2
    assert questions[0] == "What is test?"
    assert questions[1] == "Why is summary important?"

@patch('src.llm.openai_caller.OpenAI') # Patch the OpenAI class constructor
def test_generate_questions_from_community_success(mock_openai_constructor, test_api_key):
    mock_client_instance = mock_openai_constructor.return_value
    mock_choice = MagicMock()
    mock_choice.message.content = "What is the main theme?\nHow do excerpts relate?"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client_instance.chat.completions.create.return_value = mock_response

    # Instantiate OpenAIQuestionGenerator after OpenAI is patched
    generator = OpenAIQuestionGenerator(model="gpt-dummy", api_key=test_api_key, request_timeout=20.0)

    # Assert that OpenAI was constructed as expected
    mock_openai_constructor.assert_called_once_with(api_key=test_api_key, timeout=20.0)

    texts = ["Community text 1.", "Community text 2 related to 1."]
    questions = generator.generate_questions_from_community_texts(texts, community_id="C1", num_questions=2)
    
    mock_client_instance.chat.completions.create.assert_called_once()
    args, kwargs = mock_client_instance.chat.completions.create.call_args
    assert kwargs['model'] == "gpt-dummy"
    assert "Community text 1." in kwargs['messages'][1]['content']

    assert len(questions) == 2
    assert questions[0] == "What is the main theme?"
    assert questions[1] == "How do excerpts relate?"

def test_generator_initialization_no_api_key(capsys, monkeypatch):
    # Test __init__ behavior directly when no API key is available
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator_no_key_instance = OpenAIQuestionGenerator(api_key=None)  # Instantiate here
    
    init_captured = capsys.readouterr()  # Capture output from __init__
    assert generator_no_key_instance.client is None
    assert "Warning: OpenAI API key not provided" in init_captured.out

def test_generate_questions_methods_no_api_key(capsys, generator_no_api_key):
    # Uses the generator_no_api_key fixture, which ensures self.client is None
    assert generator_no_api_key.client is None # Verify fixture setup

    questions_summary = generator_no_api_key.generate_questions_from_summary("test summary")
    summary_captured = capsys.readouterr()  # Capture prints from the method call
    assert "Error: OpenAI client not initialized" in summary_captured.out  # From method
    assert "Error: OpenAI API key not set." in questions_summary[0]  # Return value from method

    questions_community = generator_no_api_key.generate_questions_from_community_texts(["test text"])
    community_captured = capsys.readouterr()
    assert "Error: OpenAI client not initialized" in community_captured.out
    assert "Error: OpenAI API key not set." in questions_community[0]

@patch('src.llm.openai_caller.OpenAI') # Patch the OpenAI class constructor
def test_generate_questions_api_error(mock_openai_constructor, capsys, test_api_key):
    mock_client_instance = mock_openai_constructor.return_value
    mock_client_instance.chat.completions.create.side_effect = Exception("API Down")

    # Instantiate OpenAIQuestionGenerator after OpenAI is patched
    generator = OpenAIQuestionGenerator(model="gpt-dummy", api_key=test_api_key, request_timeout=20.0)

    # Assert that OpenAI was constructed as expected
    mock_openai_constructor.assert_called_once_with(api_key=test_api_key, timeout=20.0)

    questions = generator.generate_questions_from_summary("test")
    captured = capsys.readouterr()

    assert "Error during OpenAI API call: API Down" in captured.out
    assert "Error generating questions: API Down" in questions[0]
    mock_client_instance.chat.completions.create.assert_called_once()

# Mark tests that use async features
@pytest.mark.asyncio
async def test_async_generate_for_one_item_success(generator, mocker): # Added generator fixture
    if not generator.async_client: # If API key wasn't set for test env
        pytest.skip("Async client not initialized, skipping async test.")

    mock_choice = MagicMock()
    mock_choice.message.content = "Async Q1\nAsync Q2"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    # Mock the async client's method
    mock_async_create = mocker.AsyncMock(return_value=mock_response)
    generator.async_client.chat.completions.create = mock_async_create

    item_id = "summary1"
    prompt = "Test prompt for async."
    
    returned_id, questions = await generator._async_generate_for_one_item(item_id, prompt, 2)
    
    assert returned_id == item_id
    assert questions == ["Async Q1", "Async Q2"]
    mock_async_create.assert_awaited_once()
    # Check call args on mock_async_create
    args, kwargs = mock_async_create.call_args
    assert kwargs['model'] == generator.model
    assert prompt in kwargs['messages'][1]['content']

@pytest.mark.asyncio
async def test_async_generate_questions_from_summaries_batch_success(generator, mocker): # Added generator fixture
    if not generator.async_client:
        pytest.skip("Async client not initialized, skipping async batch test.")

    summaries = [("s1", "Summary 1"), ("s2", "Summary 2")]
    
    # Mock _async_generate_for_one_item to control its behavior directly for this batch test
    async def mock_one_item_gen(item_id, prompt, num_q):
        await asyncio.sleep(0.01) # Simulate some async work
        if item_id == "s1":
            return "s1", ["Q1 from S1", "Q2 from S1"]
        elif item_id == "s2":
            return "s2", ["Q1 from S2"]
        return item_id, ["Unknown item"]

    mocker.patch.object(generator, '_async_generate_for_one_item', side_effect=mock_one_item_gen)

    results = await generator.async_generate_questions_from_summaries(summaries, num_questions_per_summary=2)
    
    assert len(results) == 2
    assert results["s1"] == ["Q1 from S1", "Q2 from S1"]
    assert results["s2"] == ["Q1 from S2"]
    assert generator._async_generate_for_one_item.call_count == 2


@pytest.mark.asyncio
async def test_async_generate_questions_from_community_texts_batch_success(generator, mocker): # Added generator fixture
    if not generator.async_client:
        pytest.skip("Async client not initialized, skipping async batch test.")

    communities_data = [
        {"id": "c1", "texts": ["text c1a", "text c1b"]},
        {"id": "c2", "texts": ["text c2a"]},
        {"id": "c3", "texts": []} # Test empty texts case
    ]
    
    async def mock_one_item_gen_comm(item_id, prompt, num_q):
        await asyncio.sleep(0.01)
        return item_id, [f"Q for {item_id}"]
    
    # This helper needs to be an async function for create_task
    async def mock_return_empty(item_id):
        await asyncio.sleep(0.01) # Simulate async behavior
        return item_id, []

    mocker.patch.object(generator, '_async_generate_for_one_item', side_effect=mock_one_item_gen_comm)
    mocker.patch.object(generator, '_return_empty_for_item', side_effect=mock_return_empty)

    results = await generator.async_generate_questions_from_community_texts_batch(communities_data, num_questions_per_community=2)
    
    assert len(results) == 3
    assert results["c1"] == ["Q for c1"]
    assert results["c2"] == ["Q for c2"]
    assert results["c3"] == [] # From _return_empty_for_item
    assert generator._async_generate_for_one_item.call_count == 2 # Called for c1 and c2
    assert generator._return_empty_for_item.call_count == 1 # Called for c3

@pytest.mark.asyncio
async def test_async_client_not_initialized(mocker, monkeypatch): # Added monkeypatch for env manipulation
    # Ensure API key is not available for this test
    monkeypatch.delenv("OPENAI_API_KEY", raising=False) # Use monkeypatch to safely delete env var
    
    generator_no_key = OpenAIQuestionGenerator(api_key=None, model="gpt-dummy") # Explicitly pass None
    assert generator_no_key.async_client is None

    summaries = [("s1", "Summary 1")]
    results = await generator_no_key.async_generate_questions_from_summaries(summaries)
    assert results["s1"] == ["Error: Async OpenAI client not initialized. API key may be missing or invalid."]

    communities = [{"id": "c1", "texts": ["text c1a"]}]
    results_comm = await generator_no_key.async_generate_questions_from_community_texts_batch(communities)
    assert results_comm["c1"] == ["Error: Async OpenAI client not initialized. API key may be missing or invalid."]
