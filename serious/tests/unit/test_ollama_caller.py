import pytest
from unittest.mock import patch, MagicMock, call
import os
import sys

# Add src to Python path
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.llm import OllamaQuestionGenerator

# Fixtures

@pytest.fixture
def ollama_generator_for_prompt_tests():
    """
    Provides an OllamaQuestionGenerator instance suitable for tests
    that do not mock the ollama.Client constructor itself,
    such as prompt generation and parsing logic tests.
    """
    # We don't need a real client for prompt/parsing tests,
    # but the constructor will try to initialize. We can let it fail
    # or mock it here if needed, but for prompt/parse tests,
    # we just need the instance methods. Let's mock the client init
    # to avoid actual connection attempts during these tests.
    with patch('src.llm.ollama_caller.ollama.Client') as MockClient:
        # Ensure list() call in __init__ succeeds for this fixture
        MockClient.return_value.list.return_value = MagicMock() 
        gen = OllamaQuestionGenerator(model="llama3.2:latest", host="http://localhost:11434")
        yield gen # Use yield to ensure patch is active during fixture use

@pytest.fixture
def generator_client_init_fails(capsys):
    """
    Provides an OllamaQuestionGenerator instance where client initialization fails.
    Mocks ollama.Client constructor to raise an exception.
    """
    with patch('src.llm.ollama_caller.ollama.Client') as MockClient:
        MockClient.side_effect = Exception("Ollama server not reachable")
        gen = OllamaQuestionGenerator(model="llama3.2:latest")
        # Capture init output
        capsys.readouterr() 
        yield gen

# Test Client Initialization

@patch('src.llm.ollama_caller.ollama.Client')
def test_generator_initialization_success(mock_ollama_client, capsys):
    """Tests successful Ollama client initialization."""
    # Arrange
    mock_client_instance = mock_ollama_client.return_value
    mock_client_instance.list.return_value = MagicMock() # Ensure list() call succeeds

    # Act
    generator = OllamaQuestionGenerator(model="llama3.2:latest", host="http://test-host:11434")

    # Assert
    mock_ollama_client.assert_called_once_with(host="http://test-host:11434")
    mock_client_instance.list.assert_called_once()
    assert generator.model == "llama3.2:latest"
    assert generator.client_args == {'host': "http://test-host:11434"}
    # Check for success message in output
    captured = capsys.readouterr()
    assert "Ollama client initialized successfully" in captured.out

@patch('src.llm.ollama_caller.ollama.Client')
def test_generator_initialization_default_host_success(mock_ollama_client, capsys):
    """Tests successful Ollama client initialization with default host."""
    # Arrange
    mock_client_instance = mock_ollama_client.return_value
    mock_client_instance.list.return_value = MagicMock() # Ensure list() call succeeds

    # Act
    generator = OllamaQuestionGenerator(model="llama3.2:latest")

    # Assert
    mock_ollama_client.assert_called_once_with() # Called with no args for default host
    mock_client_instance.list.assert_called_once()
    assert generator.model == "llama3.2:latest"
    assert generator.client_args == {}
    # Check for success message in output
    captured = capsys.readouterr()
    assert "Ollama client initialized successfully" in captured.out
    assert "Host: default" in captured.out

@patch('src.llm.ollama_caller.ollama.Client')
def test_generator_initialization_failure(mock_ollama_client, capsys):
    """Tests Ollama client initialization failure."""
    # Arrange
    mock_ollama_client.side_effect = Exception("Connection refused")

    # Act
    generator = OllamaQuestionGenerator(model="llama3.2:latest", host="http://non-existent:11434")

    # Assert
    mock_ollama_client.assert_called_once_with(host="http://non-existent:11434")
    # Check for warning message in output
    captured = capsys.readouterr()
    assert "Warning: Ollama client failed to initialize or connect." in captured.out
    assert "Error: Connection refused" in captured.out
    assert "Ensure Ollama server is running" in captured.out
    # The generator instance is created, but its methods should handle the underlying failure
    assert isinstance(generator, OllamaQuestionGenerator) 

# Test Prompt Creation

def test_prompt_creation_for_summary(ollama_generator_for_prompt_tests):
    """Tests the prompt generated for text summaries."""
    summary = "This is a test summary for Ollama."
    prompt = ollama_generator_for_prompt_tests._create_prompt_for_text_summary(summary, num_questions=2)
    assert "Given the following summary:" in prompt
    assert f'"{summary}"' in prompt
    assert "Generate 2 insightful and distinct questions" in prompt
    assert "Focus on \"what\", \"why\", \"how\", \"compare\", or \"what if\"" in prompt
    assert "Avoid simple yes/no questions." in prompt
    assert "Questions:" in prompt

def test_prompt_creation_for_community(ollama_generator_for_prompt_tests):
    """Tests the prompt generated for community texts."""
    texts_short = ["text one", "text two also short"]
    prompt = ollama_generator_for_prompt_tests._create_prompt_for_community(texts_short, community_id="O1", num_questions=1)
    assert "The following text excerpts belong to a related community (ID: O1)." in prompt
    assert "Excerpts:\n" in prompt
    assert "1. \"text one\"" in prompt  # No "..." for short text
    assert "2. \"text two also short\"" in prompt  # No "..." for short text
    assert "generate 1 insightful and distinct questions." in prompt
    assert "Aim for questions that explore themes, relationships, or implications." in prompt
    assert "Focus on \"what\", \"why\", \"how\", \"compare\", \"correlate\", or \"what if\"." in prompt
    assert "Avoid simple yes/no questions." in prompt
    assert "Questions:" in prompt

    # Test truncation explicitly
    long_text = "a" * 151  # Text longer than 150 chars (Ollama's limit)
    texts_for_truncation = [long_text, "short text"]
    prompt_truncated = ollama_generator_for_prompt_tests._create_prompt_for_community(texts_for_truncation, community_id="O2", num_questions=1)
    assert f"1. \"{long_text[:150]}...\"" in prompt_truncated  # Check truncated part + ellipsis
    assert "2. \"short text\"" in prompt_truncated  # Short text should remain as is

    # Test limit on number of excerpts
    texts_many = [f"text {i}" for i in range(10)]
    prompt_limited = ollama_generator_for_prompt_tests._create_prompt_for_community(texts_many, num_questions=1)
    assert "1. \"text 0\"" in prompt_limited
    assert "2. \"text 1\"" in prompt_limited
    assert "3. \"text 2\"" in prompt_limited
    assert "4. \"text 3\"" in prompt_limited
    assert "5. \"text 4\"" in prompt_limited
    assert "6." not in prompt_limited # Should only include up to 5

# Test Response Parsing

def test_parse_ollama_response(ollama_generator_for_prompt_tests):
    """Tests parsing various Ollama response formats."""
    parser = ollama_generator_for_prompt_tests._parse_ollama_response

    # Test numbered list
    response_numbered = "1. What is the first question?\n2. Why is the second question important?\n3. How does this relate?"
    parsed_numbered = parser(response_numbered)
    assert parsed_numbered == ["What is the first question?", "Why is the second question important?", "How does this relate?"]

    # Test bullet points
    response_bullet = "- Question A\n- Question B\n- Question C"
    parsed_bullet = parser(response_bullet)
    assert parsed_bullet == ["Question A", "Question B", "Question C"]

    # Test mixed format and extra newlines
    response_mixed = "1. First question.\n\n- Second question.\nThird question (no prefix)."
    parsed_mixed = parser(response_mixed)
    assert parsed_mixed == ["First question.", "Second question.", "Third question (no prefix)."]

    # Test single question (no list format)
    response_single = "This is just one question?"
    parsed_single = parser(response_single)
    assert parsed_single == ["This is just one question?"]

    # Test empty response
    response_empty = ""
    parsed_empty = parser(response_empty)
    assert parsed_empty == [""]

    # Test response with only whitespace/newlines
    response_whitespace = "\n   \n"
    parsed_whitespace = parser(response_whitespace)
    assert parsed_whitespace == [""]
    
    # Test numbered list with different formatting (e.g., space after number)
    response_numbered_space = "1. What is it?\n 2. Why is it?"
    parsed_numbered_space = parser(response_numbered_space)
    assert parsed_numbered_space == ["What is it?", "Why is it?"]

    # Test numbered list with parenthesis
    response_numbered_paren = "1) What is it?\n2) Why is it?"
    parsed_numbered_paren = parser(response_numbered_paren)
    # The current parser doesn't handle 1) format, it will keep the prefix.
    # This is acceptable based on the current implementation.
    assert parsed_numbered_paren == ["1) What is it?", "2) Why is it?"] 
    # If we wanted to handle 1) format, the parsing logic in _parse_ollama_response would need adjustment.
    # For now, testing against current behavior.

# Test Question Generation Methods (Mocking API Calls)

@patch('src.llm.ollama_caller.ollama.Client')
def test_generate_questions_from_summary_success(mock_ollama_client):
    """Tests successful question generation from summary."""
    # Arrange
    mock_client_instance = mock_ollama_client.return_value
    mock_client_instance.list.return_value = MagicMock() # Ensure init succeeds

    mock_response_content = "1. What is the summary about?\n2. What are the key points?"
    mock_response = {'message': {'content': mock_response_content}}
    mock_client_instance.chat.return_value = mock_response

    generator = OllamaQuestionGenerator(model="test-model", host="http://test-host")

    # Act
    summary = "This is a test summary."
    questions = generator.generate_questions_from_summary(summary, num_questions=2)

    # Assert
    mock_ollama_client.assert_called_once_with(host="http://test-host")
    mock_client_instance.chat.assert_called_once()
    args, kwargs = mock_client_instance.chat.call_args
    assert kwargs['model'] == "test-model"
    assert len(kwargs['messages']) == 1
    assert kwargs['messages'][0]['role'] == 'user'
    assert "This is a test summary." in kwargs['messages'][0]['content']
    assert "Generate 2 insightful" in kwargs['messages'][0]['content']

    assert len(questions) == 2
    assert questions[0] == "What is the summary about?"
    assert questions[1] == "What are the key points?"

@patch('src.llm.ollama_caller.ollama.Client')
def test_generate_questions_from_community_success(mock_ollama_client):
    """Tests successful question generation from community texts."""
    # Arrange
    mock_client_instance = mock_ollama_client.return_value
    mock_client_instance.list.return_value = MagicMock() # Ensure init succeeds

    mock_response_content = "- How are these texts related?\n- What themes emerge?"
    mock_response = {'message': {'content': mock_response_content}}
    mock_client_instance.chat.return_value = mock_response

    generator = OllamaQuestionGenerator(model="test-model-community")

    # Act
    texts = ["Text one.", "Text two."]
    questions = generator.generate_questions_from_community_texts(texts, community_id="C1", num_questions=2)

    # Assert
    mock_ollama_client.assert_called_once_with() # Default host
    mock_client_instance.chat.assert_called_once()
    args, kwargs = mock_client_instance.chat.call_args
    assert kwargs['model'] == "test-model-community"
    assert len(kwargs['messages']) == 1
    assert kwargs['messages'][0]['role'] == 'user'
    assert "Text one." in kwargs['messages'][0]['content']
    assert "community (ID: C1)" in kwargs['messages'][0]['content']
    assert "generate 2 insightful" in kwargs['messages'][0]['content']

    assert len(questions) == 2
    assert questions[0] == "How are these texts related?"
    assert questions[1] == "What themes emerge?"

@patch('src.llm.ollama_caller.ollama.Client')
def test_generate_questions_from_community_empty_texts(mock_ollama_client):
    """Tests generation from community texts with an empty list."""
    # Arrange
    mock_client_instance = mock_ollama_client.return_value
    mock_client_instance.list.return_value = MagicMock() # Ensure init succeeds

    generator = OllamaQuestionGenerator(model="test-model")

    # Act
    questions = generator.generate_questions_from_community_texts([], community_id="C1")

    # Assert
    mock_ollama_client.assert_called_once() # Client is initialized
    mock_client_instance.chat.assert_not_called() # API call should not happen
    assert questions == []

# Test Error Handling in Generation Methods

@patch('src.llm.ollama_caller.ollama.Client')
def test_generate_questions_summary_api_error(mock_ollama_client, capsys):
    """Tests error handling when Ollama API call fails for summary."""
    # Arrange
    mock_client_instance = mock_ollama_client.return_value
    mock_client_instance.list.return_value = MagicMock() # Ensure init succeeds

    mock_client_instance.chat.side_effect = Exception("Ollama API is down")

    generator = OllamaQuestionGenerator(model="test-model")

    # Act
    questions = generator.generate_questions_from_summary("test summary")

    # Assert
    mock_client_instance.chat.assert_called_once()
    captured = capsys.readouterr()
    assert "Error during Ollama API call for summary: Ollama API is down" in captured.out
    assert len(questions) == 1
    assert "Error generating Ollama questions: Ollama API is down" in questions[0]

@patch('src.llm.ollama_caller.ollama.Client')
def test_generate_questions_community_api_error(mock_ollama_client, capsys):
    """Tests error handling when Ollama API call fails for community texts."""
    # Arrange
    mock_client_instance = mock_ollama_client.return_value
    mock_client_instance.list.return_value = MagicMock() # Ensure init succeeds

    mock_client_instance.chat.side_effect = Exception("Model not found")

    generator = OllamaQuestionGenerator(model="test-model")

    # Act
    questions = generator.generate_questions_from_community_texts(["text 1"])

    # Assert
    mock_client_instance.chat.assert_called_once()
    captured = capsys.readouterr()
    assert "Error during Ollama API call for community: Model not found" in captured.out
    assert len(questions) == 1
    assert "Error generating Ollama community questions: Model not found" in questions[0]

def test_generate_questions_summary_client_init_fails(generator_client_init_fails, capsys):
    """Tests summary generation when client initialization failed."""
    # Arrange: generator_client_init_fails fixture provides the instance
    generator = generator_client_init_fails

    # Act
    questions = generator.generate_questions_from_summary("test summary")

    # Assert
    # No API call should be attempted if client init failed
    # We can't easily assert mock_ollama_client.chat.not_called here because
    # the mock was created *inside* the fixture's patch context.
    # However, the code path explicitly checks for client usability.
    # The error message printed by the method is the primary assertion target.
    captured = capsys.readouterr()
    # The Ollama generator doesn't explicitly check `self.client` before calling `ollama.Client(**self.client_args)`
    # inside the generation methods. It relies on the `ollama.Client` constructor
    # or the subsequent `chat` call to raise an exception if the server is unreachable.
    # So, the error handling path is the same as the API error path in this case.
    # Let's re-evaluate the Ollama code's error handling.
    # Ah, the Ollama code *doesn't* store the client instance. It creates a *new* client
    # in *each* generation method call: `client = ollama.Client(**self.client_args)`.
    # This means the `__init__` check (`client.list()`) only verifies connectivity *at init time*.
    # If the server goes down *after* init, the generation methods will still attempt to connect
    # and the `try...except` block around the `client.chat()` call will catch the error.
    # The `generator_client_init_fails` fixture *does* mock the `ollama.Client` constructor
    # to raise an exception *during initialization*. So, when `generate_questions_from_summary`
    # calls `ollama.Client(**self.client_args)` internally, that mocked constructor will raise,
    # and the `try...except` block in the generation method will catch it. (Old behavior)
    # NEW BEHAVIOR: __init__ fails, self.client is None. The method hits `if not self.client`.
    expected_method_print = "Error: Ollama client not initialized or connection failed during init. Cannot generate questions for summary."
    assert expected_method_print in captured.out

    expected_return_msg = "Error generating Ollama questions: Ollama client not initialized or connection failed during init."
    assert len(questions) == 1
    assert expected_return_msg in questions[0]

def test_generate_questions_community_client_init_fails(generator_client_init_fails, capsys):
    """Tests community generation when client initialization failed."""
    # Arrange: generator_client_init_fails fixture provides the instance
    generator = generator_client_init_fails
    
    # Act
    questions = generator.generate_questions_from_community_texts(["test text"])

    # Assert
    captured = capsys.readouterr()
    # NEW BEHAVIOR: __init__ fails, self.client is None. The method hits `if not self.client`.
    expected_method_print = "Error: Ollama client not initialized or connection failed during init. Cannot generate questions for community."
    assert expected_method_print in captured.out

    expected_return_msg = "Error generating Ollama community questions: Ollama client not initialized or connection failed during init."
    assert len(questions) == 1
    assert expected_return_msg in questions[0]