"""
LLM Heuristic Generator - "İlahi Kelam" (Divine Speech)

Integrates with Ollama API to generate optimization operators dynamically.
Uses the DRA metaphor: LLM as the source of divine inspiration for new heuristics.
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from .prompt_templates import (
    PromptTemplate,
    SYSTEM_PROMPT,
    ALL_TEMPLATES,
    TEMPLATE_BY_NAME,
    TEMPLATES_BY_TYPE,
    format_prompt,
    get_prompt_for_stagnation,
)
from .code_sandbox import CodeSandbox, create_fallback_operator


@dataclass
class GeneratedHeuristic:
    """Represents a generated heuristic with metadata."""

    operator: Optional[Callable]
    code: str
    template_name: str
    operator_type: str
    generation_time: float
    success: bool
    error_message: str = ""
    performance_score: float = 0.0
    usage_count: int = 0


class DivineSpeech:
    """
    LLM-based heuristic generator using Ollama API.

    Named "İlahi Kelam" (Divine Speech) in DRA metaphor:
    - The Prophet (DRL agent) requests inspiration
    - Divine Speech (LLM) provides new commandments (operators)
    - The congregation (population) follows the new rules
    """

    def __init__(
        self,
        base_url: str = "https://ollama.parts-soft.net",
        model: str = "qwen3-coder:30b",
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 3,
        sandbox_timeout: int = 5,
    ):
        """
        Initialize the Divine Speech generator.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
            api_key: Bearer token for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            sandbox_timeout: Timeout for code execution
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = 0.7  # Default temperature for generation

        self.last_request_time = 0
        self.min_request_interval = 1.0  # Seconds between requests

        # Logging setup
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "llm_history.jsonl"

        # Code sandbox for safe execution
        self.sandbox = CodeSandbox(timeout=sandbox_timeout)

        # Cache for successful heuristics
        self.heuristic_cache: Dict[str, GeneratedHeuristic] = {}
        self.heuristic_history: List[GeneratedHeuristic] = []

        # Statistics
        self._total_requests = 0
        self._successful_generations = 0
        self._failed_generations = 0
        self._total_tokens = 0

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _log_transaction(
        self,
        prompt: str,
        response_text: str,
        duration: float,
        status: str,
        error: str = None,
    ):
        """Log LLM transaction to JSONL file for live monitoring."""
        try:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration": duration,
                "status": status,
                "prompt": prompt,
                "response": response_text,
                "model": self.model,
                "error": str(error) if error else None,
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Logging failed: {e}")

    def _single_api_call(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Performs a single API call to Ollama, handles rate limiting and logging.

        Args:
            prompt: The prompt to send.

        Returns:
            Tuple of (response_text, error_message).
        """
        start_time = time.time()

        # Enforce rate limiting
        elapsed = start_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        headers = self._build_headers()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # Disable qwen3 thinking mode
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }

        response_text = ""
        error_message = None
        status = "failed"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            response_text = data.get("response", "")
            status = "success"

        except requests.exceptions.Timeout:
            error_message = "Request timeout"
        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {e}"
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
        finally:
            duration = time.time() - start_time
            self._log_transaction(
                prompt, response_text, duration, status, error_message
            )
            self.last_request_time = time.time()
            return response_text if status == "success" else None, error_message

    def _call_api(self, prompt: str) -> Tuple[Optional[str], str]:
        """
        Call Ollama API to generate response with retries.

        Args:
            prompt: The prompt to send

        Returns:
            Tuple of (response_text, error_message)
        """
        last_error = "Unknown error"
        for attempt in range(self.max_retries):
            response_text, error = self._single_api_call(prompt)

            if response_text:
                return response_text, ""

            last_error = error
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)
                continue

        return None, f"Failed after {self.max_retries} retries: {str(last_error)}"

    def generate_heuristic(
        self, template: PromptTemplate, state: Dict[str, Any], dim: int = 30
    ) -> GeneratedHeuristic:
        """
        Generate a new heuristic operator.

        Args:
            template: The prompt template to use
            state: Current optimizer state
            dim: Problem dimension for testing

        Returns:
            GeneratedHeuristic object
        """
        start_time = time.time()
        self._total_requests += 1

        # Build the prompt - no /no_think suffix for qwen3
        system_part = SYSTEM_PROMPT
        user_part = format_prompt(template, state)
        full_prompt = f"{system_part}\n\n{user_part}"

        # Call API
        response_text, error = self._call_api(full_prompt)
        if response_text is None:
            self._failed_generations += 1
            return GeneratedHeuristic(
                operator=None,
                code="",
                template_name=template.name,
                operator_type=template.operator_type,
                generation_time=time.time() - start_time,
                success=False,
                error_message=error,
            )

        # Handle qwen3 thinking mode - extract content after </think> if present
        if "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()

        # Compile and test the generated code
        operator, success, compile_error = self.sandbox.compile_and_test(
            response_text, dim=dim
        )

        generation_time = time.time() - start_time

        if success:
            self._successful_generations += 1
            heuristic = GeneratedHeuristic(
                operator=operator,
                code=response_text,
                template_name=template.name,
                operator_type=template.operator_type,
                generation_time=generation_time,
                success=True,
            )
        else:
            self._failed_generations += 1
            heuristic = GeneratedHeuristic(
                operator=None,
                code=response_text,
                template_name=template.name,
                operator_type=template.operator_type,
                generation_time=generation_time,
                success=False,
                error_message=compile_error,
            )

        # Store in history
        self.heuristic_history.append(heuristic)

        return heuristic

    def generate_for_stagnation(
        self, state: Dict[str, Any], dim: int = 30
    ) -> GeneratedHeuristic:
        """
        Generate appropriate heuristic based on stagnation type.

        Automatically selects the best template based on current state.

        Args:
            state: Current optimizer state
            dim: Problem dimension

        Returns:
            GeneratedHeuristic object
        """
        # Get appropriate prompt
        _, template = get_prompt_for_stagnation(state)

        return self.generate_heuristic(template, state, dim)

    def generate_with_fallback(
        self, template: PromptTemplate, state: Dict[str, Any], dim: int = 30
    ) -> Callable:
        """
        Generate heuristic with fallback if generation fails.

        Args:
            template: Prompt template
            state: Optimizer state
            dim: Problem dimension

        Returns:
            A callable operator (either generated or fallback)
        """
        heuristic = self.generate_heuristic(template, state, dim)

        if heuristic.success and heuristic.operator is not None:
            return heuristic.operator
        else:
            # Return fallback operator
            return create_fallback_operator(template.operator_type)

    def get_best_heuristics(self, top_k: int = 5) -> List[GeneratedHeuristic]:
        """Get top-k best performing heuristics from history."""
        successful = [h for h in self.heuristic_history if h.success]
        sorted_heuristics = sorted(
            successful, key=lambda h: h.performance_score, reverse=True
        )
        return sorted_heuristics[:top_k]

    def update_heuristic_performance(
        self, heuristic: GeneratedHeuristic, improvement: float
    ):
        """
        Update heuristic performance score based on actual improvement.

        Args:
            heuristic: The heuristic to update
            improvement: Improvement achieved (higher is better)
        """
        heuristic.usage_count += 1
        # Running average of performance
        alpha = 1 / heuristic.usage_count
        heuristic.performance_score = (
            1 - alpha
        ) * heuristic.performance_score + alpha * improvement

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_requests": self._total_requests,
            "successful": self._successful_generations,
            "failed": self._failed_generations,
            "success_rate": self._successful_generations / max(1, self._total_requests),
            "total_heuristics": len(self.heuristic_history),
            "sandbox_stats": self.sandbox.get_stats(),
        }

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test API connection.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Test the tags endpoint with auth
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, headers=self._build_headers(), timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if self.model in models:
                    return True, f"Connection successful, model {self.model} available"
                return (
                    True,
                    f"Connected but model {self.model} not found. Available: {models[:5]}",
                )
            else:
                return False, f"API returned status {response.status_code}"

        except Exception as e:
            return False, f"Connection failed: {e}"


# Convenience function for creating generator with config
def create_generator_from_config(config: Dict[str, Any]) -> DivineSpeech:
    """Create DivineSpeech generator from configuration dictionary."""
    llm_config = config.get("llm", {})
    return DivineSpeech(
        base_url=llm_config.get("base_url", "https://ollama.parts-soft.net"),
        model=llm_config.get("model", "qwen3:32b"),
        api_key=llm_config.get("api_key", ""),
        timeout=llm_config.get("timeout", 120),
        max_retries=llm_config.get("max_retries", 3),
    )
