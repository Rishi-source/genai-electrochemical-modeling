"""
Base LLM Client
Wrapper for Azure OpenAI with retry logic, token tracking, and conversation management.
"""

import os
import sys
import time
import json
from typing import List, Dict, Optional, Any, Tuple
from openai import AzureOpenAI
import tiktoken

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.azure_config import (
    AZURE_CONFIG,
    PROMPT_CONFIG,
    RETRY_CONFIG,
    TOKEN_BUDGET,
    get_azure_client_config,
    get_chat_completion_params
)


class BaseLLM:
    """
    Base LLM client for Azure OpenAI.
    
    Features:
    - Automatic retry with exponential backoff
    - Token budget tracking
    - Conversation history management
    - System prompt configuration
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_history: int = PROMPT_CONFIG["max_history"],
        verbose: bool = True
    ):
        """
        Initialize LLM client.
        
        Args:
            system_prompt: System role prompt (optional)
            max_history: Maximum conversation history to retain
            verbose: Print debug messages
        """
        self.verbose = verbose
        self.max_history = max_history
        
        # Initialize Azure OpenAI client
        config = get_azure_client_config()
        self.client = AzureOpenAI(**config)
        
        # System prompt
        self.system_prompt = system_prompt or PROMPT_CONFIG["system_role"]
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Tokenizer for counting
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        if self.verbose:
            print(f"✓ BaseLLM initialized")
            print(f"  Model: {AZURE_CONFIG['model']}")
            print(f"  Deployment: {AZURE_CONFIG['deployment_name']}")
            print(f"  Max history: {self.max_history}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Token count
        """
        return len(self.tokenizer.encode(text))
    
    def add_to_history(self, role: str, content: str):
        """
        Add message to conversation history.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Trim history if exceeds max
        if len(self.conversation_history) > self.max_history:
            # Keep system message + recent messages
            system_msgs = [m for m in self.conversation_history if m["role"] == "system"]
            recent_msgs = self.conversation_history[-(self.max_history-len(system_msgs)):]
            self.conversation_history = system_msgs + recent_msgs
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def chat_completion(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_history: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send chat completion request with retry logic.
        
        Args:
            user_message: User message
            temperature: Sampling temperature (optional)
            max_tokens: Maximum output tokens (optional)
            use_history: Include conversation history
        
        Returns:
            Tuple of (response text, metadata dict)
        """
        # Build messages
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add history if requested
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get completion params
        params = get_chat_completion_params()
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Retry logic
        max_retries = RETRY_CONFIG["max_retries"]
        delay = RETRY_CONFIG["initial_delay"]
        
        for attempt in range(max_retries):
            try:
                # API call
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=params["deployment_name"],
                    messages=messages,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    top_p=params["top_p"],
                    frequency_penalty=params["frequency_penalty"],
                    presence_penalty=params["presence_penalty"]
                )
                latency = time.time() - start_time
                
                # Extract response
                response_text = response.choices[0].message.content
                
                # Track tokens
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                
                # Add to history
                if use_history:
                    self.add_to_history("user", user_message)
                    self.add_to_history("assistant", response_text)
                
                # Metadata
                metadata = {
                    "model": AZURE_CONFIG["model"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "latency": latency,
                    "finish_reason": response.choices[0].finish_reason,
                    "attempt": attempt + 1
                }
                
                if self.verbose:
                    print(f"✓ LLM response ({output_tokens} tokens, {latency:.2f}s)")
                
                return response_text, metadata
            
            except Exception as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"⚠ API call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                        print(f"  Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * RETRY_CONFIG["exponential_base"], RETRY_CONFIG["max_delay"])
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {str(e)}")
    
    def function_call_completion(
        self,
        user_message: str,
        functions: List[Dict[str, Any]],
        function_call: Optional[str] = "auto"
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Chat completion with function calling.
        
        Args:
            user_message: User message
            functions: List of function definitions
            function_call: Function call mode ("auto", "none", or function name)
        
        Returns:
            Tuple of (response text, function call dict, metadata)
        """
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Get params
        params = get_chat_completion_params()
        
        # API call with retry
        max_retries = RETRY_CONFIG["max_retries"]
        delay = RETRY_CONFIG["initial_delay"]
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=params["deployment_name"],
                    messages=messages,
                    functions=functions,
                    function_call=function_call,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"]
                )
                latency = time.time() - start_time
                
                # Extract response
                choice = response.choices[0]
                response_text = choice.message.content if choice.message.content else None
                function_call_result = None
                
                if choice.message.function_call:
                    function_call_result = {
                        "name": choice.message.function_call.name,
                        "arguments": json.loads(choice.message.function_call.arguments)
                    }
                
                # Metadata
                metadata = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "latency": latency,
                    "finish_reason": choice.finish_reason
                }
                
                return response_text, function_call_result, metadata
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * RETRY_CONFIG["exponential_base"], RETRY_CONFIG["max_delay"])
                else:
                    raise Exception(f"Function call failed: {str(e)}")
    
    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "max_input_tokens": TOKEN_BUDGET["max_input_tokens"],
            "max_output_tokens": TOKEN_BUDGET["max_output_tokens"],
            "usage_percentage": (
                100.0 * (self.total_input_tokens + self.total_output_tokens) / 
                TOKEN_BUDGET["max_input_tokens"]
            )
        }
    
    def reset_stats(self):
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


def main():
    """Test BaseLLM."""
    print("="*70)
    print("Testing BaseLLM")
    print("="*70)
    
    # Initialize
    llm = BaseLLM(
        system_prompt="You are a helpful assistant expert in electrochemistry.",
        verbose=True
    )
    
    # Test single completion
    print("\n1. Single completion test:")
    response, metadata = llm.chat_completion(
        "What is the typical exchange current density for Pt/C catalyst in PEMFCs?"
    )
    print(f"Response: {response[:200]}...")
    print(f"Metadata: {metadata}")
    
    # Test with history
    print("\n2. Conversation with history:")
    response, _ = llm.chat_completion("And what about the temperature dependence?")
    print(f"Response: {response[:200]}...")
    
    # Get stats
    print("\n3. Token statistics:")
    stats = llm.get_token_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\n✓ BaseLLM tests complete")


if __name__ == "__main__":
    main()
