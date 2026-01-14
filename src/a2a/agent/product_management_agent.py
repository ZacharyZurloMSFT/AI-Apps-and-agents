import asyncio
import logging
import os
from collections.abc import AsyncIterable
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal
import httpx
import openai
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import BaseModel
from agent_framework import AgentThread, TextContent, ChatAgent, BaseChatClient, ai_function
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AzureOpenAIChatClient

logger = logging.getLogger(__name__)
load_dotenv()


# Custom wrapper to use AIProjectClient's OpenAI client with Responses API
class ResponsesAPIWrapper:
    """Wraps the Responses API to provide an async chat completions interface."""
    
    def __init__(self, project_client: AIProjectClient, deployment_name: str):
        self.project_client = project_client
        self.deployment_name = deployment_name
        self.sync_client = project_client.get_openai_client()
    
    async def create(self, messages, model=None, **kwargs):
        """Translate chat completions call to responses API call."""
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage
        import time
        
        # Convert messages format to responses API input format
        input_messages = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Handle different content formats
            if isinstance(content, str):
                content_items = [{'type': 'input_text', 'text': content}]
            elif isinstance(content, list):
                content_items = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content_items.append({'type': 'input_text', 'text': item.get('text', '')})
                    elif isinstance(item, str):
                        content_items.append({'type': 'input_text', 'text': item})
                    else:
                        content_items.append(item)
            else:
                content_items = [{'type': 'input_text', 'text': str(content)}]
            
            input_messages.append({
                'type': 'message',
                'role': role,
                'content': content_items
            })
        
        # Call the responses API synchronously (run in executor for async)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.sync_client.responses.create(
                model=model or self.deployment_name,
                input=input_messages,
                max_output_tokens=kwargs.get('max_tokens', 4096),
            )
        )
        
        # Extract output text from response
        output_text = getattr(response, 'output_text', '')
        if isinstance(output_text, list):
            output_text = output_text[0] if output_text else ''
        if not output_text:
            # Try to extract from output structure
            output = getattr(response, 'output', [])
            if output and len(output) > 0:
                first_output = output[0]
                if hasattr(first_output, 'content'):
                    content = first_output.content
                    if isinstance(content, list) and len(content) > 0:
                        if hasattr(content[0], 'text'):
                            output_text = content[0].text
        
        # Create proper OpenAI ChatCompletion response
        chat_response = ChatCompletion(
            id=getattr(response, 'id', f'chatcmpl-{int(time.time())}'),
            created=int(time.time()),
            model=model or self.deployment_name,
            object='chat.completion',
            choices=[
                Choice(
                    index=0,
                    finish_reason='stop',
                    message=ChatCompletionMessage(
                        role='assistant',
                        content=output_text or '',
                    ),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )
        
        return chat_response


class AsyncChatCompletionsWrapper:
    """Wrapper to provide async chat.completions interface using Responses API."""
    
    def __init__(self, responses_wrapper: ResponsesAPIWrapper):
        self._wrapper = responses_wrapper
    
    async def create(self, *args, **kwargs):
        return await self._wrapper.create(*args, **kwargs)


class AsyncOpenAIClientWrapper:
    """Wrapper to make the Responses API look like an AsyncOpenAI client."""
    
    def __init__(self, project_client: AIProjectClient, deployment_name: str):
        self._responses_wrapper = ResponsesAPIWrapper(project_client, deployment_name)
        self.chat = type('Chat', (), {'completions': AsyncChatCompletionsWrapper(self._responses_wrapper)})()


# region Chat Service Configuration

class ChatServices(str, Enum):
    """Enum for supported chat completion services."""

    AZURE_OPENAI = 'azure_openai'
    OPENAI = 'openai'


service_id = 'default'


def get_chat_completion_service(
    service_name: ChatServices,
) -> 'BaseChatClient':
    """Return an appropriate chat completion service based on the service name.

    Args:
        service_name (ChatServices): Service name.

    Returns:
        BaseChatClient: Configured chat completion service.

    Raises:
        ValueError: If the service name is not supported or required environment variables are missing.
    """
    if service_name == ChatServices.AZURE_OPENAI:
        return _get_azure_openai_chat_completion_service()
    if service_name == ChatServices.OPENAI:
        return _get_openai_chat_completion_service()
    raise ValueError(f'Unsupported service name: {service_name}')


def _get_azure_openai_chat_completion_service() -> AzureOpenAIChatClient:
    """Return Azure OpenAI chat completion service using AIProjectClient.

    This function handles Azure AI Foundry endpoints which use the
    Responses API by wrapping it in a chat completions interface.

    Returns:
        AzureOpenAIChatClient: The configured Azure OpenAI service.
    """
    endpoint = os.getenv('FOUNDRY_ENDPOINT') or os.getenv('gpt_endpoint')
    deployment_name = os.getenv('gpt_deployment')

    if not endpoint:
        raise ValueError("FOUNDRY_ENDPOINT or gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")

    logger.info(f"Configuring Azure OpenAI client for endpoint: {endpoint}")
    logger.info(f"Deployment: {deployment_name}")
    
    # Use AIProjectClient for Azure AI Foundry endpoints
    logger.info("Using AIProjectClient with DefaultAzureCredential")
    
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=credential,
    )
    
    # Create a wrapper that translates chat completions to responses API
    async_client = AsyncOpenAIClientWrapper(project_client, deployment_name)
    
    logger.info("Azure OpenAI client configured with Responses API wrapper")
    return AzureOpenAIChatClient(
        service_id=service_id,
        deployment_name=deployment_name,
        async_client=async_client,
    )

def _get_openai_chat_completion_service() -> OpenAIChatClient:
    """Return OpenAI chat completion service.

    Returns:
        OpenAIChatClient: Configured OpenAI service.
    """
    return OpenAIChatClient(
        service_id=service_id,
        model_id=os.getenv('OPENAI_MODEL_ID'),
        api_key=os.getenv('OPENAI_API_KEY'),
    )


# endregion

# region Get Products

@ai_function(
    name='get_products',
    description='Retrieves a set of products based on a natural language user query.'
)
def get_products(
    self,
    question: Annotated[
        str, 'Natural language query to retrieve products, e.g. "What kinds of paint rollers do you have in stock?"'
    ],
) -> list[dict[str, Any]]:
    try:
        # Simulate product retrieval based on the question
        # In a real implementation, this would query a database or external service
        product_dict = [
            {
                "id": "1",
                "name": "Eco-Friendly Paint Roller",
                "type": "Paint Roller",
                "description": "A high-quality, eco-friendly paint roller for smooth finishes.",
                "punchLine": "Roll with the best, paint with the rest!",
                "price": 15.99
            },
            {
                "id": "2",
                "name": "Premium Paint Brush Set",
                "type": "Paint Brush",
                "description": "A set of premium paint brushes for detailed work and fine finishes.",
                "punchLine": "Brush up your skills with our premium set!",
                "price": 25.49
            },
            {
                "id": "3",
                "name": "All-Purpose Paint Tray",
                "type": "Paint Tray",
                "description": "A durable paint tray suitable for all types of rollers and brushes.",
                "punchLine": "Tray it, paint it, love it!",
                "price": 9.99
            }
        ]
        return product_dict
    except Exception as e:
        return f'Product recommendation failed: {e!s}'


# endregion



# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


# endregion
# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework-based agents to handle Zava product management tasks."""

    agent: ChatAgent
    thread: AgentThread = None
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        # Configure the chat completion service explicitly
        chat_service = get_chat_completion_service(ChatServices.AZURE_OPENAI)

                # Define an MarketingAgent to handle marketing-related tasks
        marketing_agent = ChatAgent(
            chat_client=chat_service,
            name='MarketingAgent',
            instructions=(
                'You specialize in planning and recommending marketing strategies for products. '
                'This includes identifying target audiences, making product descriptions better, and suggesting promotional tactics. '
                'Your goal is to help businesses effectively market their products and reach their desired customers.'
            ),
        )

        # Define an RankerAgent to sort and recommend results
        ranker_agent = ChatAgent(
            chat_client=chat_service,
            name='RankerAgent',
            instructions=(
                'You specialize in ranking and recommending products based on various criteria. '
                'This includes analyzing product features, customer reviews, and market trends to provide tailored suggestions. '
                'Your goal is to help customers find the best products for their needs.'
            ),
        )

                # Define a ProductAgent to retrieve products from the Zava catalog
        product_agent = ChatAgent(
            chat_client=chat_service,
            name='ProductAgent',
            instructions=("""
                You specialize in handling product-related requests from customers and employees.
                This includes providing a list of products, identifying available quantities,
                providing product prices, and giving product descriptions as they exist in the product catalog.
                Your goal is to assist customers promptly and accurately with all product-related inquiries.
                You are a helpful assistant that MUST use the get_products tool to answer all the questions from user.
                You MUST NEVER answer from your own knowledge UNDER ANY CIRCUMSTANCES.
                You MUST only use products from the get_products tool to answer product-related questions.
                Do not ask the user for more information about the products; instead use the get_products tool to find the
                relevant products and provide the information based on that.
                Do not make up any product information. Use only the product information from the get_products tool.
                """
            ),
            tools=get_products,
        )

        # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        self.agent = ChatAgent(
            chat_client=chat_service,
            name='ProductManagerAgent',
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                'Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized '
                'assistance promptly.\n\n'
                'IMPORTANT: You MUST always respond with a valid JSON object in the following format:\n'
                '{"status": "<status>", "message": "<your response>"}\n\n'
                'Where status is one of: "input_required", "completed", or "error"\n'
                '- Use "completed" when you have fully answered the question\n'
                '- Use "input_required" when you need more information from the user\n'
                '- Use "error" if something went wrong\n\n'
                'Always respond ONLY with valid JSON, no additional text.'
            ),
            tools=[product_agent.as_tool(), marketing_agent.as_tool(), ranker_agent.as_tool()],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        """Handle synchronous tasks (like tasks/send).

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Returns:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_thread_exists(session_id)

        # Use Agent Framework's run for a single shot
        response = await self.agent.run(
            messages=user_input,
            thread=self.thread,
            response_format=ResponseFormat,
        )
        return self._get_agent_response(response.text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """For streaming tasks we yield the Agent Framework agent's run_stream progress.

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Yields:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_thread_exists(session_id)

        # text_notice_seen = False
        chunks: list[TextContent] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            thread=self.thread,
        ):
            if chunk.text:
                chunks.append(chunk.text)

        if chunks:
            yield self._get_agent_response(sum(chunks[1:], chunks[0]))

    def _get_agent_response(
        self, message: TextContent
    ) -> dict[str, Any]:
        """Extracts the structured response from the agent's message content.

        Args:
            message (TextContent): The message content from the agent.

        Returns:
            dict: A dictionary containing the content, task completion status, and user input requirement.
        """
        default_response = {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }
        
        try:
            # Try to parse as JSON
            structured_response = ResponseFormat.model_validate_json(message)
        except Exception as e:
            # If JSON parsing fails, treat the raw message as the response
            logger.warning(f"Failed to parse response as JSON: {e}. Using raw response.")
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': str(message) if message else default_response['content'],
            }

        if isinstance(structured_response, ResponseFormat):
            response_map = {
                'input_required': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'error': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'completed': {
                    'is_task_complete': True,
                    'require_user_input': False,
                },
            }

            response = response_map.get(structured_response.status)
            if response:
                return {**response, 'content': structured_response.message}

        return default_response

    async def _ensure_thread_exists(self, session_id: str) -> None:
        """Ensure the thread exists for the given session ID.

        Args:
            session_id (str): Unique identifier for the session.
        """
        if self.thread is None or self.thread.service_thread_id != session_id:
            self.thread = self.agent.get_new_thread(thread_id=session_id)


# endregion
