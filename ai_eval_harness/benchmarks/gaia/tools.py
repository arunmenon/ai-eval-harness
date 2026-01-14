"""
GAIA benchmark tools.

This module defines tools available to agents during GAIA task execution,
including web browsing, code execution, and file handling.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ...core.types import ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's definition."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments against definition."""
        errors = []
        for param in self.definition.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"Missing required parameter: {param.name}")
            if param.enum and param.name in arguments:
                if arguments[param.name] not in param.enum:
                    errors.append(f"Invalid value for {param.name}: must be one of {param.enum}")
        return errors


class WebBrowserTool(Tool):
    """
    Web browsing tool for fetching and extracting content from URLs.

    In production, this would use Playwright or similar for full browser
    automation. This implementation provides a simpler HTTP-based approach.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_browser",
            description="Fetch and extract content from a web URL. Returns the page content as text.",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL to fetch",
                    required=True,
                ),
                ToolParameter(
                    name="extract_text",
                    type="boolean",
                    description="Whether to extract only text content (default: true)",
                    required=False,
                    default=True,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        url = kwargs.get("url", "")
        tool_call_id = kwargs.get("tool_call_id", "")
        start_time = time.time()

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.text

                # Basic HTML to text conversion
                if kwargs.get("extract_text", True):
                    content = self._extract_text(content)

                return ToolResult(
                    tool_call_id=tool_call_id,
                    output=content[:50000],  # Limit output size
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                output=None,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _extract_text(self, html: str) -> str:
        """Extract text from HTML (basic implementation)."""
        import re

        # Remove scripts and styles
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        # Decode HTML entities
        import html as html_lib
        text = html_lib.unescape(text)
        return text.strip()


class PythonExecutorTool(Tool):
    """
    Python code execution tool.

    Executes Python code in a subprocess with timeout and output capture.
    For production use, consider Docker-based sandboxing.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        sandbox: bool = False,
        max_output_size: int = 10000,
    ) -> None:
        self.timeout = timeout
        self.sandbox = sandbox
        self.max_output_size = max_output_size

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python",
            description="Execute Python code and return the output. Use print() to output results.",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="The Python code to execute",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        tool_call_id = kwargs.get("tool_call_id", "")
        start_time = time.time()

        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                script_path = f.name

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: subprocess.run(
                            ["python", script_path],
                            capture_output=True,
                            text=True,
                            timeout=self.timeout,
                        ),
                    ),
                    timeout=self.timeout + 5,
                )

                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"

                if len(output) > self.max_output_size:
                    output = output[: self.max_output_size] + "\n... [truncated]"

                return ToolResult(
                    tool_call_id=tool_call_id,
                    output=output,
                    error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            finally:
                # Clean up
                Path(script_path).unlink(missing_ok=True)

        except asyncio.TimeoutError:
            return ToolResult(
                tool_call_id=tool_call_id,
                output=None,
                error=f"Execution timed out after {self.timeout}s",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                output=None,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class BashExecutorTool(Tool):
    """
    Bash command execution tool.

    Executes bash commands with timeout and output capture.
    For production use, consider Docker-based sandboxing.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        sandbox: bool = False,
        max_output_size: int = 10000,
        allowed_commands: list[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.sandbox = sandbox
        self.max_output_size = max_output_size
        self.allowed_commands = allowed_commands

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Execute a bash command and return the output.",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The bash command to execute",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        tool_call_id = kwargs.get("tool_call_id", "")
        start_time = time.time()

        # Check allowed commands if restricted
        if self.allowed_commands:
            cmd_name = command.split()[0] if command else ""
            if cmd_name not in self.allowed_commands:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    output=None,
                    error=f"Command '{cmd_name}' not allowed",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    ),
                ),
                timeout=self.timeout + 5,
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            if len(output) > self.max_output_size:
                output = output[: self.max_output_size] + "\n... [truncated]"

            return ToolResult(
                tool_call_id=tool_call_id,
                output=output,
                error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_call_id=tool_call_id,
                output=None,
                error=f"Execution timed out after {self.timeout}s",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                output=None,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class FileReaderTool(Tool):
    """
    File reading tool for accessing task-attached files.

    Supports reading text files, PDFs (with extraction), and basic
    file metadata.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read the content of a file. Supports text files and PDFs.",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to read",
                    required=True,
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="Text encoding (default: utf-8)",
                    required=False,
                    default="utf-8",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        tool_call_id = kwargs.get("tool_call_id", "")
        encoding = kwargs.get("encoding", "utf-8")
        start_time = time.time()

        try:
            path = Path(file_path)

            # Resolve relative to base_dir if provided
            if self.base_dir and not path.is_absolute():
                path = self.base_dir / path

            if not path.exists():
                return ToolResult(
                    tool_call_id=tool_call_id,
                    output=None,
                    error=f"File not found: {file_path}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Handle different file types
            suffix = path.suffix.lower()

            if suffix == ".pdf":
                content = self._read_pdf(path)
            elif suffix in [".csv", ".tsv"]:
                content = self._read_csv(path)
            elif suffix in [".json"]:
                content = self._read_json(path)
            else:
                content = path.read_text(encoding=encoding)

            # Truncate if too large
            if len(content) > 50000:
                content = content[:50000] + "\n... [truncated]"

            return ToolResult(
                tool_call_id=tool_call_id,
                output=content,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                output=None,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import pypdf

            reader = pypdf.PdfReader(path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())
            return "\n\n".join(text_parts)
        except ImportError:
            return f"[PDF file: {path.name}. Install pypdf to extract text.]"
        except Exception as e:
            return f"[Error reading PDF: {e}]"

    def _read_csv(self, path: Path) -> str:
        """Read CSV/TSV file."""
        try:
            import pandas as pd

            df = pd.read_csv(path)
            return df.to_string()
        except ImportError:
            return path.read_text()
        except Exception:
            return path.read_text()

    def _read_json(self, path: Path) -> str:
        """Read JSON file."""
        import json

        data = json.loads(path.read_text())
        return json.dumps(data, indent=2)


def get_gaia_tools(
    file_base_dir: Path | None = None,
    sandbox_mode: bool = False,
) -> list[Tool]:
    """
    Get the standard set of GAIA tools.

    Args:
        file_base_dir: Base directory for file operations
        sandbox_mode: Whether to enable sandboxing for code execution

    Returns:
        List of Tool instances
    """
    return [
        WebBrowserTool(),
        PythonExecutorTool(sandbox=sandbox_mode),
        BashExecutorTool(sandbox=sandbox_mode),
        FileReaderTool(base_dir=file_base_dir),
    ]
