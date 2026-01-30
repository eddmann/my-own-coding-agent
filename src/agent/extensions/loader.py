"""Extension loader for loading extensions from Python modules."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from agent.extensions.api import ExtensionAPI


class ExtensionLoader:
    """Load extensions from Python modules.

    Extensions are Python files with a `setup(api: ExtensionAPI)` function.
    The setup function is called with the ExtensionAPI to register handlers.

    Example extension file:
        # my_extension.py
        from agent.extensions import ExtensionAPI, ToolCallResult

        def setup(api: ExtensionAPI):
            def block_dangerous(event, ctx):
                if "rm -rf" in str(event.input):
                    return ToolCallResult(block=True, reason="Blocked dangerous command")
            api.on("tool_call", block_dangerous)
    """

    @staticmethod
    async def load(path: Path, api: ExtensionAPI) -> str | None:
        """Load an extension from a Python file.

        Args:
            path: Path to the extension .py file
            api: ExtensionAPI instance to pass to setup()

        Returns:
            Error message if loading failed, None on success
        """
        if not path.exists():
            return f"Extension file not found: {path}"

        if path.suffix != ".py":
            return f"Extension must be a Python file: {path}"

        try:
            # Generate unique module name to avoid conflicts
            module_name = f"agent_extension_{path.stem}_{id(path)}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return f"Could not load extension spec: {path}"

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Call setup function
            setup = getattr(module, "setup", None)
            if setup is None:
                return f"Extension missing setup() function: {path}"

            if not callable(setup):
                return f"Extension setup is not callable: {path}"

            result = setup(api)
            if inspect.iscoroutine(result):
                await result

            return None  # Success

        except Exception as e:
            return f"Error loading extension {path}: {e}"

    @staticmethod
    async def load_directory(directory: Path, api: ExtensionAPI) -> list[str]:
        """Load all extensions from a directory.

        Args:
            directory: Directory containing .py extension files
            api: ExtensionAPI instance

        Returns:
            List of error messages (empty if all succeeded)
        """
        errors: list[str] = []

        if not directory.exists():
            return [f"Extension directory not found: {directory}"]

        if not directory.is_dir():
            return [f"Not a directory: {directory}"]

        for path in sorted(directory.glob("*.py")):
            # Skip __init__.py and other special files
            if path.name.startswith("_"):
                continue

            error = await ExtensionLoader.load(path, api)
            if error:
                errors.append(error)

        return errors

    @staticmethod
    async def load_multiple(paths: list[Path], api: ExtensionAPI) -> list[str]:
        """Load multiple extension files.

        Args:
            paths: List of paths to extension files
            api: ExtensionAPI instance

        Returns:
            List of error messages (empty if all succeeded)
        """
        errors: list[str] = []

        for path in paths:
            if path.is_dir():
                dir_errors = await ExtensionLoader.load_directory(path, api)
                errors.extend(dir_errors)
            else:
                error = await ExtensionLoader.load(path, api)
                if error:
                    errors.append(error)

        return errors
