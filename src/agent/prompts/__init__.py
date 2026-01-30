"""Prompt templates system - dynamic prompts with argument substitution."""

from agent.prompts.loader import PromptTemplateLoader
from agent.prompts.parser import substitute_arguments
from agent.prompts.template import PromptTemplate, TemplateSource

__all__ = [
    "PromptTemplate",
    "TemplateSource",
    "PromptTemplateLoader",
    "substitute_arguments",
]
