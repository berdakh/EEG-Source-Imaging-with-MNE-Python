"""Render Jinja prompt templates from a pack's prompts/ folder."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptRenderer:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render(self, template_name: str, **context) -> str:
        return self.env.get_template(template_name).render(**context)
