import json
from pathlib import Path
from typing import Any
from uuid import uuid4

REACT_DIR = Path(__file__).parent.parent.parent.parent / "token-trace-ui"


class RenderedHTML:
    """Rendered HTML

    Enables rendering HTML in a variety of situations (e.g. Jupyter Lab)
    """

    def __init__(self, src: str):
        self.src = src

    def _repr_html_(self) -> str:
        """Jupyter/Colab HTML Representation

        When Jupyter sees this method, it renders the HTML.

        Returns:
            str: HTML for Jupyter/Colab
        """

        return self.src

    def __html__(self) -> str:
        """Used by some tooling as an alternative to _repr_html_"""
        return self._repr_html_()

    def show_code(self) -> str:
        """Show the code as HTML source code

        This loads JavaScript from the CDN, so it will not work offline.

        Returns:
            str: HTML source code (with JavaScript from CDN)
        """
        return self.src

    def __str__(self):
        """String type conversion handler

        Returns:
            str: HTML source code (with JavaScript from CDN)
        """
        return self.src


def render(react_element_name: str, **kwargs: Any) -> str:
    """Render

    Args:
        react_element_name (str): Name of the React element to render

    Returns:
        RenderedHTML: HTML for the visualization
    """
    uuid = "circuits-vis-" + str(uuid4())[:13]

    # Stringify keyword args
    props = json.dumps(kwargs)

    js_filename = Path(__file__).parent.parent.parent / "token-trace-ui" / "iife.js"
    with open(js_filename, encoding="utf-8") as file:
        inline_js = file.read()
        # Remove any closing script tags (as this breaks inline code)
        inline_js = inline_js.replace("</script>", "")

    html = f"""<div id="{uuid}" style="margin: 15px 0;"/>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script crossorigin type="module">
    {inline_js}
    
    TokenTrace.render(
      "{uuid}",
      TokenTrace.{react_element_name},
      {props}
    )
    </script>"""

    return html


def display_token_trace(
    tokens: list[str],
    layer_vals: list[list[list[tuple[int, float]]]],
    hide_boxes: bool = False,
    hide_bars: bool = False,
) -> RenderedHTML:
    return RenderedHTML(
        render(
            "TokenTrace",
            tokens=tokens,
            layerVals=layer_vals,
            hideBoxes=hide_boxes,
            hideBars=hide_bars,
        )
    )
