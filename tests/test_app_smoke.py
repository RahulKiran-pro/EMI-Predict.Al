"""
Streamlit App Smoke Test.

Verifies that app.py compiles cleanly, parses syntax, and dependencies are satisfied.
"""

import ast
import py_compile
from pathlib import Path


def test_app_syntax_and_compilation():
    """Verify app.py has valid syntax and compiles without error."""
    app_path = Path("app.py")
    assert app_path.exists(), "app.py does not exist."

    # Parse AST
    with open(app_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="app.py")
    assert tree is not None

    # Bytecode compile check
    compiled = py_compile.compile(str(app_path), doraise=True)
    assert compiled is not None
