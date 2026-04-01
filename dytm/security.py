"""AST-based security scanner for LLM-generated code.

Scans Python source code for dangerous imports, function calls, and
attribute access before allowing execution in the sandbox.  This is the
first line of defense — it runs *before* the code ever reaches subprocess.
"""

from __future__ import annotations

import ast
from typing import Optional

# ---------------------------------------------------------------------------
# Blocklists
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES: set[str] = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "importlib",
    "ctypes",
    "signal",
    "pathlib",
    "tempfile",      # we manage tempfile ourselves in the Doer
    "multiprocessing",
    "threading",
}

FORBIDDEN_CALLS: set[str] = {
    "exec",
    "eval",
    "open",
    "__import__",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
}


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------

class _SecurityVisitor(ast.NodeVisitor):
    """Walks the AST and collects violations."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    # -- import X / import X as Y ------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top_level = alias.name.split(".")[0]
            if top_level in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Forbidden import: '{alias.name}' (line {node.lineno})"
                )
        self.generic_visit(node)

    # -- from X import Y ---------------------------------------------------
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            top_level = node.module.split(".")[0]
            if top_level in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Forbidden import: 'from {node.module} import ...' "
                    f"(line {node.lineno})"
                )
        self.generic_visit(node)

    # -- Bare calls: exec(...), eval(...), open(...) -----------------------
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.violations.append(
                    f"Forbidden call: '{node.func.id}(...)' "
                    f"(line {node.lineno})"
                )
        # Catch os.system(...), subprocess.run(...), etc.
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in FORBIDDEN_MODULES:
                    self.violations.append(
                        f"Forbidden attribute call: "
                        f"'{node.func.value.id}.{node.func.attr}(...)' "
                        f"(line {node.lineno})"
                    )
        self.generic_visit(node)

    # -- Attribute access on blocked modules: os.path, etc. ----------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            if node.value.id in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Forbidden attribute access: "
                    f"'{node.value.id}.{node.attr}' (line {node.lineno})"
                )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_code(code: str) -> tuple[bool, Optional[str]]:
    """Scan Python source code for security violations.

    Args:
        code: A string of Python source code to analyse.

    Returns:
        A ``(safe, reason)`` tuple.  If the code is safe, returns
        ``(True, None)``.  Otherwise returns ``(False, description)``
        where *description* lists every violation found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    visitor = _SecurityVisitor()
    visitor.visit(tree)

    if visitor.violations:
        report = "; ".join(visitor.violations)
        return False, report

    return True, None

