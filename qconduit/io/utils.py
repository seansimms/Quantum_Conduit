"""Utility functions for QASM and JSON I/O modules.

This module provides helper functions for parsing QASM angle expressions,
formatting angles for QASM output, normalizing gate names, and parsing
qubit index lists.
"""

from __future__ import annotations

import ast
import math
import re
from typing import List


def angle_str_to_float(s: str) -> float:
    """
    Parse a QASM angle expression to a float value in radians.

    Supports numeric expressions composed of:
    - Decimal numbers (e.g., "1.5", "0.785")
    - Pi multiples (e.g., "pi", "pi/2", "3*pi/4", "-pi")
    - Arithmetic operations: *, /, parentheses, unary minus

    Uses AST parsing for safe evaluation (no eval on arbitrary input).

    Parameters
    ----------
    s : str
        Angle expression string (e.g., "pi/2", "3*pi/4", "0.785").

    Returns
    -------
    float
        Angle value in radians.

    Raises
    ------
    ValueError
        If the expression cannot be parsed or contains disallowed operations.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty angle expression.")

    # Replace pi with a placeholder that AST can handle
    # We'll use a function call pi() that we'll evaluate as math.pi
    # But AST doesn't allow function calls in expressions we want to evaluate
    # So we'll replace pi with a number and track it
    # Actually, simpler: replace pi with the string representation of math.pi
    # But that won't work for expressions like "2*pi"
    # Better: parse the AST and replace Name(id='pi') nodes

    # First, check for disallowed characters (only allow numbers, pi, operators, whitespace)
    allowed_chars = re.compile(r"^[0-9\.\s\*\-/\(\)pi]+$", re.IGNORECASE)
    if not allowed_chars.match(s):
        raise ValueError(
            f"Angle expression contains disallowed characters: {s!r}. "
            "Only numbers, 'pi', '*', '/', '-', '(', ')' are allowed."
        )

    # Replace 'pi' (case-insensitive) with a placeholder number
    # We'll use a large number that's unlikely to appear, then divide by it
    # Actually, simpler: replace pi with math.pi as a string and evaluate
    # But we need to be careful with expressions

    # Parse with AST to check structure
    try:
        # Normalize: replace 'pi' (case-insensitive) with 'PI_PLACEHOLDER'
        normalized = re.sub(r"\bpi\b", "PI_PLACEHOLDER", s, flags=re.IGNORECASE)
        # Replace PI_PLACEHOLDER with math.pi in the AST
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid angle expression syntax: {s!r}. Error: {e}")

    # Walk AST and evaluate safely
    def eval_node(node: ast.AST) -> float:
        """Recursively evaluate AST node, replacing PI_PLACEHOLDER with math.pi."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type in angle expression: {s!r}")

        if isinstance(node, ast.Name):
            if node.id == "PI_PLACEHOLDER":
                return math.pi
            raise ValueError(
                f"Unknown identifier '{node.id}' in angle expression: {s!r}. "
                "Only 'pi' is supported."
            )

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            if isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError(f"Division by zero in angle expression: {s!r}")
                return left / right
            elif isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            else:
                raise ValueError(
                    f"Unsupported operator in angle expression: {s!r}. "
                    "Only *, /, +, - are supported."
                )

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -eval_node(node.operand)
            elif isinstance(node.op, ast.UAdd):
                return eval_node(node.operand)
            else:
                raise ValueError(
                    f"Unsupported unary operator in angle expression: {s!r}"
                )

        # Check for disallowed node types
        if isinstance(node, ast.Call):
            raise ValueError(
                f"Function calls are not allowed in angle expressions: {s!r}"
            )
        if isinstance(node, ast.Attribute):
            raise ValueError(
                f"Attribute access is not allowed in angle expressions: {s!r}"
            )

        raise ValueError(f"Unsupported AST node type in angle expression: {s!r}")

    try:
        result = eval_node(tree.body)
        return float(result)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error evaluating angle expression {s!r}: {e}")


def float_to_angle_str(angle: float, tol: float = 1e-10) -> str:
    """
    Convert a float angle (in radians) to a readable QASM-compatible expression.

    If the angle is a rational multiple of π with small denominator (q ≤ 12),
    returns a simplified expression like "(p/q*pi)" or "(-p/q*pi)".
    Otherwise, returns a decimal string with sufficient precision (12 digits).

    Parameters
    ----------
    angle : float
        Angle in radians.
    tol : float
        Tolerance for detecting rational multiples of π.

    Returns
    -------
    str
        QASM-compatible angle expression.
    """
    # Check if angle is approximately zero
    if abs(angle) < tol:
        return "0"

    # Check if angle is a rational multiple of pi: angle ≈ (p/q) * pi
    # Try denominators from 1 to 12
    pi_multiple = angle / math.pi

    for q in range(1, 13):
        p_float = pi_multiple * q
        p_int = round(p_float)

        if abs(p_float - p_int) < tol:
            p = p_int
            # Simplify fraction
            gcd_val = math.gcd(abs(p), q)
            p_simplified = p // gcd_val
            q_simplified = q // gcd_val

            if q_simplified == 1:
                if p_simplified == 1:
                    return "pi"
                elif p_simplified == -1:
                    return "-pi"
                else:
                    return f"{p_simplified}*pi"
            else:
                if p_simplified == 1:
                    return f"pi/{q_simplified}"
                elif p_simplified == -1:
                    return f"-pi/{q_simplified}"
                else:
                    return f"({p_simplified}/{q_simplified}*pi)"

    # Not a simple rational multiple, return decimal
    # Format with 12 significant digits, but avoid scientific notation for small numbers
    if abs(angle) < 1e-4 or abs(angle) > 1e4:
        # Use scientific notation for very small/large numbers
        return f"{angle:.12e}"
    else:
        # Use fixed-point notation
        return f"{angle:.12f}".rstrip("0").rstrip(".")


def gate_name_normalize(name: str) -> str:
    """
    Normalize gate name to canonical form used in QuantumCircuit.

    Maps common synonyms and case variations to the standard uppercase form:
    - "cx", "CNOT", "cnot" → "CNOT"
    - "h", "H" → "H"
    - "x", "X" → "X"
    - "y", "Y" → "Y"
    - "z", "Z" → "Z"
    - "s", "S" → "S"
    - "t", "T" → "T"
    - "rx", "RX" → "RX"
    - "ry", "RY" → "RY"
    - "rz", "RZ" → "RZ"
    - "u1", "U1" → "U1" (will be decomposed)
    - "u2", "U2" → "U2" (will be decomposed)
    - "u3", "U3" → "U3" (will be decomposed)

    Parameters
    ----------
    name : str
        Gate name (case-insensitive).

    Returns
    -------
    str
        Normalized gate name (uppercase).
    """
    name_upper = name.upper().strip()

    # Map synonyms
    synonym_map = {
        "CX": "CNOT",
        "CNOT": "CNOT",
        "NOT": "X",  # Some QASM variants use NOT for X
    }

    normalized = synonym_map.get(name_upper, name_upper)

    return normalized


def safe_int_list_from_str(s: str) -> List[int]:
    """
    Parse a string representation of a list of integers.

    Handles formats like "[0,1]", "0,1", "[0, 1, 2]", etc.

    Parameters
    ----------
    s : str
        String representation of integer list.

    Returns
    -------
    List[int]
        List of integers.

    Raises
    ------
    ValueError
        If the string cannot be parsed as a list of integers.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty string cannot be parsed as integer list.")

    # Remove brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    # Split by comma
    parts = [p.strip() for p in s.split(",") if p.strip()]

    result = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            raise ValueError(
                f"Cannot parse '{part}' as integer in list string: {s!r}"
            )

    return result


__all__ = [
    "angle_str_to_float",
    "float_to_angle_str",
    "gate_name_normalize",
    "safe_int_list_from_str",
]



