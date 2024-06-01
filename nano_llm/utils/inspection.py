#!/usr/bin/env python3
import inspect

from types import FunctionType, MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

# https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/function_calling.py
# https://github.com/langchain-ai/langchain/commit/7a5e1bcf99ab445cf9438f5412f19e5024e5e123
def _get_python_function_name(function: Callable) -> str:
    """Get the name of a Python function."""
    return function.__name__
    
def _parse_python_function_docstring(function: Callable) -> Tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    docstring = inspect.getdoc(function)
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            elif block.startswith("Returns:") or block.startswith("Example:"):
                # Don't break in case Args come after
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors)
    else:
        description = ""
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":", maxsplit=1)
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += " " + line.strip()
    return description, arg_descriptions
    
def _get_python_function_arguments(function: Callable, arg_descriptions: dict) -> dict:
    """Get JsonSchema describing a Python functions arguments.

    Assumes all function arguments are of primitive types (int, float, str, bool) or
    are subclasses of pydantic.BaseModel.
    """
    properties = {}
    annotations = inspect.getfullargspec(function).annotations
    for arg, arg_type in annotations.items():
        if arg == "return":
            continue
        #if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
            # Mypy error:
            # "type" has no attribute "schema"
            #properties[arg] = arg_type.schema()  # type: ignore[attr-defined]
        #elif (
        if(
            hasattr(arg_type, "__name__")
            and getattr(arg_type, "__name__") in PYTHON_TO_JSON_TYPES
        ):
            properties[arg] = {"type": PYTHON_TO_JSON_TYPES[arg_type.__name__]}
        elif (
            hasattr(arg_type, "__dict__")
            and getattr(arg_type, "__dict__").get("__origin__", None) == Literal
        ):
            properties[arg] = {
                "enum": list(arg_type.__args__),  # type: ignore
                "type": PYTHON_TO_JSON_TYPES[arg_type.__args__[0].__class__.__name__],  # type: ignore
            }
        if arg in arg_descriptions:
            if arg not in properties:
                properties[arg] = {}
            properties[arg]["description"] = arg_descriptions[arg]
    return properties
    
def _get_python_function_required_args(function: Callable) -> List[str]:
    """Get the required arguments for a Python function."""
    spec = inspect.getfullargspec(function)
    required = spec.args[: -len(spec.defaults)] if spec.defaults else spec.args
    required += [k for k in spec.kwonlyargs if k not in (spec.kwonlydefaults or {})]

    is_function_type = isinstance(function, FunctionType)
    is_method_type = isinstance(function, MethodType)
    if required and is_function_type and required[0] == "self":
        required = required[1:]
    elif required and is_method_type and required[0] == "cls":
        required = required[1:]
    return required
    
    
def convert_to_openai_tool(
    function: Callable,
) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.
    """
    description, arg_descriptions = _parse_python_function_docstring(function)
    return {
        "name": _get_python_function_name(function),
        "description": inspect.getdoc(function), #description, 
        "parameters": {
            "type": "object",
            "properties": _get_python_function_arguments(function, arg_descriptions),
            "required": _get_python_function_required_args(function),
        },
    }
          
def get_class_that_defined_method(meth):
    """
    Given a function or method, return the class type it belongs to
    https://stackoverflow.com/a/25959545
    """
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return None  # not required since None would have been implicitly returned anyway
    
def function_has_kwargs(func):
    """
    Return true if the function accepts kwargs, otherwise false.
    """
    return 'kwargs' in inspect.signature(func).parameters
    
