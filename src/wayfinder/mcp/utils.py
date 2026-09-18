from wayfinder.mcp.types import ToolSuccess, ToolError, T

# Convenience function to create error responses
def tool_error(message: str, code: str, **details) -> ToolError:
    """Create a standardized tool error response."""
    return ToolError(error=message, error_code=code, details=details)


def tool_success(data: T) -> ToolSuccess[T]:
    """Create a standardized tool success response."""
    return ToolSuccess(data=data)
