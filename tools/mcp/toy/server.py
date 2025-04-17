from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from pathlib import Path


mcp = FastMCP("toy")


kv_storage = {
    "key1": "value1",
    "key2": "value2",
}


@mcp.tool()
def list_directory(path: Path) -> list[str]:
    """
    List the contents of a directory.
    """
    return [str(p) for p in path.iterdir()]


@mcp.resource("data://kv_storage/{key}")
def read_storage(key: str) -> str:
    """
    Get a value from a dictionary.
    """
    return kv_storage.get(key, "not found")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]


if __name__ == "__main__":
    mcp.run()
