"""
TinyForgeAI MCP Server

Model Context Protocol (MCP) integration for TinyForgeAI.
Enables AI assistants (Claude, ChatGPT, Gemini) to interact with
TinyForgeAI's training, inference, and data connector capabilities.

Usage:
    # Run as standalone server
    python -m mcp.server

    # Or use with Claude Desktop
    Add to claude_desktop_config.json:
    {
        "mcpServers": {
            "tinyforge": {
                "command": "python",
                "args": ["-m", "mcp.server"],
                "cwd": "/path/to/TinyForgeAI"
            }
        }
    }
"""

__version__ = "0.1.0"
