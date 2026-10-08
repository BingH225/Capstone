"""Compatibility entry point; the API implementation is in the installed package."""

from smartstress_langgraph.server import app, main

if __name__ == "__main__":
    main()
