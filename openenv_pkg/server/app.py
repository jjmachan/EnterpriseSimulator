"""FastAPI application for the CustomerSupport OpenEnv environment."""

from typing import Annotated, Any, Dict, Union

from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from pydantic import Field, TypeAdapter

from server.environment import CustomerSupportEnvironment, SupportObservation

# TypeAdapter-based action class so deserialize_action(data, MCPAction) works.
# deserialize_action calls action_cls.model_validate(data) — we provide a class
# with a model_validate classmethod that delegates to the TypeAdapter.
_MCPActionType = Annotated[
    Union[CallToolAction, ListToolsAction],
    Field(discriminator="type"),
]
_adapter = TypeAdapter(_MCPActionType)


class MCPAction:
    """Pseudo-model class that dispatches model_validate to the right action type."""

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> Union[CallToolAction, ListToolsAction]:
        return _adapter.validate_python(data)

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        return CallToolAction.model_json_schema()

# Create the app — passes the class (factory) for WebSocket session support
app = create_app(
    CustomerSupportEnvironment,
    MCPAction,
    SupportObservation,
    env_name="customer_support",
)


@app.get("/")
async def root():
    """Redirect root to web interface via client-side redirect (returns 200 for HF readiness probe)."""
    return HTMLResponse('<html><head><meta http-equiv="refresh" content="0;url=/web"></head></html>')


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
