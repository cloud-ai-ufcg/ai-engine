from fastapi import FastAPI, Body
import uvicorn
from typing import List, Dict, Any
import json

app = FastAPI(
    title="Actuator API",
    description="Fake Actuator that accepts recommendations and logs them",
    version="1.0.0",
)


@app.post("/apply")
async def apply_recommendations(recommendations: List[Dict[str, Any]] = Body(...)) -> Dict[str, str]:
    """Endpoint that receives a list of recommendations and simply logs them.

    Args:
        recommendations: List of recommendation objects coming from the AI Engine.
    Returns:
        Confirmation message.
    """
    # Pretty-print recommendations to stdout so the user can see them.
    formatted = json.dumps(recommendations, indent=2, ensure_ascii=False)
    print("Received recommendations:\n", formatted)

    return {"status": "success", "message": "Recommendations received"}


if __name__ == "__main__":
    uvicorn.run("fake_actuator:app", host="0.0.0.0", port=8084, reload=True)
