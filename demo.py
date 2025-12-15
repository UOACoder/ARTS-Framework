import os
import sys
import json
import re
import time
from typing import Dict, Any

# ==========================================
# 1. Environment Setup & Path Configuration
# ==========================================
# Add current directory to path to ensure 'core' package is discoverable
sys.path.append(os.getcwd())

# Try to load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Loaded .env configuration")
except ImportError:
    pass

# ==========================================
# 2. Import Model Factory
# ==========================================
try:
    from core.models.factory import ModelFactory

    print("✅ Successfully imported core.models.factory")
except ImportError as e:
    print("❌ Import failed. Please check your directory structure.")
    print(f"Error details: {e}")
    print("Expected location: ./core/models/factory.py")
    sys.exit(1)

# ==========================================
# 3. Mock Test Data
# ==========================================
# A representative ARTS test case containing recursive logic and execution traces.
# This allows the demo to run without external JSON files.
DEMO_CASE = {
    "id": "demo_hierarchical_reasoning_001",
    "code": """
_var_trace = {}
trace = _var_trace

def system_B(val_init, param_X):
    # Local trace scope
    trace = {} 
    
    # Layer 1: Simple calculation
    leaf_1 = val_init + 10
    trace['leaf_1_local'] = leaf_1
    
    # Layer 0: Recursive logic
    val_final = leaf_1 * param_X
    trace['val_final_local'] = val_final
    
    return trace

# Global execution
output_A = 500
trace['output_A'] = output_A

param_X = ... # To be solved
trace['param_X'] = param_X

# Execution
predicted_trace = system_B(val_init=40, param_X=param_X)
output_B = predicted_trace['val_final_local']
trace['predicted_output_B'] = output_B

# [CRITICAL] Explicit merge: Merging internal traces into global scope
trace.update(predicted_trace)

final_result = trace
""",
    "golden_answer": {
        "param_X": 10,  # Calculation: (40+10)*10 = 500
        "predicted": {"predicted_output_B": 500},
    },
}


# ==========================================
# 4. Utility: Robust JSON Parser
# ==========================================
def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    A standalone JSON parser to handle LLM responses.
    Can extract JSON from Markdown code blocks (```json ... ```).
    """
    try:
        # 1. Try parsing directly
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from ```json ... ``` blocks
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # 3. Try finding the first brace pair { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {"error": "Failed to parse JSON", "raw": text}


# ==========================================
# 5. Main Execution Logic
# ==========================================
def run_demo():
    print("=" * 60)
    print("🚀 ARTS Framework - Quick Start Demo")
    print("=" * 60)

    # --- Configuration ---
    # You can change the model here, e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-2.5-pro"
    # Ensure the corresponding API key is set in your .env file
    MODEL_NAME = "gemini-2.5-pro"

    # Basic check for API Key (Example for Gemini)
    if "gemini" in MODEL_NAME.lower() and not os.getenv("GOOGLE_API_KEY"):
        print(f"⚠️ WARNING: GOOGLE_API_KEY not found. Model initialization might fail.")

    print(f"🤖 Initializing Model: {MODEL_NAME}...")
    try:
        model = ModelFactory.create(MODEL_NAME)
    except Exception as e:
        print(f"❌ Model Initialization Failed: {e}")
        print(
            "💡 Hint: Ensure 'requirements.txt' is installed and API keys are set in .env"
        )
        return

    print(f"🧪 Loading Test Case: {DEMO_CASE['id']}")
    print(f"📝 Code Snippet:\n{'-'*20}\n{DEMO_CASE['code'][:150]}...\n{'-'*20}")

    # Construct Prompt
    prompt = [
        {
            "role": "system",
            "content": "You are a Python code analyzer. Predict the execution trace and final result. Return ONLY a JSON object.",
        },
        {"role": "user", "content": DEMO_CASE["code"]},
    ]

    print("Thinking... (This may take a few seconds)")
    start_time = time.time()

    try:
        # Call the Model
        response_text = model.call(messages=prompt)
        duration = time.time() - start_time

        # Parse Response
        # Try using the model's built-in parser if available, fallback to local parser
        if hasattr(model, "parse_response"):
            try:
                result_json = model.parse_response(response_text)
            except:
                result_json = safe_parse_json(response_text)
        else:
            result_json = safe_parse_json(response_text)

        print(f"✅ Execution Complete (Time: {duration:.2f}s)")
        print("=" * 60)
        print("📊 Model Output (Parsed JSON):")
        print(json.dumps(result_json, indent=2, ensure_ascii=False))
        print("=" * 60)

        # Simple Validation
        expected_param_X = DEMO_CASE["golden_answer"]["param_X"]
        if result_json.get("param_X") == expected_param_X:
            print(
                f"🎉 Validation PASSED! Model correctly identified param_X={expected_param_X}."
            )
        else:
            print(
                f"🤔 Result Mismatch. Expected param_X={expected_param_X}, got {result_json.get('param_X')}"
            )

    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
