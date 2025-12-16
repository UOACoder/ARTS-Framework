import os
import sys
import json
import glob
import time
import re
from typing import Dict, Any, Optional

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    "model_name": "gemini-2.5-pro",  # Change to "gpt-4o" or "claude-3-5-sonnet" as needed
    "examples_dir": "examples",
    "required_env_vars": ["2"],  # Adjust based on model choice
}

GENERAL_SYSTEM_PROMPT = """You are operating in a sandboxed, formal reasoning environment. Your assigned role is a **Pure Logic Processor**.

**Core Directives:**
1.  **Reject Intuition and Analogy:** Your primary directive is to completely suppress any pattern-matching, statistical correlation, or real-world knowledge. The variable names (e.g., ZIKLO, BLAF) are intentionally nonsensical and have no connection to your training data. Any attempt to use semantic shortcuts will lead to incorrect results.
2.  **Embrace Deterministic Simulation:** You must function as a perfect, step-by-step Python interpreter. Your task is to mentally simulate the execution of the provided script with absolute precision, tracking the state of each variable as it is computed. The logic flow defined in the code is the only truth.
3.  **No Extrapolation:** Do not infer any rules, relationships, or values that are not explicitly stated in the code. This is a closed-world problem.

**Task Specification:**
-   **Input:** You will receive a self-contained Python script.
-   **Process:** Execute the script's final line, which is a function call.
-   **Output:** The function's return value will be a Python dictionary. Your entire response MUST be ONLY a valid JSON object that is the direct, exact equivalent of that final dictionary.

Do not include any preamble, explanation, conversational text, or markdown formatting.
"""

# Ensure root directory is in python path
sys.path.append(os.getcwd())

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ==========================================
# Core Imports
# ==========================================
try:
    from core.models.factory import ModelFactory
except ImportError as e:
    print(f"❌ Critical Error: Failed to import ARTS Core modules.\nDetails: {e}")
    sys.exit(1)


# ==========================================
# Helpers
# ==========================================
def load_test_case() -> Dict[str, Any]:
    """
    Scans the examples directory and loads the first available test case.
    Handles both 'single case' and 'dataset list' JSON structures.
    """
    pattern = os.path.join(CONFIG["examples_dir"], "*.json")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No JSON files found in {CONFIG['examples_dir']}/")

    target_file = files[0]
    print(f"📂 Loading data from: {os.path.basename(target_file)}")

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Heuristic: If it contains a list of cases, pick the first one
    if isinstance(data, dict) and "test_cases" in data:
        print(
            f"   ℹ️  Dataset detected ({len(data['test_cases'])} items). Using index 0."
        )
        return data["test_cases"][0]
    elif isinstance(data, list) and len(data) > 0:
        return data[0]

    return data


def robust_json_parse(text: str) -> Dict[str, Any]:
    """
    Extracts JSON from LLM output, handling Markdown fences and raw strings.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback 1: Markdown code blocks
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Fallback 2: Brute-force brace matching
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

    return {"error": "JSON_PARSE_FAILED", "raw_output": text}


def validate_result(golden: Dict[str, Any], predicted: Dict[str, Any]):
    """
    Compares model output against the golden answer (Ground Truth).
    """
    print("🏆 Validation Results:")

    # 1. Check 'param_X' (Inverse Reasoning Target)
    if "param_X" in golden:
        exp, got = golden["param_X"], predicted.get("param_X")
        is_match = str(exp) == str(got)
        icon = "✅" if is_match else "❌"
        print(f"   {icon} param_X: Expected {exp} | Got {got}")

    # 2. Check 'predicted_output_B' (Execution Target)
    # Handle nested 'predicted' dictionary in golden answer
    golden_pred = golden.get("predicted", {})
    for k, v in golden_pred.items():
        # Flattened lookup: check if key exists in root or nested 'predicted'
        got = predicted.get(k) or predicted.get("predicted", {}).get(k)
        is_match = str(v) == str(got)
        icon = "✅" if is_match else "❌"
        print(f"   {icon} {k}: Expected {v} | Got {got}")


# ==========================================
# Main Execution
# ==========================================
def main():
    print(f"\n🚀 ARTS Framework | Diagnostic Demo")
    print("=" * 50)

    # 1. Initialize Model
    model_name = CONFIG["model_name"]
    print(f"🤖 Initializing Model: {model_name}")

    try:
        model = ModelFactory.create(model_name)
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        print("   -> Check .env for API keys or requirements.txt")
        return

    # 2. Load Data
    try:
        case = load_test_case()
    except Exception as e:
        print(f"❌ Data Load Failed: {e}")
        return

    # 3. Prepare Inference
    code_snippet = case.get("code", "")
    if not code_snippet:
        print("❌ Error: Valid 'code' field missing in test case.")
        return

    print(f"🧪 Case ID: {case.get('id', 'Unknown')}")
    print(f"📝 Input Code (Preview):\n{'-'*30}\n{code_snippet[:200]}...\n{'-'*30}")

    prompt = [
        {
            "role": "system",
            "content": GENERAL_SYSTEM_PROMPT,
        },
        {"role": "user", "content": code_snippet},
    ]

    # 4. Run Inference
    print("⏳ Running Inference...")
    t0 = time.perf_counter()

    try:
        raw_response = model.call(messages=prompt)
        duration = time.perf_counter() - t0

        # Parse
        if hasattr(model, "parse_response"):
            result = model.parse_response(raw_response) or robust_json_parse(
                raw_response
            )
        else:
            result = robust_json_parse(raw_response)

        print(f"✅ Completed in {duration:.2f}s")
        print("=" * 50)
        print(json.dumps(result, indent=2))
        print("=" * 50)

        # 5. Validation
        if "golden_answer" in case:
            validate_result(case["golden_answer"], result)
        else:
            print("ℹ️  No golden answer found for validation.")

    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
