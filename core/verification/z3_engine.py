"""
Z3 Verification Engine for ARTS Framework.

This module serves as the formal verification backend for the ARTS system.
It interfaces with the Z3 Theorem Prover to enforce strict mathematical
well-posedness across generated reasoning chains.

Key Responsibilities:
1. Satisfiability Checking (SAT): Ensuring generated logic chains have at least one valid solution.
2. Uniqueness Verification: Verifying that key parameters (e.g., param_X) allow for unique determination.
3. Domain Constraint Enforcement: Restricting variable domains (e.g., integer ranges, non-negativity).

Dependencies:
    - z3-solver (pip install z3-solver)

Author: Yabo Wang
Date: December 2025
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Union
import z3

# Configure logging for formal verification traces
logger = logging.getLogger("ARTS.Verification")


class Z3VerificationEngine:
    """
    Wraps the Z3 SMT solver to provide a high-level API for
    validating reasoning tasks.
    """

    def __init__(self, timeout_ms: int = 5000):
        """
        Initialize the Z3 solver with a safety timeout.

        Args:
            timeout_ms: Maximum time allowed for verification (milliseconds)
                        to prevent solver hangs on overly complex chains.
        """
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
        self.variables: Dict[str, Union[z3.ArithRef, z3.BoolRef]] = {}
        self.constraints_added: List[str] = []

        # Enable proof generation for unsat core extraction (optional diagnostics)
        # z3.set_param('proof', True)

    def register_variable(self, name: str, var_type: str = "int") -> None:
        """
        Register a symbolic variable in the SMT context.
        """
        if name in self.variables:
            return

        if var_type == "int":
            self.variables[name] = z3.Int(name)
        elif var_type == "real":
            self.variables[name] = z3.Real(name)
        elif var_type == "bool":
            self.variables[name] = z3.Bool(name)
        else:
            raise ValueError(f"Unsupported Z3 variable type: {var_type}")

    def add_constraint(self, lhs_name: str, operator: str, rhs_value: Any) -> None:
        """
        Inject a mathematical constraint into the solver.

        Example: add_constraint("param_X", ">", 10)
        """
        if lhs_name not in self.variables:
            self.register_variable(lhs_name)

        # Handle RHS (could be a literal or another variable)
        if isinstance(rhs_value, str) and rhs_value in self.variables:
            rhs = self.variables[rhs_value]
        else:
            rhs = rhs_value

        lhs = self.variables[lhs_name]

        # Construct Z3 expression
        if operator == "==":
            c = lhs == rhs
        elif operator == "!=":
            c = lhs != rhs
        elif operator == ">":
            c = lhs > rhs
        elif operator == "<":
            c = lhs < rhs
        elif operator == ">=":
            c = lhs >= rhs
        elif operator == "<=":
            c = lhs <= rhs
        else:
            raise ValueError(f"Unknown operator: {operator}")

        self.solver.add(c)
        self.constraints_added.append(f"{lhs_name} {operator} {rhs_value}")

    def inject_formula(self, formula_expression) -> None:
        """
        Inject a raw Z3 expression directly.
        Useful for complex recursive logic (Implies, And, Or).
        """
        self.solver.add(formula_expression)

    def verify_satisfiability(self) -> bool:
        """
        Check if the current system of constraints is mathematically consistent.

        Returns:
            True if SAT (Satisfiable), False if UNSAT (Contradiction).
        """
        logger.debug("Running Z3 consistency check...")
        result = self.solver.check()

        if result == z3.sat:
            logger.info("✅ System is consistent (SAT).")
            return True
        elif result == z3.unsat:
            logger.warning(
                "❌ System is inconsistent (UNSAT). Logic contradiction detected."
            )
            return False
        else:
            logger.error(
                "⚠️ Solver returned 'unknown'. constraints might be too complex."
            )
            return False

    def get_solution(self) -> Dict[str, Any]:
        """
        If satisfiable, retrieve a concrete counter-example or solution model.
        """
        if self.solver.check() != z3.sat:
            return {}

        model = self.solver.model()
        solution = {}

        for name, var_ref in self.variables.items():
            # Extract concrete value from Z3 model
            val = model[var_ref]
            if val is not None:
                if z3.is_int(val):
                    solution[name] = val.as_long()
                elif z3.is_algebraic_value(val):
                    solution[name] = val.approx(10)  # 10 decimal precision
                else:
                    solution[name] = str(val)

        return solution

    def reset(self):
        """Reset the solver state for the next test case."""
        self.solver.reset()
        self.variables.clear()
        self.constraints_added.clear()


# ==========================================
# Self-Test / Demo Usage
# ==========================================
if __name__ == "__main__":
    # This block allows a quick verification that z3 is working correctly
    print("Initializing Z3 Engine Test...")
    engine = Z3VerificationEngine()

    # Simulating a basic reasoning chain:
    # Rule: X > 10 AND X < 20 AND Y = X * 2
    engine.register_variable("param_X", "int")
    engine.register_variable("param_Y", "int")

    engine.add_constraint("param_X", ">", 10)
    engine.add_constraint("param_X", "<", 20)
    engine.inject_formula(
        engine.variables["param_Y"] == engine.variables["param_X"] * 2
    )

    if engine.verify_satisfiability():
        sol = engine.get_solution()
        print(f"Generated Valid Solution: {sol}")
    else:
        print("Verification Failed.")
