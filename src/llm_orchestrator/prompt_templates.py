SYSTEM_PROMPTS = {
    "code_generator": """You are an expert in electrochemical engineering and scientific computing.
Your task is to generate clean, executable Python code for fuel cell and battery modeling.
Generate code that:
- Uses NumPy and SciPy for numerical operations
- Follows physics constraints (dimensional consistency, conservation laws)
- Includes proper parameter bounds and initial guesses
- Is syntactically correct and ready to execute
- Uses descriptive variable names matching electrochemical conventions

Return ONLY the Python code without explanations or markdown formatting.""",

    "physics_aware": """You are an expert in electrochemical systems (PEMFCs, VRFBs).
Generate code that respects fundamental physics:
- Current density i must be less than limiting current i_L
- Activation overpotential η_act ≥ 0
- Transfer coefficient α ∈ (0, 1)
- Temperatures must be in valid range
- No division by zero or log of negative numbers
- Proper unit conversions (mV to V, mA to A, etc.)

Generate clean, executable code without comments.""",

    "solver_focused": """You are an expert in numerical optimization and solving.
Generate code that:
- Uses appropriate solvers (scipy.optimize.least_squares, lsqnonlin)
- Provides good initial guesses
- Sets tolerance parameters explicitly
- Handles convergence failures gracefully
- Scales variables appropriately for numerical stability

Return executable code only."""
}


CODE_GENERATION_TEMPLATES = {
    "pemfc_fitting": """Task: Generate Python code to fit PEMFC polarization data.

Context from literature:
{rag_context}

User requirements:
{user_query}

Operating conditions:
- Temperature: {temperature} °C
- Pressure H2: {p_h2} atm
- Pressure O2: {p_o2} atm

Generate code that:
1. Defines the voltage model: V = E_Nernst - η_act - η_ohm - η_mt
2. Uses Butler-Volmer kinetics for activation losses
3. Fits parameters (i0, alpha, R_ohm, i_L) to data
4. Returns fitted parameters and RMSE

Data available as: current_density (A/cm²), voltage (V)""",

    "vrfb_optimization": """Task: Generate Python code to optimize VRFB design.

Context from literature:
{rag_context}

User requirements:
{user_query}

Design variables:
- Electrode thickness δ_e (mm)
- Flow rate Q (mL/s)
- Operating current density i (A/cm²)

Generate code that:
1. Computes voltage efficiency η_V
2. Calculates pumping power P_pump
3. Optimizes trade-off: max(η_V - β*P_pump/P_elec)
4. Enforces constraints: i < i_L, ΔP < ΔP_max
5. Returns optimal design parameters""",

    "equation_derivation": """Task: Derive and implement governing equations.

Context from literature:
{rag_context}

Requirements:
{user_query}

Generate code that:
1. Implements the governing equations
2. Ensures dimensional consistency
3. Applies boundary conditions
4. Uses appropriate numerical methods
5. Returns solution with diagnostics"""
}


FEEDBACK_TEMPLATES = {
    "syntax_error": """The generated code has syntax errors:
{errors}

Please fix these syntax issues and regenerate valid Python code.""",

    "dimensional_error": """The code has dimensional inconsistencies:
{errors}

Ensure all terms in equations have compatible units. Add unit conversions where needed.
Regenerate the corrected code.""",

    "physics_violation": """The code violates physics constraints:
{errors}

Fix these physics violations:
- Ensure i < i_L checks
- Keep α ∈ (0, 1)
- Verify η_act ≥ 0
- Add proper bounds
Regenerate the corrected code.""",

    "numerical_instability": """The code may have numerical stability issues:
{warnings}

Improve numerical stability:
- Add variable scaling
- Set explicit tolerances
- Provide good initial guesses
- Handle edge cases
Regenerate the improved code.""",

    "general_feedback": """The code validation found issues:

Errors:
{errors}

Warnings:
{warnings}

Please address these issues and regenerate corrected code."""
}


REFINEMENT_PROMPTS = {
    "add_constraints": """Add the following physics constraints to the code:
{constraints}

Modify the code to enforce these constraints.""",

    "improve_stability": """Improve numerical stability by:
1. Scaling variables to O(1)
2. Using appropriate solver tolerances
3. Providing better initial guesses

Regenerate the improved code.""",

    "fix_units": """Fix unit inconsistencies:
{unit_issues}

Add proper unit conversions and regenerate."""
}


def get_code_generation_prompt(
    task_type: str,
    rag_context: str,
    user_query: str,
    **kwargs
) -> str:
    template = CODE_GENERATION_TEMPLATES.get(task_type, CODE_GENERATION_TEMPLATES["equation_derivation"])
    return template.format(
        rag_context=rag_context,
        user_query=user_query,
        **kwargs
    )


def get_feedback_prompt(
    validation_result,
    attempt_number: int
) -> str:
    if validation_result.syntax_errors:
        return FEEDBACK_TEMPLATES["syntax_error"].format(
            errors="\n".join(validation_result.syntax_errors)
        )
    
    if validation_result.dimensional_errors:
        return FEEDBACK_TEMPLATES["dimensional_error"].format(
            errors="\n".join(validation_result.dimensional_errors)
        )
    
    if validation_result.physics_errors:
        return FEEDBACK_TEMPLATES["physics_violation"].format(
            errors="\n".join(validation_result.physics_errors)
        )
    
    return FEEDBACK_TEMPLATES["general_feedback"].format(
        errors="\n".join(validation_result.violations),
        warnings="\n".join(validation_result.warnings)
    )


def get_system_prompt(mode: str = "physics_aware") -> str:
    return SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["physics_aware"])
