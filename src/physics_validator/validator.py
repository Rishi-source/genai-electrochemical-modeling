import ast
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    syntax_errors: List[str] = field(default_factory=list)
    dimensional_errors: List[str] = field(default_factory=list)
    physics_errors: List[str] = field(default_factory=list)
    
    def add_violation(self, category: str, message: str):
        self.violations.append(f"[{category}] {message}")
        self.is_valid = False
        
        if category == "syntax":
            self.syntax_errors.append(message)
        elif category == "dimensional":
            self.dimensional_errors.append(message)
        elif category == "physics":
            self.physics_errors.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_violations": len(self.violations),
            "syntax_errors": len(self.syntax_errors),
            "dimensional_errors": len(self.dimensional_errors),
            "physics_errors": len(self.physics_errors),
            "warnings": len(self.warnings)
        }


class PhysicsValidator:
    """
    Implements the "Physics-Constrained Validation" module.
    
    Responsible for ensuring generated code adheres to:
    1. Syntactic correctness (AST parsing)
    2. Dimensional consistency (Unit checking)
    3. Physics constraints (e.g., i < i_L, alpha in [0,1], T > 0)
    4. Numerical stability (e.g., no log(negative), reasonable exponents)
    
    As described in Section 'Physics-Informed Verification' of the paper.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        
        self.unit_conversions = {
            'mV': 0.001,  
            'mA': 0.001,  
            'mA/cm2': 0.01,  
            'mW': 0.001,  
            'mm': 0.001,  
            'cm': 0.01,  
            'mL': 1e-6,  
            'C': 1.0,  
            'atm': 101325,  
        }
        
        
        self.physics_bounds = {
            'current_density': (0, 5.0),  
            'voltage': (0, 2.0),  
            'temperature_C': (-50, 200),  
            'pressure_atm': (0, 10),  
            'exchange_current': (1e-10, 1e-2),  
            'transfer_coefficient': (0.1, 0.9),  
            'resistance': (0, 10),  
        }
    
    def validate(self, code: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if self.verbose:
            print("Validating generated code...")
        
        
        self._check_syntax(code, result)
        
        
        self._check_dimensional_consistency(code, result)
        
        
        self._check_physics_constraints(code, context or {}, result)
        
        
        self._check_numerical_stability(code, result)
        
        if self.verbose:
            summary = result.get_summary()
            if result.is_valid:
                print(f"✓ Validation passed ({summary['warnings']} warnings)")
            else:
                print(f"✗ Validation failed ({summary['total_violations']} violations)")
        
        return result
    
    def _check_syntax(self, code: str, result: ValidationResult):
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.add_violation("syntax", f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result.add_violation("syntax", f"Parsing error: {str(e)}")
    
    def _check_dimensional_consistency(self, code: str, result: ValidationResult):
        unit_pattern = r'(\w+)\s*=\s*([^\s#]+)\s*#?\s*(\w+)?'
        matches = re.findall(unit_pattern, code)

        variables_with_units = {}
        for match in matches:
            if len(match) >= 3 and match[2]:
                var, value, unit = match[0], match[1], match[2]
                variables_with_units[var] = unit.strip()
        
        
        
        incompatible_ops = [
            (r'voltage\s*\+\s*current', "Cannot add voltage and current"),
            (r'current\s*\+\s*voltage', "Cannot add current and voltage"),
            (r'power\s*\+\s*voltage', "Cannot add power and voltage"),
        ]
        
        for pattern, msg in incompatible_ops:
            if re.search(pattern, code, re.IGNORECASE):
                result.add_violation("dimensional", msg)
        
        
        if 'mV' in code and 'V' in code:
            if not any(conv in code for conv in ['* 0.001', '/ 1000', 'mV_to_V']):
                result.add_warning("Potential unit mixing: mV and V both present")
        
        if 'mA' in code and 'A' in code:
            if not any(conv in code for conv in ['* 0.001', '/ 1000', 'mA_to_A']):
                result.add_warning("Potential unit mixing: mA and A both present")
    
    def _check_physics_constraints(
        self,
        code: str,
        context: Dict[str, Any],
        result: ValidationResult
    ):
        params = context.get('parameters', {})
        
        
        if 'i_L' in code or 'limiting_current' in code:
            
            if not re.search(r'i\s*<\s*i_L', code):
                if not re.search(r'i_L\s*>\s*i', code):
                    result.add_violation(
                        "physics",
                        "Missing check: i < i_L (current must be below limiting current)"
                    )
        
        
        if 'eta_act' in code or 'η_act' in code:
            
            if not re.search(r'eta_act\s*[><=]\s*0', code):
                result.add_warning("No explicit check for eta_act ≥ 0")
        
        
        if 'E_Nernst' in code or 'E_nernst' in code:
            if not re.search(r'E_[Nn]ernst\s*[><=]\s*0', code):
                result.add_warning("No explicit check for E_Nernst > 0")
        
        
        alpha_pattern = r'alpha\s*=\s*([\d.]+)'
        alpha_match = re.search(alpha_pattern, code)
        if alpha_match:
            try:
                alpha = float(alpha_match.group(1))
                if alpha <= 0 or alpha >= 1:
                    result.add_violation(
                        "physics",
                        f"Transfer coefficient α={alpha} out of bounds (0, 1)"
                    )
            except:
                pass
        
        
        temp_pattern = r'T(?:emp)?\s*=\s*([\d.]+)'
        temp_match = re.search(temp_pattern, code)
        if temp_match:
            try:
                T = float(temp_match.group(1))
                if T < 0:
                    result.add_violation("physics", f"Temperature T={T} cannot be negative")
                if T > 0 and T < 50:
                    
                    if T < 273:
                        result.add_warning(f"Temperature T={T} K is below freezing")
            except:
                pass
        
        
        log_patterns = [
            r'log\s*\(\s*([^)]+)\)',
            r'np\.log\s*\(\s*([^)]+)\)',
        ]
        for pattern in log_patterns:
            matches = re.findall(pattern, code)
            for arg in matches:
                
                if '-' in arg and not any(op in arg for op in ['abs', 'max', 'clip']):
                    result.add_warning(f"Potential log of negative value: log({arg})")
        
        
        div_pattern = r'/\s*([^/\n;]+)'
        div_matches = re.findall(div_pattern, code)
        for denominator in div_matches:
            denom = denominator.strip()
            if denom in ['0', '0.0', '0.']:
                result.add_violation("physics", f"Division by zero: /{denom}")
    
    def _check_numerical_stability(self, code: str, result: ValidationResult):        
        exp_pattern = r'exp\s*\(\s*([^)]+)\)'
        exp_matches = re.findall(exp_pattern, code)
        for arg in exp_matches:
            
            const_match = re.search(r'([\d.]+)', arg)
            if const_match:
                try:
                    val = float(const_match.group(1))
                    if abs(val) > 50:
                        result.add_warning(f"Large exponent: exp({arg}) may overflow")
                except:
                    pass
        
        
        solver_funcs = ['lsqnonlin', 'least_squares', 'fmincon', 'minimize']
        for func in solver_funcs:
            if func in code:
                if not any(tol in code for tol in ['tol', 'tolerance', 'rtol', 'atol']):
                    result.add_warning(f"Solver {func} called without explicit tolerance")
        
        
        if 'least_squares' in code or 'lsqnonlin' in code:
            if not re.search(r'x0\s*=', code):
                result.add_warning("No initial guess (x0) specified for optimization")
    
    def validate_parameters(
        self,
        params: Dict[str, float],
        param_type: str = "PEMFC"
    ) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        
        if param_type == "PEMFC":
            if 'i0' in params:
                i0 = params['i0']
                bounds = self.physics_bounds['exchange_current']
                if not (bounds[0] <= i0 <= bounds[1]):
                    result.add_violation(
                        "physics",
                        f"Exchange current i0={i0:.2e} A/cm² out of bounds {bounds}"
                    )
            
            if 'alpha' in params:
                alpha = params['alpha']
                bounds = self.physics_bounds['transfer_coefficient']
                if not (bounds[0] <= alpha <= bounds[1]):
                    result.add_violation(
                        "physics",
                        f"Transfer coefficient α={alpha} out of bounds {bounds}"
                    )
            
            if 'R_ohm' in params:
                R = params['R_ohm']
                if R < 0:
                    result.add_violation("physics", f"Resistance R_Ω={R} cannot be negative")
                if R > self.physics_bounds['resistance'][1]:
                    result.add_warning(f"Unusually high resistance: R_Ω={R} Ω·cm²")
            
            if 'i_L' in params:
                i_L = params['i_L']
                if i_L <= 0:
                    result.add_violation("physics", f"Limiting current i_L={i_L} must be positive")
        
        
        elif param_type == "VRFB":
            if 'delta_e' in params:
                delta_e = params['delta_e']
                if delta_e <= 0 or delta_e > 20:
                    result.add_violation(
                        "physics",
                        f"Electrode thickness δₑ={delta_e} mm out of reasonable range (0, 20)"
                    )
            
            if 'Q' in params:
                Q = params['Q']
                if Q <= 0 or Q > 100:
                    result.add_violation(
                        "physics",
                        f"Flow rate Q={Q} mL/s out of reasonable range (0, 100)"
                    )
        
        return result
    
    def get_violation_rate(self, results: List[ValidationResult]) -> float:
        if not results:
            return 0.0
        
        invalid_count = sum(1 for r in results if not r.is_valid)
        return invalid_count / len(results)
    
    def get_error_breakdown(
        self,
        results: List[ValidationResult]
    ) -> Dict[str, float]:
        if not results:
            return {}
        
        n = len(results)
        return {
            "syntax_error_rate": sum(len(r.syntax_errors) > 0 for r in results) / n,
            "dimensional_error_rate": sum(len(r.dimensional_errors) > 0 for r in results) / n,
            "physics_error_rate": sum(len(r.physics_errors) > 0 for r in results) / n,
            "any_violation_rate": sum(not r.is_valid for r in results) / n,
        }


def main():
    print("="*70)
    print("Testing Physics Validator")
    print("="*70)
    
    validator = PhysicsValidator(verbose=True)
    
    
    print("\n1. Valid PEMFC code:")
    valid_code = """
import numpy as np


i0 = 1e-7  
alpha = 0.5  
R_ohm = 0.15  
i_L = 2.0  


if i < i_L:
    eta_act = (R * T / (alpha * F)) * np.log(i / i0)
    V = E_Nernst - eta_act - i * R_ohm
"""
    result = validator.validate(valid_code)
    print(f"Result: {result.get_summary()}")
    
    
    print("\n2. Code with syntax error:")
    invalid_syntax = """
def calculate_voltage(
    
"""
    result = validator.validate(invalid_syntax)
    print(f"Result: {result.get_summary()}")
    print(f"Violations: {result.violations}")
    
    
    print("\n3. Code with physics violation:")
    physics_violation = """
alpha = 1.5  
i_L = -0.5  
"""
    result = validator.validate(physics_violation)
    print(f"Result: {result.get_summary()}")
    print(f"Violations: {result.violations}")
    
    
    print("\n4. Parameter validation:")
    params_valid = {'i0': 1e-7, 'alpha': 0.5, 'R_ohm': 0.15}
    result = validator.validate_parameters(params_valid, "PEMFC")
    print(f"Valid params: {result.is_valid}")
    
    params_invalid = {'i0': 1e-1, 'alpha': 1.5, 'R_ohm': -0.1}
    result = validator.validate_parameters(params_invalid, "PEMFC")
    print(f"Invalid params: {result.is_valid}")
    print(f"Violations: {result.violations}")
    
    print("\n✓ Physics validator tests complete")


if __name__ == "__main__":
    main()
