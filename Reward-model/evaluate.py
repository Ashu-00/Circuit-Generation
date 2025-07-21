import subprocess
import os

def check_netlist_syntax(netlist_path):
    """
    Checks the syntax of a SPICE netlist using ngspice.

    Args:
        netlist_path (str): The path to the SPICE netlist file (.cir).

    Returns:
        tuple: (bool, str)
               - True if syntax is likely correct, False otherwise.
               - A string containing the combined stdout and stderr from ngspice.
    """
    if not os.path.exists(netlist_path):
        return False, f"Error: Netlist file not found at {netlist_path}"

    command = ['ngspice', '-b', netlist_path]
    
    try:
        # Run ngspice, capture output, and don't raise an exception for non-zero exit codes
        # stderr=subprocess.STDOUT merges stderr into stdout
        result = subprocess.run(command, capture_output=True, text=True, check=False,)
        
        output = result.stdout + result.stderr
        # print(f"NGSPICE Output:\n{output.strip()}")  # Debugging output

        
        # Look for common error indicators in the output
        # ngspice often prints "Error:" or "warning:" (lowercase) for syntax issues
        # It also might print "unknown command", "syntax error", etc.
        error_indicators = ["Error:", "Error", "error:", "warning:", "Warning:", "unknown command", "syntax error"]
        
        is_syntax_correct = True
        for indicator in error_indicators:
            if indicator in output:
                is_syntax_correct = False
                break
        
        # A more robust check might also involve the return code,
        # but ngspice's return codes can be tricky. Parsing output is often more reliable.
        # If check=True was used, a CalledProcessError would be raised for non-zero exit codes.
        # We explicitly set check=False to handle the output ourselves.

        return is_syntax_correct, output

    except FileNotFoundError:
        return False, "Error: 'ngspice' command not found. Make sure ngspice is installed and in your system's PATH."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Create a correct netlist file
    correct_netlist_content = """
    * Simple RC Low-Pass Filter
    V1 1 0 AC 1 SIN(0 1 1k)
    R1 1 2 1k
    C1 2 0 1u
    .tran 1u 5m
    .end
    """
    with open("correct_circuit.cir", "w") as f:
        f.write(correct_netlist_content)

    # 2. Create an incorrect netlist file (with a typo)
    incorrect_netlist_content = """
    * This is a test netlist with a syntax error
    V1 1 0 DC 5V
    R1 1 2 1k
    C1 2 0 1u
    .tran 1u 5m
    .endx  <-- Typo here
    """
    with open("incorrect_circuit.cir", "w") as f:
        f.write(incorrect_netlist_content)

    # 3. Check the correct netlist
    print("--- Checking correct_circuit.cir ---")
    is_correct, output = check_netlist_syntax("correct_circuit.cir")
    print(f"Syntax Correct: {is_correct}")
    if not is_correct:
        print("NGSPICE Output:\n", output)
    else:
        print("No obvious syntax errors detected by ngspice.")

    print("\n--- Checking incorrect_circuit.cir ---")
    # 4. Check the incorrect netlist
    is_correct, output = check_netlist_syntax("incorrect_circuit.cir")
    print(f"Syntax Correct: {is_correct}")
    if not is_correct:
        print("NGSPICE Output:\n", output)
    else:
        print("No obvious syntax errors detected by ngspice.")

    # Clean up created files
    os.remove("correct_circuit.cir")
    os.remove("incorrect_circuit.cir")