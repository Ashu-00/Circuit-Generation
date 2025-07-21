import random
import re

def inject_error(netlist_content, error_type=None, num_errors=1):
    """
    Injects a specified type of error (or a random type) into a SPICE netlist.

    Args:
        netlist_content (str): The content of the SPICE netlist as a string.
        error_type (str, optional): The type of error to inject. 
                                    Can be 'syntax', 'structural', 'value', 'model_missing', 'random'.
                                    If None or 'random', a random error type is chosen.
        num_errors (int, optional): The number of errors to inject. Defaults to 1.

    Returns:
        str: The netlist content with errors injected.
        list: A list of injected error descriptions.
    """

    lines = netlist_content.strip().split('\n')
    injected_errors = []

    error_types = ['syntax', 'structural', 'value', 'model_missing']

    if error_type and error_type != 'random' and error_type not in error_types:
        raise ValueError(f"Invalid error_type: {error_type}. Must be one of {error_types} or 'random'.")

    # Keep track of indices to avoid injecting errors into already modified lines
    available_line_indices = list(range(len(lines)))

    for _ in range(num_errors):
        if not available_line_indices:
            break # No more lines to inject errors into

        chosen_error_type = error_type if error_type and error_type != 'random' else random.choice(error_types)

        # Select a random line that hasn't been used yet
        line_index_to_modify = random.choice(available_line_indices)
        line = lines[line_index_to_modify]

        modified = False
        error_description = ""

        if chosen_error_type == 'syntax':
            # Try to introduce a syntax error (e.g., misspelled component, missing value)
            if re.match(r'^[rcldqmv][a-zA-Z0-9_]+', line, re.IGNORECASE): # Component line
                parts = line.split()
                if len(parts) >= 2:
                    # Misspell component name
                    original_name = parts[0]
                    misspelled_name = original_name[:-1] + random.choice('xyz') if len(original_name) > 1 else original_name + 'Z'
                    lines[line_index_to_modify] = misspelled_name + ' ' + ' '.join(parts[1:])
                    error_description = f"Syntax: Misspelled component '{original_name}' to '{misspelled_name}' on line {line_index_to_modify + 1}"
                    modified = True
                elif len(parts) >= 3 and random.random() < 0.5: # Try to remove a value
                    lines[line_index_to_modify] = ' '.join(parts[:-1]) # Remove the last part (usually value)
                    error_description = f"Syntax: Removed value from component '{parts[0]}' on line {line_index_to_modify + 1}"
                    modified = True
            elif line.strip().startswith('.'): # Control line
                if random.random() < 0.5 and len(line.split()) > 1:
                    # Misspell control command
                    original_command = line.split()[0]
                    misspelled_command = original_command[:-1] + random.choice('xyz') if len(original_command) > 1 else original_command + 'Z'
                    lines[line_index_to_modify] = misspelled_command + ' ' + ' '.join(line.split()[1:])
                    error_description = f"Syntax: Misspelled control command '{original_command}' on line {line_index_to_modify + 1}"
                    modified = True
            
            if not modified: # If above didn't modify, try a simpler syntax error
                if line.strip() and not line.strip().startswith('*'): # Not empty or comment
                    original_char = line[random.randint(0, len(line) - 1)]
                    random_char = random.choice('!@#$%^&()-_=+[]{}|;:\'",<.>/?')
                    lines[line_index_to_modify] = line.replace(original_char, random_char, 1)
                    error_description = f"Syntax: Replaced '{original_char}' with '{random_char}' on line {line_index_to_modify + 1}"
                    modified = True

        elif chosen_error_type == 'structural':
            # Try to introduce a structural error (e.g., floating node, short circuit)
            if re.match(r'^[rcldqmv][a-zA-Z0-9_]+\s+\S+\s+\S+', line, re.IGNORECASE): # Component line with at least 2 nodes
                parts = line.split()
                if len(parts) >= 3:
                    node1 = parts[1]
                    node2 = parts[2]
                    
                    if random.random() < 0.6: # Floating node: Make one node the same as another
                        lines[line_index_to_modify] = f"{parts[0]} {node1} {node1} {' '.join(parts[3:])}"
                        error_description = f"Structural: Created floating node by setting node '{node2}' to '{node1}' on line {line_index_to_modify + 1}"
                        modified = True
                    else: # Short circuit: Connect a node to ground (0)
                        if node1 != '0' and node2 != '0': # Avoid grounding already grounded nodes
                            node_to_ground = random.choice([node1, node2])
                            if node_to_ground == node1:
                                lines[line_index_to_modify] = f"{parts[0]} 0 {node2} {' '.join(parts[3:])}"
                            else:
                                lines[line_index_to_modify] = f"{parts[0]} {node1} 0 {' '.join(parts[3:])}"
                            error_description = f"Structural: Shorted node '{node_to_ground}' to ground (0) on line {line_index_to_modify + 1}"
                            modified = True

        elif chosen_error_type == 'value':
            # Try to inject an absurd value
            match = re.search(r'(\d+(\.\d*)?([eE][+-]?\d+)?(?:[pnumkGT]?))', line) # Find a number with optional unit
            if match:
                original_value_str = match.group(1)
                try:
                    # Attempt to convert to float, handle units for actual numeric operations
                    # For simplicity, we'll just replace the string. A real parser would handle units.
                    if 'p' in original_value_str or 'n' in original_value_str: # Make a very large value
                         new_value = float(re.sub(r'[pnu]', '', original_value_str)) * 1e12 # e.g., 10p to 10e12
                    elif 'k' in original_value_str or 'meg' in original_value_str or 'G' in original_value_str: # Make a very small value
                         new_value = float(re.sub(r'[kMG]', '', original_value_str)) / 1e12 # e.g., 10k to 10e-12
                    else:
                        num = float(original_value_str)
                        if num > 0:
                            new_value = num * 1e12 if random.random() < 0.5 else num / 1e12
                        else: # Handle zero or negative values differently
                            new_value = 1e-12 # Just make it a very small positive number
                    
                    lines[line_index_to_modify] = line.replace(original_value_str, f"{new_value:.10e}", 1)
                    error_description = f"Value: Changed '{original_value_str}' to absurd '{new_value:.10e}' on line {line_index_to_modify + 1}"
                    modified = True
                except ValueError:
                    pass # Not a straightforward number to modify

        elif chosen_error_type == 'model_missing':
            # Try to remove a .model definition if a device (q, m, d) is used
            device_pattern = re.compile(r'^[qmd][a-zA-Z0-9_]+\s+\S+\s+\S+\s+\S+\s+(\S+)', re.IGNORECASE)
            model_name_match = device_pattern.match(line)
            if model_name_match:
                model_name_to_remove = model_name_match.group(1)
                
                # Find the .model line for this model
                model_line_index = -1
                for i, l in enumerate(lines):
                    if l.strip().lower().startswith(f'.model {model_name_to_remove.lower()}'):
                        model_line_index = i
                        break
                
                if model_line_index != -1:
                    lines[model_line_index] = '' # Remove the model definition line
                    error_description = f"Model: Removed '.model {model_name_to_remove}' definition on line {model_line_index + 1}"
                    modified = True
            
            if not modified: # If no device was found on this line, find a random .model and remove it
                model_lines = [i for i, l in enumerate(lines) if l.strip().lower().startswith('.model')]
                if model_lines:
                    idx_to_remove = random.choice(model_lines)
                    model_line_content = lines[idx_to_remove].strip()
                    lines[idx_to_remove] = ''
                    error_description = f"Model: Randomly removed model definition '{model_line_content}' on line {idx_to_remove + 1}"
                    modified = True

        if modified:
            injected_errors.append(error_description)
            available_line_indices.remove(line_index_to_modify) # Ensure this line isn't modified again
        else:
            # If no error could be injected on the chosen line, try again with a different line
            available_line_indices.remove(line_index_to_modify) # Remove it so we don't try again
            if available_line_indices: # Only recurse if there are still lines left
                # This ensures we try to inject `num_errors` successfully
                # Or you could just let it complete with fewer errors if lines are exhausted/unsuitable
                continue 
            else:
                break # No more suitable lines

    # Reconstruct the netlist, filtering out any blank lines from removal
    final_netlist = '\n'.join([line for line in lines if line is not None])
    return final_netlist, injected_errors

