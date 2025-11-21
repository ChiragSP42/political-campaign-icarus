#%%
"""
Dynamic Election Cycle Context Generator

This script generates dynamic, real-time election cycle context for various office positions.
It calculates which elections have occurred within a specified lookback period from the current date,
without hard-coding values, making it work dynamically across different years.

Author: AI AWS Cloud Solutions Architect
Date: October 29, 2025
"""

import json
import datetime
from typing import Dict, List, Optional, Any


def generate_election_context(
    election_cycles: Dict[str, List[Dict[str, Any]]],
    current_date: Optional[datetime.datetime] = None,
    lookback_years: int = 5
) -> str:
    """
    Generate dynamic election cycle context based on office positions.
    
    This function produces a detailed, human-readable report about election cycles
    for each office, including the specific years elections were held and the pattern.
    
    Args:
        office_mapping: Dictionary mapping office names to their IDs
        current_date: Current date (defaults to today if not provided)
        lookback_years: Number of years to look back for elections (default: 5)
    
    Returns:
        Formatted string describing election years for each office with detailed descriptions
        
    Example:
        >>> office_mapping = {"President": 1, "Governor": 3}
        >>> context = generate_election_context(office_mapping)
        >>> print(context)
    """
    
    # Use current date if not provided
    if current_date is None:
        current_date = datetime.datetime.now()
    
    current_year = current_date.year
    start_year = current_year - lookback_years
    end_year = current_year
    
    def calculate_election_years(config: Dict) -> List[int]:
        """Calculate election years for a given office within the lookback period."""
        cycle = config["cycle"]
        pattern = config["election_pattern"]
        years = []
        
        if pattern == "even":
            # Elections in even years only
            start = current_year if current_year % 2 == 0 else current_year - 1
            y = start
            while y >= start_year:
                years.append(y)
                y -= cycle
                
        elif pattern == "odd":
            # Elections in odd years only
            start = current_year if current_year % 2 == 1 else current_year - 1
            y = start
            while y >= start_year:
                years.append(y)
                y -= cycle
                
        elif pattern == "even_biennial":
            # Elections every 2 years in even years (like U.S. Senate)
            start = current_year if current_year % 2 == 0 else current_year - 1
            y = start
            while y >= start_year:
                years.append(y)
                y -= 2
                
        elif pattern == "annual":
            # Annual elections
            years = list(range(start_year, end_year + 1))
            
        elif pattern == "periodic":
            # Periodic elections (e.g., every 5 years)
            y = current_year
            while y >= start_year:
                years.append(y)
                y -= cycle
                
        # Filter to ensure all years are within range and sort
        years = sorted([y for y in years if start_year <= y <= end_year])
        return years
    
    # Generate context for each office
    context_lines = []
    context_lines.append(f"Election Context (Current Year: {current_year}, Lookback: {lookback_years} years)")
    context_lines.append("=" * 80)
    context_lines.append("")
    
    for election_cycle in election_cycles["election_cycles"]:
        years = calculate_election_years(election_cycle)
        
        # Format office name for display
        display_name = election_cycle["election"].replace("_", " ")
        
        if years:
            years_str = ", ".join(str(y) for y in years)
            context_lines.append(f"{display_name} (ID: {election_cycle['id']})")
            context_lines.append(f"  Elections held in: {years_str}")
            context_lines.append(f"  Pattern: {election_cycle['description']}")
            context_lines.append("")
        else:
            context_lines.append(f"{display_name} (ID: {election_cycle['id']})")
            context_lines.append(f"  No elections found in the last {lookback_years} years")
            context_lines.append("")
    
    return "\n".join(context_lines)


def generate_llm_context(
    election_cycles: Dict[str, List[Dict[str, Any]]],
    current_date: Optional[datetime.datetime] = None,
    lookback_years: int = 5
) -> str:
    """
    Generate concise LLM-ready context about election cycles.
    
    This function produces a compact, LLM-friendly string that summarizes
    which elections occurred in which years for each office. Perfect for
    providing context to language models.
    
    Args:
        office_mapping: Dictionary mapping office names to their IDs
        current_date: Current date (defaults to today if not provided)
        lookback_years: Number of years to look back for elections (default: 5)
    
    Returns:
        Concise formatted string for LLM consumption
        
    Example:
        >>> office_mapping = {"President": 1, "Governor": 3}
        >>> context = generate_llm_context(office_mapping)
        >>> print(context)
        The President office held elections in: 2020, 2024.
        The Governor office held elections in: 2021, 2025.
    """
    
    if current_date is None:
        current_date = datetime.datetime.now()
    
    current_year = current_date.year
    start_year = current_year - lookback_years
    
    # Election cycles configuration (same as above)
    
    def calculate_election_years(config: Dict) -> List[int]:
        """Calculate election years for a given office."""
        cycle = config["cycle"]
        pattern = config["election_pattern"]
        years = []
        
        if pattern == "even":
            start = current_year if current_year % 2 == 0 else current_year - 1
            y = start
            while y >= start_year:
                years.append(y)
                y -= cycle
        elif pattern == "odd":
            start = current_year if current_year % 2 == 1 else current_year - 1
            y = start
            while y >= start_year:
                years.append(y)
                y -= cycle
        elif pattern == "even_biennial":
            start = current_year if current_year % 2 == 0 else current_year - 1
            y = start
            while y >= start_year:
                years.append(y)
                y -= 2
        elif pattern == "annual":
            years = list(range(start_year, current_year + 1))
        elif pattern == "periodic":
            y = current_year
            while y >= start_year:
                years.append(y)
                y -= cycle
        
        return sorted([y for y in years if start_year <= y <= current_year])
    
    # Generate concise context
    context_lines = []
    for election_cycle in election_cycles["election_cycles"]:
        years = calculate_election_years(election_cycle)
        
        display_name = election_cycle['election'].replace("_", " ")
        
        if years:
            years_str = ", ".join(str(y) for y in years)
            context_lines.append(f"The {display_name} office held elections in: {years_str}.")

    return "\n".join(context_lines)


def load_office_mapping_from_file(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load office mapping from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing office mappings
        
    Returns:
        Dictionary mapping office names to their IDs
        
    Example:
        >>> mapping = load_office_mapping_from_file('mapping.json')
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("DYNAMIC ELECTION CYCLE CONTEXT GENERATOR")
    print("=" * 80)
    print()
    
    # Example 1: Load from file
    election_cycles = load_office_mapping_from_file('election_cycles.json')
    print("✓ Loaded office mapping from 'election_cycles.json'")
    
    # Example 2: Generate detailed context with current date
    print("=" * 80)
    print("EXAMPLE 1: DETAILED ELECTION CONTEXT")
    print("=" * 80)
    print()
    
    current_date = datetime.datetime.now()
    detailed_context = generate_election_context(election_cycles, current_date, lookback_years=5)
    print(detailed_context)
    
    # Example 3: Generate concise LLM context
    print("\n" + "=" * 80)
    print("EXAMPLE 2: CONCISE LLM CONTEXT")
    print("=" * 80)
    print()
    
    llm_context = generate_llm_context(election_cycles, current_date, lookback_years=5)
    print(llm_context)
    
    # Example 4: Usage with custom date
    print("\n" + "=" * 80)
    print("EXAMPLE 3: CUSTOM DATE (10 YEARS LOOKBACK)")
    print("=" * 80)
    print()
    
    custom_date = datetime.datetime(2025, 10, 29)
    custom_context = generate_llm_context(election_cycles, custom_date, lookback_years=10)
    print(custom_context)
    
    # Example 5: Show usage patterns
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES IN CODE")
    print("=" * 80)
# %%
