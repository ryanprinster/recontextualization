#!/usr/bin/env python
"""Script to view completions in a readable text format."""

import json
import argparse
import sys
from pathlib import Path


def format_completions(json_path, output_path=None):
    """Load completions JSON and format as readable text."""
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"COMPLETIONS VIEWER")
    output_lines.append(f"Temperature: {data.get('temperature', 'N/A')}")
    output_lines.append(f"N (completions per prompt): {data.get('N', 'N/A')}")
    output_lines.append(f"Total prompts: {len(data.get('prompt_to_completions', {}))}")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Iterate through each prompt and its completions
    for prompt_idx, prompt_data in data.get('prompt_to_completions', {}).items():
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"PROMPT #{prompt_idx}")
        output_lines.append("=" * 80)
        
        # Show the original prompt/question if available
        if 'question' in prompt_data:
            output_lines.append(f"\nQUESTION:\n{prompt_data['question']}")
        elif 'prompt' in prompt_data:
            output_lines.append(f"\nPROMPT:\n{prompt_data['prompt']}")
        
        # Show any other metadata
        metadata_keys = [k for k in prompt_data.keys() if k not in ['question', 'prompt', 'completions']]
        if metadata_keys:
            output_lines.append("\nMETADATA:")
            for key in metadata_keys:
                value = prompt_data[key]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                output_lines.append(f"  {key}: {value}")
        
        # Show completions
        completions = prompt_data.get('completions', {})
        output_lines.append(f"\nCOMPLETIONS ({len(completions)} total):")
        output_lines.append("-" * 40)
        
        for comp_idx, comp_data in completions.items():
            output_lines.append(f"\n[Completion {comp_idx}]")
            
            # Show raw text
            raw_text = comp_data.get('raw_text', 'No text found')
            output_lines.append(f"\nRaw text:\n{raw_text}")
            
            # Show extracted tags if any
            extracted_fields = [k for k in comp_data.keys() if k not in ['raw_text', 'reward']]
            if extracted_fields:
                output_lines.append("\nExtracted fields:")
                for field in extracted_fields:
                    output_lines.append(f"\n<{field}>:\n{comp_data[field]}")
            
            # Show reward if available
            if 'reward' in comp_data:
                output_lines.append(f"\nReward: {comp_data['reward']}")
            
            output_lines.append("-" * 40)
    
    # Join all lines
    output_text = '\n'.join(output_lines)
    
    # Save or print
    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_text)
        print(f"Saved readable completions to {output_path}")
    else:
        print(output_text)


def main():
    parser = argparse.ArgumentParser(description="View completions in readable format")
    parser.add_argument('input_json', help='Path to completions JSON file')
    parser.add_argument('-o', '--output', help='Output text file (if not specified, prints to stdout)')
    parser.add_argument('--limit', type=int, help='Limit number of prompts to show')
    
    args = parser.parse_args()
    
    if not Path(args.input_json).exists():
        print(f"Error: File {args.input_json} not found")
        sys.exit(1)
    
    format_completions(args.input_json, args.output)


if __name__ == '__main__':
    main()