"""Utilities for extracting and managing schema metadata."""

from typing import Dict, Any


def extract_schema_metadata(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata about each column's generator.
    
    Args:
        schema: The dataset schema mapping column names to functions
        
    Returns:
        Dict containing metadata for each column
    """
    metadata = {}
    
    for column, func in schema.items():
        if hasattr(func, 'prompt_template'):  # GeneratorFunction
            metadata[column] = {
                "type": "generator",
                "prompt": func.prompt_template,
                "variables": list(func.variables.keys()) if func.variables else [],
                "provider": func.generator.__class__.__name__.replace('Generator', '').lower()
            }
        elif hasattr(func, 'choices'):  # SampleFunction
            if hasattr(func, 'weights'):
                metadata[column] = {
                    "type": "weighted_sample",
                    "choices": dict(zip(func.choices, func.weights)) if func.weights else func.choices
                }
            else:
                metadata[column] = {
                    "type": "sample",
                    "choices": func.choices
                }
        elif hasattr(func, 'start') and hasattr(func, 'end'):  # RangeSampler
            metadata[column] = {
                "type": "range",
                "start": func.start,
                "end": func.end,
                "is_int": getattr(func, 'is_int', False)
            }
        elif hasattr(func, '__name__') and func.__name__ == 'UUIDSampler':  # UUIDSampler
            metadata[column] = {
                "type": "uuid"
            }
        elif hasattr(func, 'start') and hasattr(func, 'delta'):  # DatetimeSampler
            metadata[column] = {
                "type": "datetime",
                "start": func.start.isoformat(),
                "end": func.end.isoformat()
            }
        elif callable(func):
            metadata[column] = {
                "type": "function",
                "name": getattr(func, '__name__', 'anonymous')
            }
        else:
            metadata[column] = {
                "type": "static",
                "value": func
            }
    
    return metadata


def update_schema_from_metadata(schema: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Update schema based on modified metadata.
    
    Args:
        schema: Original schema
        metadata: Updated metadata
        
    Returns:
        Updated schema (note: this is a placeholder - full implementation would
        require recreating the function objects)
    """
    # This is a placeholder for now - implementing full schema reconstruction
    # would require access to the original generator/sampler classes
    # For now, we just update the metadata that can be changed
    
    updated_schema = schema.copy()
    
    for column, meta in metadata.items():
        if column in updated_schema:
            func = updated_schema[column]
            
            # Update generator prompts
            if meta.get("type") == "generator" and hasattr(func, 'prompt_template'):
                func.prompt_template = meta["prompt"]
            
            # Update sample choices (this would need more work to properly recreate the objects)
            elif meta.get("type") in ["sample", "weighted_sample"] and hasattr(func, 'choices'):
                if isinstance(meta["choices"], dict):
                    func.choices = list(meta["choices"].keys())
                    func.weights = list(meta["choices"].values()) if hasattr(func, 'weights') else None
                else:
                    func.choices = meta["choices"]
    
    return updated_schema


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate metadata structure.
    
    Args:
        metadata: Metadata to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ["type"]
    
    for column, meta in metadata.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Metadata for column '{column}' must be a dictionary")
        
        for field in required_fields:
            if field not in meta:
                raise ValueError(f"Missing required field '{field}' in metadata for column '{column}'")
        
        # Validate specific types
        if meta["type"] == "generator":
            if "prompt" not in meta:
                raise ValueError(f"Generator column '{column}' missing 'prompt' field")
        elif meta["type"] in ["sample", "weighted_sample"]:
            if "choices" not in meta:
                raise ValueError(f"Sample column '{column}' missing 'choices' field")
    
    return True


def get_column_summary(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Get a summary of each column for display.
    
    Args:
        metadata: Column metadata
        
    Returns:
        Dict mapping column names to summary strings
    """
    summaries = {}
    
    for column, meta in metadata.items():
        if meta["type"] == "generator":
            summaries[column] = f"LLM ({meta.get('provider', 'unknown')}): {meta.get('prompt', '')[:50]}..."
        elif meta["type"] == "sample":
            choices = meta.get("choices", [])
            if isinstance(choices, list):
                choice_str = ", ".join(str(c) for c in choices[:3])
                if len(choices) > 3:
                    choice_str += "..."
            else:
                choice_str = str(choices)[:30]
            summaries[column] = f"Sample: [{choice_str}]"
        elif meta["type"] == "weighted_sample":
            choices = meta.get("choices", {})
            if isinstance(choices, dict):
                choice_str = ", ".join(str(k) for k in list(choices.keys())[:3])
                if len(choices) > 3:
                    choice_str += "..."
            else:
                choice_str = str(choices)[:30]
            summaries[column] = f"Weighted: [{choice_str}]"
        elif meta["type"] == "range":
            summaries[column] = f"Range: {meta.get('start', '?')} - {meta.get('end', '?')}"
        elif meta["type"] == "uuid":
            summaries[column] = "UUID"
        elif meta["type"] == "datetime":
            summaries[column] = f"Date: {meta.get('start', '?')} - {meta.get('end', '?')}"
        elif meta["type"] == "function":
            summaries[column] = f"Function: {meta.get('name', 'anonymous')}"
        elif meta["type"] == "static":
            summaries[column] = f"Static: {str(meta.get('value', ''))[:30]}"
        else:
            summaries[column] = f"Unknown: {meta['type']}"
    
    return summaries
