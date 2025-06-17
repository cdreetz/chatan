"""Async generation functions for the viewer."""

import asyncio
from typing import Dict, Any, Optional
import pandas as pd
from .live_viewer import LiveViewer


async def generate_with_viewer(
    dataset_instance,
    n: Optional[int] = None,
    batch_size: int = 5,
    viewer_title: str = "Dataset Generation",
    auto_open: bool = True,
    cell_delay: float = 0.1
):
    """Generate dataset with live viewer showing batch generation.
    
    Args:
        dataset_instance: The Dataset instance
        n: Number of samples to generate
        batch_size: Number of rows to generate simultaneously
        viewer_title: Title for the HTML viewer
        auto_open: Whether to auto-open browser
        cell_delay: Delay between cell generations within a row
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    viewer = LiveViewer(title=viewer_title, auto_open=auto_open)
    num_samples = n or dataset_instance.n
    
    try:
        url = await viewer.start(dataset_instance.schema)
        print(f"Live viewer started at: {url}")
        
        dependencies = dataset_instance._build_dependency_graph()
        execution_order = dataset_instance._topological_sort(dependencies)
        
        data = []
        
        # Process in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = list(range(batch_start, batch_end))
            
            # Start all rows in this batch
            for i in batch_indices:
                viewer.start_row(i)
            
            # Generate batch data concurrently
            batch_tasks = []
            for i in batch_indices:
                task = generate_row_async(i, execution_order, dataset_instance, viewer, cell_delay)
                batch_tasks.append(task)
            
            # Wait for all rows in batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Add completed rows to data
            for i, row_data in enumerate(batch_results):
                row_index = batch_indices[i]
                data.append(row_data)
                viewer.complete_row(row_index, row_data)
        
        viewer.complete()
        dataset_instance._data = pd.DataFrame(data)
        return dataset_instance._data
        
    except Exception as e:
        await viewer.stop()
        raise e
    finally:
        await asyncio.sleep(1)
        await viewer.stop()


async def generate_row_async(row_index: int, execution_order: list, dataset_instance, viewer, cell_delay: float):
    """Generate a single row asynchronously with live updates."""
    row = {}
    
    for column in execution_order:
        # Generate cell value
        value = await generate_value_async(dataset_instance, column, row)
        row[column] = value
        
        # Update viewer
        viewer.update_cell(row_index, column, value)
        
        # Small delay for visual effect
        if cell_delay > 0:
            await asyncio.sleep(cell_delay)
    
    return row


async def generate_value_async(dataset_instance, column: str, context: Dict[str, Any]) -> Any:
    """Generate a single value asynchronously."""
    func = dataset_instance.schema[column]
    
    # Check if the generator supports async
    if hasattr(func, 'generate_async'):
        return await func.generate_async(context)
    elif hasattr(func, '__call__'):
        # For non-async functions, run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, context)
    else:
        # Static value
        return func
