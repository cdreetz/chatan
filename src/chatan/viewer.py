"""Live HTML viewer for dataset generation with async batch processing."""

import json
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time
import asyncio
from aiohttp import web
import aiohttp_cors
import atexit


class LiveViewer:
    """Live HTML viewer for streaming dataset generation results."""
    
    def __init__(self, title: str = "Dataset Generation", auto_open: bool = True):
        self.title = title
        self.auto_open = auto_open
        self.temp_dir = None
        self.html_file = None
        self.data_file = None
        self.app = None
        self.runner = None
        self.site = None
        self.port = 8000
        self._active = False
        
    async def start(self, schema: Dict[str, Any]) -> str:
        """Start the viewer and return the URL."""
        self.temp_dir = tempfile.mkdtemp()
        self.html_file = Path(self.temp_dir) / "viewer.html"
        self.data_file = Path(self.temp_dir) / "data.json"
        
        # Initialize empty data file
        with open(self.data_file, 'w') as f:
            json.dump({"rows": [], "completed": False, "current_rows": {}}, f)
        
        # Create HTML file
        html_content = self._generate_html(list(schema.keys()))
        with open(self.html_file, 'w') as f:
            f.write(html_content)
        
        # Start async server
        await self._start_server()
        
        # Open in browser
        url = f"http://localhost:{self.port}/viewer.html"
        if self.auto_open:
            webbrowser.open(url)
        
        self._active = True
        return url
    
    def start_row(self, row_index: int):
        """Start a new row with empty cells."""
        if not self._active or not self.data_file:
            return
            
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}
        
        data["current_rows"][str(row_index)] = {"index": row_index, "cells": {}}
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
    
    def update_cell(self, row_index: int, column: str, value: Any):
        """Update a single cell in a specific row."""
        if not self._active or not self.data_file:
            return
            
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}
        
        row_key = str(row_index)
        if row_key in data.get("current_rows", {}):
            data["current_rows"][row_key]["cells"][column] = value
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
    
    def complete_row(self, row_index: int, row_data: Dict[str, Any]):
        """Mark a row as complete and move it to completed rows."""
        if not self._active or not self.data_file:
            return
            
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}
        
        # Add to completed rows
        data["rows"].append(row_data)
        
        # Remove from current rows
        row_key = str(row_index)
        if row_key in data.get("current_rows", {}):
            del data["current_rows"][row_key]
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
    
    def complete(self):
        """Mark generation as complete."""
        if not self._active or not self.data_file:
            return
            
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}
        
        data["completed"] = True
        data["current_rows"] = {}
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
    
    async def stop(self):
        """Stop the viewer and cleanup resources."""
        self._active = False
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
    
    async def _start_server(self):
        """Start an async HTTP server."""
        self.app = web.Application()
        
        # Add CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add routes
        self.app.router.add_get('/viewer.html', self._serve_html)
        self.app.router.add_get('/data.json', self._serve_data)
        
        # Add CORS to routes
        for route in list(self.app.router.routes()):
            cors.add(route)
        
        # Find available port and start server
        for port in range(8000, 8100):
            try:
                self.runner = web.AppRunner(self.app)
                await self.runner.setup()
                self.site = web.TCPSite(self.runner, 'localhost', port)
                await self.site.start()
                self.port = port
                break
            except OSError:
                if self.runner:
                    await self.runner.cleanup()
                continue
    
    async def _serve_html(self, request):
        """Serve the HTML file."""
        with open(self.html_file, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')
    
    async def _serve_data(self, request):
        """Serve the data JSON file."""
        with open(self.data_file, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='application/json')
    
    def _generate_html(self, columns) -> str:
        """Generate the HTML content."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8fafc;
            color: #1e293b;
        }}
        
        .header {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .title {{
            font-size: 24px;
            font-weight: 600;
            margin: 0 0 10px 0;
        }}
        
        .status {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #64748b;
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 1.5s infinite;
        }}
        
        .status-dot.complete {{
            background: #6366f1;
            animation: none;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .table-container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
            max-height: 70vh;
            overflow-y: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}
        
        th {{
            background: #f1f5f9;
            padding: 16px;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid #e2e8f0;
            position: sticky;
            top: 0;
            z-index: 10;
            position: relative;
        }}

        th:not(:last-child), td:not(:last-child) {{
            border-right: 1px solid #e2e8f0;
        }}

        .col-resizer {{
            position: absolute;
            right: 0;
            top: 0;
            height: 100%;
            width: 5px;
            cursor: col-resize;
            user-select: none;
        }}
        
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid #f1f5f9;
            vertical-align: top;
        }}
        
        tr:hover {{
            background: #f8fafc;
        }}
        
        .row-number {{
            color: #64748b;
            font-size: 12px;
            font-weight: 500;
            width: 60px;
        }}
        
        .cell-content {{
            max-width: 300px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }}
        
        .cell-generating {{
            background: linear-gradient(90deg, #f1f5f9, #e2e8f0, #f1f5f9);
            background-size: 200% 200%;
            animation: shimmer 1.5s ease-in-out infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        
        .new-row {{
            animation: slideIn 0.3s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">{self.title}</div>
        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Generating...</span>
            <span id="rowCount">0 rows</span>
        </div>
    </div>
    
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th class="row-number">#</th>
                    {' '.join(f'<th>{col}</th>' for col in columns)}
                </tr>
            </thead>
            <tbody id="tableBody">
                <tr>
                    <td colspan="{len(columns) + 1}" class="loading">
                        Waiting for data...
                    </td>
                </tr>
            </tbody>
        </table>
    </div>

    <script>
        let rowCount = 0;
        let currentRowElements = {{}};

        function makeColumnsResizable(table) {{
            const headers = table.querySelectorAll('th');
            headers.forEach((th, index) => {{
                if (index === headers.length - 1) return;
                const resizer = document.createElement('div');
                resizer.className = 'col-resizer';
                th.appendChild(resizer);

                let startX, startWidth;

                resizer.addEventListener('mousedown', (e) => {{
                    startX = e.clientX;
                    startWidth = th.offsetWidth;
                    document.addEventListener('mousemove', doDrag);
                    document.addEventListener('mouseup', stopDrag);
                }});

                function doDrag(e) {{
                    const width = startWidth + e.clientX - startX;
                    th.style.width = width + 'px';
                }}

                function stopDrag() {{
                    document.removeEventListener('mousemove', doDrag);
                    document.removeEventListener('mouseup', stopDrag);
                }}
            }});
        }}

        async function fetchData() {{
            try {{
                const response = await fetch('data.json?' + new Date().getTime());
                const data = await response.json();
                
                // Handle current rows updates
                if (data.current_rows) {{
                    Object.values(data.current_rows).forEach(currentRow => {{
                        updateCurrentRow(currentRow);
                    }});
                }}
                
                // Handle completed rows
                if (data.rows.length > rowCount) {{
                    rowCount = data.rows.length;
                    updateStatus(data.completed);
                }}
                
                if (data.completed) {{
                    document.getElementById('statusDot').classList.add('complete');
                    document.getElementById('statusText').textContent = 'Complete';
                    Object.keys(currentRowElements).forEach(key => {{
                        if (currentRowElements[key] && currentRowElements[key].parentNode) {{
                            currentRowElements[key].remove();
                        }}
                    }});
                    currentRowElements = {{}};
                    return;
                }}
            }} catch (error) {{
                console.error('Error fetching data:', error);
            }}
            
            setTimeout(fetchData, 100);
        }}
        
        function updateCurrentRow(currentRow) {{
            const tbody = document.getElementById('tableBody');
            const rowIndex = currentRow.index;
            
            if (tbody.children.length === 1 && tbody.children[0].cells.length === {len(columns) + 1}) {{
                tbody.innerHTML = '';
            }}
            
            if (!currentRowElements[rowIndex]) {{
                const rowElement = document.createElement('tr');
                rowElement.className = 'new-row';
                rowElement.dataset.rowIndex = rowIndex;
                
                const numCell = document.createElement('td');
                numCell.className = 'row-number';
                numCell.textContent = rowIndex + 1;
                rowElement.appendChild(numCell);
                
                {json.dumps(columns)}.forEach(col => {{
                    const td = document.createElement('td');
                    td.className = 'cell-content cell-generating';
                    td.textContent = '...';
                    td.id = `cell-${{rowIndex}}-${{col}}`;
                    rowElement.appendChild(td);
                }});
                
                const existingRows = Array.from(tbody.children);
                let insertBefore = null;
                for (let i = 0; i < existingRows.length; i++) {{
                    const existingIndex = parseInt(existingRows[i].dataset.rowIndex);
                    if (existingIndex > rowIndex) {{
                        insertBefore = existingRows[i];
                        break;
                    }}
                }}
                
                if (insertBefore) {{
                    tbody.insertBefore(rowElement, insertBefore);
                }} else {{
                    tbody.appendChild(rowElement);
                }}
                
                currentRowElements[rowIndex] = rowElement;
            }}
            
            Object.entries(currentRow.cells).forEach(([col, value]) => {{
                const cell = document.getElementById(`cell-${{rowIndex}}-${{col}}`);
                if (cell) {{
                    cell.textContent = value || '';
                    cell.classList.remove('cell-generating');
                }}
            }});
        }}
        
        function updateStatus(completed) {{
            document.getElementById('rowCount').textContent = `${{rowCount}} rows`;
            if (completed) {{
                document.getElementById('statusText').textContent = 'Complete';
                document.getElementById('statusDot').classList.add('complete');
            }}
        }}
        
        makeColumnsResizable(document.querySelector('table'));
        fetchData();
    </script>
</body>
</html>"""


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
        # Start viewer
        url = await viewer.start(dataset_instance.schema)
        print(f"Live viewer started at: {url}")
        
        # Build dependency graph
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
        
        # Import pandas dynamically
        import pandas as pd
        dataset_instance._data = pd.DataFrame(data)
        return dataset_instance._data
        
    except Exception as e:
        await viewer.stop()
        raise e
    finally:
        await asyncio.sleep(1)
        await viewer.stop()  # Keep server running briefly


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
        return func


def create_viewer_callback(viewer: LiveViewer) -> Callable[[Dict[str, Any]], None]:
    """Create a callback function for dataset generation progress."""
    def callback(row: Dict[str, Any]):
        viewer.add_row(row)
    return callback
