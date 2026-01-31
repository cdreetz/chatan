"""Live HTML viewer for dataset generation."""

import atexit
import json
import os
import tempfile
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class LiveViewer:
    """Live HTML viewer for streaming dataset generation results."""

    def __init__(self, title: str = "Dataset Generation", auto_open: bool = True):
        self.title = title
        self.auto_open = auto_open
        self.temp_dir = None
        self.html_file = None
        self.data_file = None
        self.server = None
        self.server_thread = None
        self.port = 8000
        self._active = False

    def start(self, schema: Dict[str, Any]) -> str:
        """Start the viewer and return the URL."""
        self.temp_dir = tempfile.mkdtemp()
        self.html_file = Path(self.temp_dir) / "viewer.html"
        self.data_file = Path(self.temp_dir) / "data.json"

        # Initialize empty data file
        with open(self.data_file, "w") as f:
            json.dump({"rows": [], "completed": False, "current_row": None}, f)

        # Create HTML file
        html_content = self._generate_html(list(schema.keys()))
        with open(self.html_file, "w") as f:
            f.write(html_content)

        # Start local server
        self._start_server()

        # Open in browser
        url = f"http://localhost:{self.port}/viewer.html"
        if self.auto_open:
            webbrowser.open(url)

        self._active = True
        return url

    def add_row(self, row: Dict[str, Any]):
        """Add a new row to the viewer."""
        if not self._active or not self.data_file:
            return

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Add row exception: {e}")
            data = {"rows": [], "completed": False, "current_row": None}

        data["rows"].append(row)
        # Keep current_row so we can update the UI with final values
        data["completed_row"] = {"index": len(data["rows"]) - 1, "data": row}
        data["current_row"] = None  # Clear current row when row is complete

        with open(self.data_file, "w") as f:
            json.dump(data, f)

    def start_row(self, row_index: int):
        """Start a new row with empty cells."""
        if not self._active or not self.data_file:
            return

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Start row exception: {e}")
            data = {"rows": [], "completed": False, "current_row": None}

        data["current_row"] = {"index": row_index, "cells": {}}

        with open(self.data_file, "w") as f:
            json.dump(data, f)

    def update_cell(self, column: str, value: Any):
        """Update a single cell in the current row."""
        if not self._active or not self.data_file:
            return

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Update cell exception: {e}")
            data = {"rows": [], "completed": False, "current_row": None}

        if data.get("current_row"):
            data["current_row"]["cells"][column] = value

            with open(self.data_file, "w") as f:
                json.dump(data, f)

    def complete(self):
        """Mark generation as complete."""
        if not self._active or not self.data_file:
            return

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Complete exception: {e}")
            data = {"rows": [], "completed": False, "current_row": None}

        data["completed"] = True

        with open(self.data_file, "w") as f:
            json.dump(data, f)

    def stop(self):
        """Stop the viewer and cleanup resources."""
        self._active = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def _start_server(self):
        """Start a local HTTP server."""
        os.chdir(self.temp_dir)

        # Find available port
        for port in range(8000, 8100):
            try:
                self.server = HTTPServer(("localhost", port), SimpleHTTPRequestHandler)
                self.port = port
                break
            except OSError:
                continue

        if self.server:
            self.server_thread = threading.Thread(
                target=self.server.serve_forever, daemon=True
            )
            self.server_thread.start()

            # Register cleanup on exit
            atexit.register(self.stop)

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
        let currentRowElement = null;

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
                
                // Handle current row updates
                if (data.current_row) {{
                    updateCurrentRow(data.current_row);
                }}
                
                // Handle completed rows
                if (data.completed_row) {{
                    completeRow(data.completed_row);
                }}
                
                if (data.rows.length > rowCount) {{
                    rowCount = data.rows.length;
                    updateStatus(data.completed);
                }}
                
                if (data.completed) {{
                    document.getElementById('statusDot').classList.add('complete');
                    document.getElementById('statusText').textContent = 'Complete';
                    return;
                }}
            }} catch (error) {{
                console.error('Error fetching data:', error);
            }}
            
            setTimeout(fetchData, 200); // Faster polling for cell updates
        }}
        
        function updateCurrentRow(currentRow) {{
            const tbody = document.getElementById('tableBody');
            
            // Remove loading message if present
            if (tbody.children.length === 1 && tbody.children[0].cells.length === {len(columns) + 1}) {{
                tbody.innerHTML = '';
            }}
            
            // Create new row element only if we don't have one for this index
            if (!currentRowElement || parseInt(currentRowElement.dataset.rowIndex) !== currentRow.index) {{
                currentRowElement = document.createElement('tr');
                currentRowElement.className = 'new-row';
                currentRowElement.dataset.rowIndex = currentRow.index;
                
                // Row number
                const numCell = document.createElement('td');
                numCell.className = 'row-number';
                numCell.textContent = currentRow.index + 1;
                currentRowElement.appendChild(numCell);
                
                // Create empty cells for all columns
                {json.dumps(columns)}.forEach(col => {{
                    const td = document.createElement('td');
                    td.className = 'cell-content cell-generating';
                    td.textContent = '...';
                    td.id = `cell-${{currentRow.index}}-${{col}}`;
                    currentRowElement.appendChild(td);
                }});
                
                tbody.appendChild(currentRowElement);
            }}
            
            // Update cells with values
            Object.entries(currentRow.cells).forEach(([col, value]) => {{
                const cell = document.getElementById(`cell-${{currentRow.index}}-${{col}}`);
                if (cell) {{
                    cell.textContent = value || '';
                    cell.classList.remove('cell-generating');
                }}
            }});
        }}
        
        function completeRow(completedRow) {{
            // Find the row element and update it with final data
            const rowElement = document.querySelector(`tr[data-row-index="${{completedRow.index}}"]`);
            if (rowElement) {{
                {json.dumps(columns)}.forEach((col, colIndex) => {{
                    const cell = rowElement.cells[colIndex + 1]; // +1 for row number column
                    if (cell) {{
                        cell.textContent = completedRow.data[col] || '';
                        cell.classList.remove('cell-generating');
                    }}
                }});
            }}
            
            // Clear current row if this was it
            if (currentRowElement && parseInt(currentRowElement.dataset.rowIndex) === completedRow.index) {{
                currentRowElement = null;
            }}
        }}
        
        function addRows(rows) {{
            const tbody = document.getElementById('tableBody');
            
            rows.forEach((row, index) => {{
                const tr = document.createElement('tr');
                tr.className = 'new-row';
                
                const numCell = document.createElement('td');
                numCell.className = 'row-number';
                numCell.textContent = rowCount - rows.length + index + 1;
                tr.appendChild(numCell);
                
                {json.dumps(columns)}.forEach(col => {{
                    const td = document.createElement('td');
                    td.className = 'cell-content';
                    td.textContent = row[col] || '';
                    tr.appendChild(td);
                }});
                
                tbody.appendChild(tr);
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


def create_viewer_callback(viewer: LiveViewer) -> Callable[[Dict[str, Any]], None]:
    """Create a callback function for dataset generation progress."""

    def callback(row: Dict[str, Any]):
        viewer.add_row(row)

    return callback


async def _generate_with_viewer_async(
    dataset_instance,
    viewer: LiveViewer,
    num_samples: int,
    stream_delay: float,
    cell_delay: float,
):
    """Async implementation of generate_with_viewer."""
    import asyncio
    import pandas as pd

    from .generator import GeneratorFunction
    from .sampler import SampleFunction

    # Build dependency graph
    dependencies = dataset_instance._build_dependency_graph()
    execution_order = dataset_instance._topological_sort(dependencies)

    # Generate data with live updates
    data = []

    for i in range(num_samples):
        # Start new row
        viewer.start_row(i)

        row = {}
        for column in execution_order:
            # Generate cell value
            func = dataset_instance.schema[column]

            if isinstance(func, GeneratorFunction):
                value = await func(row)
            elif isinstance(func, SampleFunction):
                value = func(row)
            elif callable(func):
                if asyncio.iscoroutinefunction(func):
                    value = await func(row)
                else:
                    value = func(row)
            else:
                value = func

            row[column] = value

            # Update viewer with new cell value
            viewer.update_cell(column, value)

            # Delay to show cell generation effect
            if cell_delay > 0:
                await asyncio.sleep(cell_delay)

        # Complete the row
        data.append(row)
        viewer.add_row(row)

        # Small delay between rows
        if stream_delay > 0:
            await asyncio.sleep(stream_delay)

    viewer.complete()

    dataset_instance._data = pd.DataFrame(data)
    return dataset_instance._data


def generate_with_viewer(
    dataset_instance,
    n: Optional[int] = None,
    progress: bool = True,
    viewer_title: str = "Dataset Generation",
    auto_open: bool = True,
    stream_delay: float = 0.05,
    cell_delay: float = 0.3,
):
    """Generate dataset with live viewer showing cell-by-cell generation.

    Args:
        dataset_instance: The Dataset instance
        n: Number of samples to generate
        progress: Show progress bar (ignored when using viewer)
        viewer_title: Title for the HTML viewer
        auto_open: Whether to auto-open browser
        stream_delay: Delay between rows for streaming effect
        cell_delay: Delay between individual cell generations

    Returns:
        pd.DataFrame: Generated dataset
    """
    import asyncio

    viewer = LiveViewer(title=viewer_title, auto_open=auto_open)
    num_samples = n or dataset_instance.n

    try:
        # Start viewer
        url = viewer.start(dataset_instance.schema)
        print(f"Live viewer started at: {url}")

        # Run async generation
        result = asyncio.run(
            _generate_with_viewer_async(
                dataset_instance,
                viewer,
                num_samples,
                stream_delay,
                cell_delay,
            )
        )
        return result

    except Exception as e:
        viewer.stop()
        raise e
    finally:
        # Keep server running for a bit so user can see final state
        time.sleep(1)
