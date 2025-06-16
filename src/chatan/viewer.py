"""Live HTML viewer for dataset generation."""

import json
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import atexit


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
        with open(self.data_file, 'w') as f:
            json.dump({"rows": [], "completed": False}, f)
        
        # Create HTML file
        html_content = self._generate_html(list(schema.keys()))
        with open(self.html_file, 'w') as f:
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
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False}
        
        data["rows"].append(row)
        
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
            data = {"rows": [], "completed": False}
        
        data["completed"] = True
        
        with open(self.data_file, 'w') as f:
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
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
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
        
        async function fetchData() {{
            try {{
                const response = await fetch('data.json?' + new Date().getTime());
                const data = await response.json();
                
                if (data.rows.length > rowCount) {{
                    const newRows = data.rows.slice(rowCount);
                    addRows(newRows);
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
            
            setTimeout(fetchData, 300);
        }}
        
        function addRows(rows) {{
            const tbody = document.getElementById('tableBody');
            
            if (tbody.children.length === 1 && tbody.children[0].cells.length === {len(columns) + 1}) {{
                tbody.innerHTML = '';
            }}
            
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
        
        fetchData();
    </script>
</body>
</html>"""


def create_viewer_callback(viewer: LiveViewer) -> Callable[[Dict[str, Any]], None]:
    """Create a callback function for dataset generation progress."""
    def callback(row: Dict[str, Any]):
        viewer.add_row(row)
    return callback


def generate_with_viewer(
    dataset_instance,
    n: Optional[int] = None,
    progress: bool = True,
    viewer_title: str = "Dataset Generation",
    auto_open: bool = True,
    stream_delay: float = 0.05
):
    """Generate dataset with live viewer.
    
    Args:
        dataset_instance: The Dataset instance
        n: Number of samples to generate
        progress: Show progress bar (ignored when using viewer)
        viewer_title: Title for the HTML viewer
        auto_open: Whether to auto-open browser
        stream_delay: Delay between rows for streaming effect
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    viewer = LiveViewer(title=viewer_title, auto_open=auto_open)
    num_samples = n or dataset_instance.n
    
    try:
        # Start viewer
        url = viewer.start(dataset_instance.schema)
        print(f"Live viewer started at: {url}")
        
        # Build dependency graph (copied from dataset logic)
        dependencies = dataset_instance._build_dependency_graph()
        execution_order = dataset_instance._topological_sort(dependencies)
        
        # Generate data with live updates
        data = []
        
        for i in range(num_samples):
            row = {}
            for column in execution_order:
                value = dataset_instance._generate_value(column, row)
                row[column] = value
            
            data.append(row)
            viewer.add_row(row)
            
            # Small delay for streaming effect
            if stream_delay > 0:
                time.sleep(stream_delay)
        
        viewer.complete()
        
        # Import pandas dynamically to avoid circular imports
        import pandas as pd
        dataset_instance._data = pd.DataFrame(data)
        return dataset_instance._data
        
    except Exception as e:
        viewer.stop()
        raise e
    finally:
        # Keep server running for a bit so user can see final state
        time.sleep(1)
