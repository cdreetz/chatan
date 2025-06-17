# viewer/live_viewer.py
"""Core LiveViewer class for managing the web interface."""

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, Any
import asyncio
from aiohttp import web
import aiohttp_cors
import pkg_resources

from .schema_utils import extract_schema_metadata


class LiveViewer:
    """Live HTML viewer for streaming dataset generation results."""
    
    def __init__(self, title: str = "Dataset Generation", auto_open: bool = True):
        self.title = title
        self.auto_open = auto_open
        self.temp_dir = None
        self.data_file = None
        self.schema_file = None
        self.port = 8000
        self._active = False
        self.app = None
        self.runner = None
        self.site = None
        
    async def start(self, schema: Dict[str, Any]) -> str:
        """Start the viewer and return the URL."""
        self._setup_files(schema)
        await self._start_server()
        
        url = f"http://localhost:{self.port}/"
        if self.auto_open:
            webbrowser.open(url)
        
        self._active = True
        return url
    
    def _setup_files(self, schema: Dict[str, Any]):
        """Setup temporary files for the viewer."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = Path(self.temp_dir) / "data.json"
        self.schema_file = Path(self.temp_dir) / "schema.json"
        
        # Initialize data files
        with open(self.data_file, 'w') as f:
            json.dump({"rows": [], "completed": False, "current_rows": {}}, f)
        
        schema_metadata = extract_schema_metadata(schema)
        with open(self.schema_file, 'w') as f:
            json.dump(schema_metadata, f)
        
        # Copy static web files to temp directory
        self._copy_web_files()
    
    def _copy_web_files(self):
        """Copy HTML, CSS, JS files from package to temp directory."""
        import shutil
        
        # Get the package directory
        package_dir = Path(__file__).parent / "static"
        
        if package_dir.exists():
            # Copy all static files
            for file_path in package_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(package_dir)
                    dest_path = Path(self.temp_dir) / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
        
        # Update the HTML template with dynamic values
        self._update_html_template()
    
    def _update_html_template(self):
        """Update HTML template with dynamic values like title."""
        html_file = Path(self.temp_dir) / "index.html"
        if html_file.exists():
            with open(html_file, 'r') as f:
                content = f.read()
            
            # Replace placeholders
            content = content.replace('{{TITLE}}', self.title)
            
            with open(html_file, 'w') as f:
                f.write(content)
    
    async def _start_server(self):
        """Start the async HTTP server."""
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
        
        # Serve static files
        self.app.router.add_static('/', self.temp_dir, name='static')
        
        # API routes
        self.app.router.add_get('/api/data', self._serve_data)
        self.app.router.add_get('/api/schema', self._serve_schema)
        self.app.router.add_post('/api/update_schema', self._update_schema)
        
        # Add CORS to routes
        for route in list(self.app.router.routes()):
            cors.add(route)
        
        # Start server
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
    
    # Data management methods (unchanged)
    def start_row(self, row_index: int):
        """Start a new row with empty cells."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(lambda data: self._add_current_row(data, row_index))
    
    def update_cell(self, row_index: int, column: str, value: Any):
        """Update a single cell in a specific row."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(lambda data: self._update_cell_data(data, row_index, column, value))
    
    def complete_row(self, row_index: int, row_data: Dict[str, Any]):
        """Mark a row as complete."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(lambda data: self._complete_row_data(data, row_index, row_data))
    
    def complete(self):
        """Mark generation as complete."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(lambda data: self._mark_complete(data))
    
    def _update_data_file(self, update_func):
        """Helper to update data file with error handling."""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}
        
        update_func(data)
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
    
    def _add_current_row(self, data, row_index):
        data["current_rows"][str(row_index)] = {"index": row_index, "cells": {}}
    
    def _update_cell_data(self, data, row_index, column, value):
        row_key = str(row_index)
        if row_key in data.get("current_rows", {}):
            data["current_rows"][row_key]["cells"][column] = value
    
    def _complete_row_data(self, data, row_index, row_data):
        data["rows"].append(row_data)
        row_key = str(row_index)
        if row_key in data.get("current_rows", {}):
            del data["current_rows"][row_key]
    
    def _mark_complete(self, data):
        data["completed"] = True
        data["current_rows"] = {}
    
    # API handlers
    async def _serve_data(self, request):
        """Serve the data JSON file."""
        with open(self.data_file, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='application/json')
    
    async def _serve_schema(self, request):
        """Serve the schema metadata."""
        with open(self.schema_file, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='application/json')
    
    async def _update_schema(self, request):
        """Update schema and trigger regeneration."""
        data = await request.json()
        with open(self.schema_file, 'w') as f:
            json.dump(data, f)
        return web.Response(text='{"status": "updated"}', content_type='application/json')
    
    async def stop(self):
        """Stop the viewer and cleanup resources."""
        self._active = False
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
