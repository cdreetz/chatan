"""Core LiveViewer class with proper Chatan backend integration."""

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, Any, List
import asyncio
from aiohttp import web
import aiohttp_cors

# Import chatan components
from chatan import dataset, generator, sample
from .schema_utils import extract_schema_metadata


class LiveViewer:
    """Live HTML viewer for streaming dataset generation results with Chatan integration."""

    def __init__(self, title: str = "Dataset Generation", auto_open: bool = True):
        self.title = title
        self.auto_open = auto_open
        self.temp_dir = None
        self.data_file = None
        self.schema_file = None
        self.columns_file = None
        self.port = 8000
        self._active = False
        self.app = None
        self.runner = None
        self.site = None

        # Generation state
        self.current_dataset = None
        self.is_generating = False
        self.generation_task = None
        self.generation_progress = {"current": 0, "total": 0}
        self.ui_columns = []  # Store UI format columns

        # Generator client (will be set when user provides API key or uses mock)
        self.generator_client = None

    async def start(self, schema: Dict[str, Any] = None) -> str:
        """Start the viewer and return the URL."""
        self._setup_files(schema or {})
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
        self.columns_file = Path(self.temp_dir) / "columns.json"

        # Initialize data files
        with open(self.data_file, "w") as f:
            json.dump({"rows": [], "completed": False, "current_rows": {}}, f)

        # Convert any existing schema to UI columns format
        if schema:
            schema_metadata = extract_schema_metadata(schema)
            self.ui_columns = self._chatan_to_ui_columns(schema, schema_metadata)
        else:
            self.ui_columns = []

        with open(self.columns_file, "w") as f:
            json.dump({"columns": {col["name"]: col for col in self.ui_columns}}, f)

        with open(self.schema_file, "w") as f:
            json.dump({}, f)  # Legacy compatibility

        # Copy React build files to temp directory
        self._copy_react_build()

    def _chatan_to_ui_columns(
        self, schema: Dict[str, Any], metadata: Dict[str, Any]
    ) -> List[Dict]:
        """Convert chatan schema to UI columns format."""
        columns = []
        for name, meta in metadata.items():
            if meta["type"] == "generator":
                columns.append(
                    {
                        "name": name,
                        "type": "prompt",
                        "config": {"prompt": meta.get("prompt", "")},
                    }
                )
            elif meta["type"] in ["sample", "weighted_sample"]:
                columns.append(
                    {
                        "name": name,
                        "type": "choice",
                        "config": {"choices": meta.get("choices", [])},
                    }
                )
            # Add more type conversions as needed
        return columns

    def _ui_to_chatan_schema(self, ui_columns: List[Dict]) -> Dict[str, Any]:
        """Convert UI columns format to chatan schema."""
        schema = {}

        for col in ui_columns:
            name = col["name"]
            col_type = col["type"]
            config = col["config"]

            if col_type == "prompt":
                # Create generator function
                if self.generator_client:
                    schema[name] = self.generator_client(config.get("prompt", ""))
                else:
                    # Fallback to mock data for testing without API key
                    schema[name] = lambda ctx, prompt=config.get(
                        "prompt", ""
                    ): f"Mock: {prompt[:20]}..."

            elif col_type == "choice":
                choices = config.get("choices", [])
                if isinstance(choices, dict):
                    # Weighted choices
                    schema[name] = sample.weighted(choices)
                elif isinstance(choices, list):
                    # Simple choices
                    schema[name] = sample.choice(choices)
                else:
                    schema[name] = sample.choice(["Option1", "Option2"])

            elif col_type == "reference":
                template = config.get("template", "")
                if self.generator_client:
                    schema[name] = self.generator_client(template)
                else:
                    # Mock reference
                    schema[name] = (
                        lambda ctx, tmpl=template: f"Mock ref: {tmpl[:20]}..."
                    )

        return schema

    def set_generator_client(self, provider: str, api_key: str, **kwargs):
        """Set the generator client for LLM generation.

        Additional keyword arguments are passed directly to
        :func:`chatan.generator.generator`, allowing callers to
        configure options such as model names or Azure specific
        settings.
        """
        try:
            self.generator_client = generator(provider, api_key, **kwargs)
        except Exception as e:
            print(f"Warning: Could not initialize generator client: {e}")
            self.generator_client = None

    def _copy_react_build(self):
        """Copy React build files from package to temp directory."""
        import shutil

        # Get the package directory - look for React build first, fallback to old static
        package_dir = Path(__file__).parent
        react_build_dir = package_dir / "frontend" / "dist"
        old_static_dir = package_dir / "static"

        if react_build_dir.exists():
            # Copy React build files
            for file_path in react_build_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(react_build_dir)
                    dest_path = Path(self.temp_dir) / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
            print(f"Serving React app from {react_build_dir}")
        elif old_static_dir.exists():
            # Fallback to old static files
            for file_path in old_static_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(old_static_dir)
                    dest_path = Path(self.temp_dir) / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)

            # Update the HTML template with dynamic values for old static
            self._update_html_template()
            print(f"Serving legacy static files from {old_static_dir}")
        else:
            print("Warning: No frontend files found. Creating minimal fallback.")
            self._create_fallback_html()

    def _update_html_template(self):
        """Update HTML template with dynamic values like title (for legacy static files)."""
        html_file = Path(self.temp_dir) / "index.html"
        if html_file.exists():
            with open(html_file, "r") as f:
                content = f.read()

            # Replace placeholders
            content = content.replace("{{TITLE}}", self.title)

            with open(html_file, "w") as f:
                f.write(content)

    def _create_fallback_html(self):
        """Create a minimal HTML fallback if no static files exist."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .error {{ color: #d32f2f; background: #ffebee; padding: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="error">
        <h2>Frontend Not Built</h2>
        <p>The React frontend has not been built yet. Please run:</p>
        <pre>cd src/chatan/viewer/frontend && npm run build</pre>
        <p>Then restart the viewer.</p>
    </div>
</body>
</html>"""

        with open(Path(self.temp_dir) / "index.html", "w") as f:
            f.write(html_content)

    async def _start_server(self):
        """Start the async HTTP server."""
        self.app = web.Application()

        # Add CORS
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            },
        )

        # API routes for React app
        self.app.router.add_get("/api/columns", self._get_columns)
        self.app.router.add_post("/api/columns", self._create_column)
        self.app.router.add_put("/api/columns/{name}", self._update_column)
        self.app.router.add_delete("/api/columns/{name}", self._delete_column)

        self.app.router.add_post("/api/generate", self._start_generation)
        self.app.router.add_post("/api/generate/stop", self._stop_generation)
        self.app.router.add_get("/api/generate/status", self._get_generation_status)

        self.app.router.add_get("/api/data", self._serve_data)
        self.app.router.add_get("/api/export/{format}", self._export_data)

        # Legacy routes for compatibility
        self.app.router.add_get("/api/schema", self._serve_schema)
        self.app.router.add_post("/api/update_schema", self._update_schema)

        # Explicit route for root to serve index
        async def serve_index(request):
            index_path = Path(self.temp_dir) / "index.html"
            return web.FileResponse(index_path)

        # Serve static files
        self.app.router.add_get("/", serve_index)
        self.app.router.add_static("/", self.temp_dir, name="static")

        # Add CORS to routes
        for route in list(self.app.router.routes()):
            cors.add(route)

        # Start server
        for port in range(8000, 8100):
            try:
                self.runner = web.AppRunner(self.app)
                await self.runner.setup()
                self.site = web.TCPSite(self.runner, "localhost", port)
                await self.site.start()
                self.port = port
                break
            except OSError:
                if self.runner:
                    await self.runner.cleanup()
                continue

    # Column management API handlers
    async def _get_columns(self, request):
        """Get all columns with their definitions."""
        try:
            with open(self.columns_file, "r") as f:
                columns_data = json.load(f)
            return web.json_response(columns_data)
        except Exception as e:
            return web.json_response({"columns": {}})

    async def _create_column(self, request):
        """Create a new column."""
        try:
            column_data = await request.json()

            # Validate column data
            if not column_data.get("name") or not column_data.get("type"):
                return web.json_response(
                    {"error": "Name and type are required"}, status=400
                )

            # Add to UI columns
            self.ui_columns.append(column_data)

            # Save to file
            columns_dict = {col["name"]: col for col in self.ui_columns}
            with open(self.columns_file, "w") as f:
                json.dump({"columns": columns_dict}, f)

            return web.json_response({"status": "created", "column": column_data})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _update_column(self, request):
        """Update an existing column."""
        try:
            column_name = request.match_info["name"]
            column_data = await request.json()

            # Find and update column
            for i, col in enumerate(self.ui_columns):
                if col["name"] == column_name:
                    self.ui_columns[i] = column_data
                    break
            else:
                return web.json_response({"error": "Column not found"}, status=404)

            # Save to file
            columns_dict = {col["name"]: col for col in self.ui_columns}
            with open(self.columns_file, "w") as f:
                json.dump({"columns": columns_dict}, f)

            return web.json_response({"status": "updated", "column": column_data})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _delete_column(self, request):
        """Delete a column."""
        try:
            column_name = request.match_info["name"]

            # Remove from UI columns
            self.ui_columns = [
                col for col in self.ui_columns if col["name"] != column_name
            ]

            # Save to file
            columns_dict = {col["name"]: col for col in self.ui_columns}
            with open(self.columns_file, "w") as f:
                json.dump({"columns": columns_dict}, f)

            return web.json_response({"status": "deleted"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    # Generation control API handlers
    async def _start_generation(self, request):
        """Start dataset generation using Chatan."""
        if self.is_generating:
            return web.json_response(
                {"error": "Generation already in progress"}, status=400
            )

        try:
            gen_params = await request.json()
            ui_columns = gen_params.get("columns", [])
            row_count = gen_params.get("rowCount", 10)

            if not ui_columns:
                return web.json_response({"error": "No columns provided"}, status=400)

            # Convert UI columns to chatan schema
            chatan_schema = self._ui_to_chatan_schema(ui_columns)

            if not chatan_schema:
                return web.json_response(
                    {"error": "No valid columns to generate"}, status=400
                )

            # Create chatan dataset
            self.current_dataset = dataset(chatan_schema, n=row_count)

            # Clear previous data
            with open(self.data_file, "w") as f:
                json.dump({"rows": [], "completed": False, "current_rows": {}}, f)

            # Start generation task
            self.is_generating = True
            self.generation_progress = {"current": 0, "total": row_count}
            self.generation_task = asyncio.create_task(self._run_generation(row_count))

            return web.json_response({"status": "started", "total": row_count})

        except Exception as e:
            self.is_generating = False
            return web.json_response({"error": str(e)}, status=400)

    async def _run_generation(self, row_count: int):
        """Run the actual generation process."""
        try:
            # Generate data using chatan
            for i in range(row_count):
                if not self.is_generating:  # Check if stopped
                    break

                # Generate one row
                single_row_dataset = dataset(self.current_dataset.schema, n=1)
                df = single_row_dataset.generate()

                if len(df) > 0:
                    row_data = df.iloc[0].to_dict()

                    # Update data file
                    self._add_generated_row(row_data)

                    # Update progress
                    self.generation_progress["current"] = i + 1

                # Small delay to simulate real-time generation
                await asyncio.sleep(0.1)

            # Mark as completed
            self.is_generating = False
            self._mark_generation_complete()

        except Exception as e:
            print(f"Generation error: {e}")
            self.is_generating = False
            self._mark_generation_complete()

    def _add_generated_row(self, row_data: Dict[str, Any]):
        """Add a generated row to the data file."""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}

        data["rows"].append(row_data)

        with open(self.data_file, "w") as f:
            json.dump(data, f)

    def _mark_generation_complete(self):
        """Mark generation as complete."""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}

        data["completed"] = True
        data["current_rows"] = {}

        with open(self.data_file, "w") as f:
            json.dump(data, f)

    async def _stop_generation(self, request):
        """Stop current generation."""
        try:
            self.is_generating = False
            if self.generation_task:
                self.generation_task.cancel()
            self._mark_generation_complete()
            return web.json_response({"status": "stopped"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _get_generation_status(self, request):
        """Get current generation status."""
        try:
            return web.json_response(
                {
                    "status": "generating" if self.is_generating else "idle",
                    "progress": self.generation_progress["current"],
                    "total": self.generation_progress["total"],
                    "current_row": self.generation_progress["current"],
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # Data serving API handlers
    async def _serve_data(self, request):
        """Serve the data JSON file."""
        try:
            with open(self.data_file, "r") as f:
                content = f.read()
            return web.Response(text=content, content_type="application/json")
        except:
            return web.Response(
                text='{"rows": [], "completed": false, "current_rows": {}}',
                content_type="application/json",
            )

    async def _export_data(self, request):
        """Export data in requested format."""
        format_type = request.match_info["format"]

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)

            rows = data.get("rows", [])
            if not rows:
                return web.json_response({"error": "No data to export"}, status=400)

            if format_type == "csv":
                import pandas as pd

                df = pd.DataFrame(rows)
                csv_content = df.to_csv(index=False)
                return web.Response(
                    text=csv_content,
                    content_type="text/csv",
                    headers={
                        "Content-Disposition": 'attachment; filename="dataset.csv"'
                    },
                )

            elif format_type == "json":
                return web.Response(
                    text=json.dumps(rows, indent=2),
                    content_type="application/json",
                    headers={
                        "Content-Disposition": 'attachment; filename="dataset.json"'
                    },
                )

            else:
                return web.json_response({"error": "Unsupported format"}, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # Legacy API handlers for compatibility
    async def _serve_schema(self, request):
        """Serve the schema metadata (legacy)."""
        try:
            with open(self.columns_file, "r") as f:
                content = f.read()
            return web.Response(text=content, content_type="application/json")
        except:
            return web.Response(text='{"columns": {}}', content_type="application/json")

    async def _update_schema(self, request):
        """Update schema (legacy)."""
        try:
            data = await request.json()
            with open(self.schema_file, "w") as f:
                json.dump(data, f)
            return web.Response(
                text='{"status": "updated"}', content_type="application/json"
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    # Data management methods (legacy compatibility)
    def start_row(self, row_index: int):
        """Start a new row with empty cells."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(lambda data: self._add_current_row(data, row_index))

    def update_cell(self, row_index: int, column: str, value: Any):
        """Update a single cell in a specific row."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(
            lambda data: self._update_cell_data(data, row_index, column, value)
        )

    def complete_row(self, row_index: int, row_data: Dict[str, Any]):
        """Mark a row as complete."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(
            lambda data: self._complete_row_data(data, row_index, row_data)
        )

    def complete(self):
        """Mark generation as complete."""
        if not self._active or not self.data_file:
            return
        self._update_data_file(lambda data: self._mark_complete(data))

    def _update_data_file(self, update_func):
        """Helper to update data file with error handling."""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
        except:
            data = {"rows": [], "completed": False, "current_rows": {}}

        update_func(data)

        with open(self.data_file, "w") as f:
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

    async def stop(self):
        """Stop the viewer and cleanup resources."""
        self._active = False
        self.is_generating = False
        if self.generation_task:
            self.generation_task.cancel()
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()


# Factory function for easy use
async def launch_viewer(
    title: str = "Chatan Dataset Builder",
    provider: str = None,
    api_key: str = None,
    auto_open: bool = True,
    **generator_kwargs,
) -> LiveViewer:
    """Launch the viewer with optional LLM configuration.

    Additional keyword arguments are forwarded to
    :meth:`LiveViewer.set_generator_client` to configure model
    selection or provider specific settings such as Azure
    endpoints.
    """
    viewer = LiveViewer(title=title, auto_open=auto_open)

    # Set up generator if provided
    if provider and api_key:
        viewer.set_generator_client(provider, api_key, **generator_kwargs)

    # Start with empty schema
    url = await viewer.start({})
    print(f"Viewer started at: {url}")

    return viewer
