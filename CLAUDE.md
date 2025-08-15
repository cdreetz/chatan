# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatan is a Python library for creating synthetic datasets using LLM generators and sampling functions. Users define dataset schemas with generators (typically LLM prompts) and samplers, then generate structured data with dependency resolution and optional live web viewer for real-time monitoring.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/chatan

# Run specific test file
pytest tests/test_dataset.py

# Run specific test class/method
pytest tests/test_generator.py::TestOpenAIGenerator::test_init_default_model
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Type checking with MyPy
mypy src/chatan/
```

### Documentation
```bash
# Build Sphinx documentation
cd docs/
make html

# Clean documentation build
make clean
```

### Development Setup
Uses UV for dependency management. Install development dependencies:
```bash
uv sync --dev
```

## Architecture

### Core Components

**Dataset (`src/chatan/dataset.py`)**: Central class that manages schema definitions and data generation. Builds dependency graphs from schema references (e.g., `{column_name}` in prompts) and executes topological sort for proper generation order. Supports both sync and async generation modes.

**Generator (`src/chatan/generator.py`)**: Abstracts LLM providers (OpenAI, Anthropic) with both sync and async interfaces. `GeneratorFunction` wraps prompts with template substitution and context variables. Supports prompt templates with `{variable}` placeholders that get resolved from row context.

**Sampler (`src/chatan/sampler.py`)**: Provides non-LLM data generation functions like `choice()`, `uuid()`, `range()`, `datetime()`, `from_dataset()`, etc. All samplers implement the same callable interface for consistent schema integration.

**Viewer (`src/chatan/viewer/`)**: Live web interface for monitoring dataset generation in real-time. Uses async aiohttp server to serve HTML/CSS/JS interface. Shows progress, partial results, and allows schema modifications during generation.

### Data Flow

1. User defines schema with mix of generators and samplers
2. Dataset builds dependency graph by parsing `{variable}` references in generator prompts
3. Topological sort determines column execution order
4. For each row: execute columns in dependency order, passing completed column values as context
5. Optional viewer streams progress via JSON API and WebSocket-like polling

### Key Patterns

- **Template Resolution**: Generator prompts use `{column_name}` syntax, resolved from row context during generation
- **Async Support**: All generators support both sync and async modes for batch processing
- **Provider Abstraction**: Generator factory handles OpenAI/Anthropic differences transparently
- **Static Files**: Viewer copies HTML/CSS/JS from `src/chatan/viewer/static/` to temp directory for serving

## Testing Structure

Tests are organized by module (`test_dataset.py`, `test_generator.py`, `test_sampler.py`) with comprehensive coverage including edge cases, async functionality, and integration tests. Mock external API calls for deterministic testing.

## Viewer v2.0 Planning - Excel-like Interface for Non-Developers

### Vision
Transform the current viewer from a basic monitoring tool into a full-featured, Excel-like interface that enables non-developers to create synthetic datasets through an intuitive spreadsheet UI. Users should feel like they're using familiar spreadsheet software with AI-powered enhancements.

### Core User Experience
- **Familiar Interface**: Look and feel identical to Excel/Google Sheets
- **Column Header Prompts**: Click column headers to define generation logic via natural language prompts
- **Real-time Generation**: Generate data on-demand as users define columns
- **No Learning Curve**: Leverage existing spreadsheet knowledge rather than requiring new concepts

### Technical Architecture

#### Frontend (React-based)
- **Grid Component**: Excel-like spreadsheet interface with resizable columns, cell selection, keyboard navigation
- **Column Definition Modal**: Rich text editor for prompts with syntax highlighting and autocomplete
- **Generation Controls**: Play/pause/stop buttons, batch size controls, progress indicators
- **Data Export**: CSV, Excel, JSON export with format preview
- **Template Gallery**: Pre-built column templates for common use cases

#### Backend Integration
- **Schema Translation**: Convert UI column definitions to chatan schema format
- **Streaming Updates**: WebSocket connection for real-time cell population
- **Smart Batching**: Optimize generation order based on dependencies and user interaction
- **State Persistence**: Save/load workspace sessions with schema definitions

### Column Definition Interface

#### Prompt Mode
```
Column Header: "Customer Name"
Definition: "Generate realistic customer names for a B2B software company"
```

#### Sample Mode  
```
Column Header: "Status"
Definition: [Active, Inactive, Pending] (with optional weights)
```

#### Reference Mode
```
Column Header: "Email"  
Definition: "Create email for {Customer Name} at their company"
```

### Key Features

#### 1. Smart Column Dependencies
- Automatically detect column references in prompts (e.g., `{Customer Name}`)
- Visual dependency indicators in column headers
- Automatic reordering for proper generation sequence

#### 2. Interactive Data Types
- **Text Generation**: LLM prompts with context awareness
- **Choice Sampling**: Dropdown lists with optional weighting
- **Numeric Ranges**: Min/max sliders with distribution options
- **Date Ranges**: Calendar pickers with realistic date generation
- **File References**: Upload CSV/Excel to sample from existing data

#### 3. Real-time Collaboration
- Multiple users can define columns simultaneously
- Live cursor indicators showing who's editing what
- Schema change notifications with conflict resolution

#### 4. Generation Controls
- **Preview Mode**: Generate 3-5 sample rows before full generation
- **Incremental Generation**: Add rows on-demand without regenerating existing data
- **Quality Controls**: Regenerate individual cells or entire columns
- **Export Pipeline**: One-click export to various formats

### Implementation Plan

#### MVP - Build Now
**Core Excel-like Interface**
- Basic spreadsheet grid with resizable columns and cell navigation
- Click column headers to open definition modal
- Three column definition modes:
  - **Prompt**: Text input for LLM generation (e.g., "Generate customer names")
  - **Choice**: List input for sampling (e.g., [Active, Inactive, Pending])  
  - **Reference**: Template with other column references (e.g., "Email for {Customer Name}")
- Generate button to populate N rows using existing chatan backend
- Basic progress indicator showing generation status
- Simple CSV/JSON export
- Automatic dependency detection from `{column}` references in prompts

**Technical Approach**
- Replace current HTML/CSS/JS with React frontend
- Keep existing aiohttp backend, add endpoints for column schema CRUD
- Direct integration with chatan Dataset class - no complex state management needed
- Simple client-side state for grid and column definitions
- Generate entire dataset at once (no streaming initially)

#### Phase 1.2 - File Reference Columns
**File-based Data Extraction**
- **File Path Column**: Column type for file paths/uploads (PDFs, images, documents)
- **File Reference Prompts**: Enhanced reference mode that can extract data from files
  - Example: "Extract applicant email from {resume_file}"
  - Example: "Summarize key points from {contract_pdf}"
  - Example: "Get company name from {invoice_file}"
- **File Upload UI**: Drag-drop interface for uploading files to populate file path columns
- **Document Processing**: Backend integration with document parsing (PDF text extraction, OCR for images)

**Technical Implementation**
- Extend generator system to accept file content as context alongside text variables
- Add file storage handling (temp directory or cloud storage for uploaded files)
- Integrate document parsing libraries (PyPDF2, python-docx, PIL/OCR)
- Enhanced GeneratorFunction that can process file content in prompts
- File column type in UI with upload widget and file path display

**Use Cases**
- Resume screening: Upload resumes → extract names, emails, skills, experience
- Invoice processing: Upload invoices → extract amounts, dates, vendor info
- Document analysis: Upload contracts → extract key terms, parties, dates
- Receipt processing: Upload receipts → extract merchant, amount, category

#### Future Enhancements
**Advanced UX**
- Real-time streaming of cell population during generation
- Preview mode (generate 3-5 sample rows first)
- Incremental generation (add more rows without regenerating)
- Undo/redo for schema changes
- Template gallery with pre-built column patterns
- Drag-drop file upload for sampling from existing data

**Collaboration & Scale**
- Real-time multiplayer editing with WebSocket
- Virtual scrolling for large datasets (1M+ rows)
- User authentication and workspace management  
- Session persistence and save/load functionality

**Enterprise Features**
- API integration for custom data sources
- Scheduled generation and pipeline integration
- Advanced analytics and data quality metrics
- Rate limiting and queue management for LLM requests

### Technical Notes for MVP
- Focus on familiar Excel keyboard shortcuts (Tab, Enter, arrow keys)
- Use existing chatan schema format internally - UI is just a visual layer
- Keep generation synchronous initially - users click generate and wait
- Handle small datasets (< 10k rows) without performance optimizations

### Success Metrics
- **Onboarding Time**: Users creating first dataset within 2 minutes
- **Feature Adoption**: 80% of users utilizing both prompts and sampling
- **Export Rate**: 90% of generated datasets get exported
- **Return Usage**: Users return to create additional datasets within 1 week