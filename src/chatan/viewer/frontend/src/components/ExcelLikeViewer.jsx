import React, { useState, useEffect } from "react";
import {
  Download,
  Play,
  Square,
  Plus,
  Settings,
  FileText,
  List,
  Link,
  X,
} from "lucide-react";

const ExcelLikeViewer = () => {
  const [selectedCell, setSelectedCell] = useState({ row: 0, col: 0 });
  const [showColumnModal, setShowColumnModal] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Initialize with default A-Z columns like Excel
  const getDefaultColumns = () => {
    return Array.from({ length: 26 }, (_, i) => ({
      header: String.fromCharCode(65 + i), // A, B, C, ..., Z (display header)
      name: null, // Actual column name (null until configured)
      type: "empty",
      config: {},
    }));
  };

  const getDefaultRows = () => {
    return Array.from({ length: 10 }, (_, i) => ({})); // 10 empty rows
  };

  // Real state - starts with Excel-like defaults
  const [columns, setColumns] = useState(getDefaultColumns());
  const [data, setData] = useState(getDefaultRows());
  const [datasetTitle, setDatasetTitle] = useState("New Dataset");
  const [generationProgress, setGenerationProgress] = useState({
    current: 0,
    total: 0,
  });

  // Modal form state
  const [selectedColumnType, setSelectedColumnType] = useState("prompt");
  const [columnName, setColumnName] = useState("");
  const [columnConfig, setColumnConfig] = useState("");
  const [modalErrors, setModalErrors] = useState({});
  const [columnWidths, setColumnWidths] = useState({});
  const [isResizing, setIsResizing] = useState(false);
  const [resizingColumn, setResizingColumn] = useState(null);

  // Load initial state
  useEffect(() => {
    loadColumns();
    loadData();
    // Poll for generation status
    const interval = setInterval(checkGenerationStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  // Column resizing functionality
  const handleMouseDown = (e, columnName) => {
    e.preventDefault();
    setIsResizing(true);
    setResizingColumn(columnName);
    
    const startX = e.clientX;
    const startWidth = columnWidths[columnName] || 96; // Default width
    
    const handleMouseMove = (e) => {
      const diff = e.clientX - startX;
      const newWidth = Math.max(60, Math.min(300, startWidth + diff)); // Min 60px, max 300px
      setColumnWidths(prev => ({ ...prev, [columnName]: newWidth }));
    };
    
    const handleMouseUp = () => {
      setIsResizing(false);
      setResizingColumn(null);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const loadColumns = async () => {
    try {
      const response = await fetch("/api/columns");
      if (response.ok) {
        const result = await response.json();
        if (result.columns && Object.keys(result.columns).length > 0) {
          // Convert backend schema to UI format and merge with default columns
          const configuredColumns = Object.entries(result.columns).map(
            ([name, config]) => ({
              name,
              type: config.type || "prompt",
              config: config,
            }),
          );
          
          // Start with default A-Z columns, then replace configured ones
          const defaultCols = getDefaultColumns();
          const mergedColumns = defaultCols.map(defaultCol => {
            const configured = configuredColumns.find(c => c.name === defaultCol.header);
            return configured ? { ...configured, header: defaultCol.header } : defaultCol;
          });
          
          setColumns(mergedColumns);
        } else {
          // No configured columns, keep defaults
          setColumns(getDefaultColumns());
        }
      } else {
        // API error, keep defaults
        setColumns(getDefaultColumns());
      }
    } catch (error) {
      console.error("Failed to load columns:", error);
      // Error, keep defaults
      setColumns(getDefaultColumns());
    }
  };

  const loadData = async () => {
    try {
      const response = await fetch("/api/data");
      if (response.ok) {
        const result = await response.json();
        if (result.rows && result.rows.length > 0) {
          setData(result.rows);
        } else {
          // No data from backend, keep default empty rows
          setData(getDefaultRows());
        }
        if (result.completed) {
          setIsGenerating(false);
        }
      } else {
        // API error, keep defaults
        setData(getDefaultRows());
      }
    } catch (error) {
      console.error("Failed to load data:", error);
      // Error, keep defaults
      setData(getDefaultRows());
    }
  };

  const checkGenerationStatus = async () => {
    try {
      const response = await fetch("/api/generate/status");
      if (response.ok) {
        const status = await response.json();
        if (status.status === "generating") {
          setIsGenerating(true);
          setGenerationProgress({
            current: status.current_row || 0,
            total: status.total || 0,
          });
          // Refresh data
          loadData();
        } else {
          setIsGenerating(false);
        }
      }
    } catch (error) {
      console.error("Status check failed:", error);
    }
  };

  const getColumnIcon = (type) => {
    switch (type) {
      case "prompt":
        return <FileText className="w-3 h-3" />;
      case "choice":
        return <List className="w-3 h-3" />;
      case "reference":
        return <Link className="w-3 h-3" />;
      case "empty":
        return null;
      default:
        return <FileText className="w-3 h-3" />;
    }
  };

  const getColumnBadgeColor = (type) => {
    switch (type) {
      case "prompt":
        return "bg-blue-100 text-blue-700 border-blue-200";
      case "choice":
        return "bg-green-100 text-green-700 border-green-200";
      case "reference":
        return "bg-purple-100 text-purple-700 border-purple-200";
      case "empty":
        return "bg-gray-50 text-gray-400 border-gray-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  const getColumnDescription = (col) => {
    switch (col.type) {
      case "prompt":
        return `AI: ${(col.config.prompt || "Generate text").substring(0, 25)}...`;
      case "choice":
        if (
          typeof col.config.choices === "object" &&
          !Array.isArray(col.config.choices)
        ) {
          return `Weighted: ${Object.keys(col.config.choices).slice(0, 2).join(", ")}...`;
        } else if (Array.isArray(col.config.choices)) {
          return `Sample: ${col.config.choices.slice(0, 2).join(", ")}...`;
        }
        return "Choice: Click to configure";
      case "reference":
        return `Ref: ${(col.config.template || "Reference other columns").substring(0, 25)}...`;
      case "empty":
        return "";
      default:
        return "Click to configure";
    }
  };

  const handleAddColumn = () => {
    setSelectedColumn(null);
    setColumnName("");
    setColumnConfig("");
    setSelectedColumnType("prompt");
    setModalErrors({});
    setShowColumnModal(true);
  };

  const handleEditColumn = (column) => {
    setSelectedColumn(column);
    setColumnName(column.name || ""); // Use actual name or empty for new columns
    setSelectedColumnType(column.type === "empty" ? "prompt" : column.type);

    // Set config based on type
    if (column.type === "prompt") {
      setColumnConfig(column.config.prompt || "");
    } else if (column.type === "choice") {
      if (Array.isArray(column.config.choices)) {
        setColumnConfig(column.config.choices.join(", "));
      } else {
        setColumnConfig(JSON.stringify(column.config.choices, null, 2));
      }
    } else if (column.type === "reference") {
      setColumnConfig(column.config.template || "");
    } else {
      setColumnConfig(""); // Empty column
    }

    setModalErrors({});
    setShowColumnModal(true);
  };

  const validateColumnForm = () => {
    const errors = {};

    if (!columnName.trim()) {
      errors.name = "Column name is required";
    }

    if (!columnConfig.trim()) {
      if (selectedColumnType === "prompt") {
        errors.config = "Prompt template is required";
      } else if (selectedColumnType === "choice") {
        errors.config = "Choices are required";
      } else if (selectedColumnType === "reference") {
        errors.config = "Reference template is required";
      }
    }

    // Check for duplicate column names (excluding current edit)
    if (
      columns.some(
        (col) =>
          col.name === columnName.trim() && col.name !== selectedColumn?.name,
      )
    ) {
      errors.name = "Column name already exists";
    }

    setModalErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSaveColumn = async () => {
    if (!validateColumnForm()) return;

    // Build column object in UI format
    const columnData = {
      name: columnName.trim(),
      type: selectedColumnType,
      config: {},
    };

    // Set config based on type
    if (selectedColumnType === "prompt") {
      columnData.config.prompt = columnConfig.trim();
    } else if (selectedColumnType === "choice") {
      try {
        // Try to parse as JSON first (for weighted choices)
        columnData.config.choices = JSON.parse(columnConfig);
      } catch {
        // Fall back to comma-separated list
        columnData.config.choices = columnConfig
          .split(",")
          .map((s) => s.trim())
          .filter((s) => s);
      }
    } else if (selectedColumnType === "reference") {
      columnData.config.template = columnConfig.trim();
    }

    try {
      // Determine if this is creating a new column or updating existing
      const isNewColumn = !selectedColumn.name; // New if no name was set before
      const method = isNewColumn ? "POST" : "PUT";
      const url = isNewColumn
        ? "/api/columns"
        : `/api/columns/${selectedColumn.name}`;

      const response = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(columnData),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Save column failed:", response.status, errorText);
        throw new Error(`Failed to save column: ${response.status} - ${errorText}`);
      }

      // Update local state
      setColumns(
        columns.map((col) =>
          col.header === selectedColumn.header ? { ...columnData, header: col.header } : col,
        ),
      );

      setShowColumnModal(false);
    } catch (error) {
      console.error("Error saving column:", error);
      setModalErrors({ save: `Failed to save column: ${error.message}` });
    }
  };

  const handleDeleteColumn = async (columnToDelete) => {
    if (!confirm(`Delete column "${columnToDelete.name}"?`)) return;

    try {
      const response = await fetch(`/api/columns/${columnToDelete.name}`, {
        method: "DELETE",
      });

      if (response.ok) {
        setColumns(columns.filter((col) => col.name !== columnToDelete.name));
        // Clear any data for this column
        setData(
          data.map((row) => {
            const newRow = { ...row };
            delete newRow[columnToDelete.name];
            return newRow;
          }),
        );
      }
    } catch (error) {
      console.error("Error deleting column:", error);
    }
  };

  const handleGenerate = async () => {
    if (!hasConfiguredColumns) {
      alert("Please configure at least one column before generating");
      return;
    }

    const rowCount = prompt("How many rows to generate?", "10");
    if (!rowCount || isNaN(rowCount)) return;

    setIsGenerating(true);
    setData(getDefaultRows()); // Reset to empty rows

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          columns: columns.filter(col => col.type !== "empty"), // Only send configured columns
          rowCount: parseInt(rowCount),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to start generation");
      }

      // Generation started - status polling will handle updates
    } catch (error) {
      console.error("Generation failed:", error);
      setIsGenerating(false);
      alert("Failed to start generation. Please try again.");
    }
  };

  const handleStopGeneration = async () => {
    try {
      await fetch("/api/generate/stop", { method: "POST" });
      setIsGenerating(false);
    } catch (error) {
      console.error("Failed to stop generation:", error);
    }
  };

  const handleExport = async (format = "csv") => {
    if (data.length === 0) {
      alert("No data to export");
      return;
    }

    try {
      const response = await fetch(`/api/export/${format}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `dataset.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  // Check if we have any configured columns (non-empty)
  const hasConfiguredColumns = columns.some(col => col.type !== "empty");
  // Always show spreadsheet view, never show empty state with A-Z columns
  const isEmpty = false;

  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900">
              {datasetTitle}
            </h1>
            <p className="text-sm text-gray-500 mt-1">
              {isGenerating ? (
                <span className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  Generating row {generationProgress.current} of{" "}
                  {generationProgress.total}...
                </span>
              ) : (
                <>
                  {!hasConfiguredColumns
                    ? "No columns defined"
                    : `${columns.filter(col => col.type !== "empty").length} columns configured`}
                  {data.some(row => Object.keys(row).length > 0) && ` • ${data.filter(row => Object.keys(row).length > 0).length} rows generated`}
                </>
              )}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              className={`px-4 py-2 rounded-lg flex items-center gap-2 font-medium ${
                !hasConfiguredColumns || isGenerating
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-blue-600 text-white hover:bg-blue-700"
              }`}
              onClick={isGenerating ? handleStopGeneration : handleGenerate}
              disabled={!hasConfiguredColumns}
            >
              {isGenerating ? (
                <>
                  <Square className="w-4 h-4" />
                  Stop
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Generate
                </>
              )}
            </button>

            <button
              className={`px-4 py-2 border rounded-lg flex items-center gap-2 ${
                !data.some(row => Object.keys(row).length > 0)
                  ? "border-gray-300 text-gray-400 cursor-not-allowed"
                  : "border-gray-300 text-gray-700 hover:bg-gray-50"
              }`}
              onClick={() => handleExport("csv")}
              disabled={!data.some(row => Object.keys(row).length > 0)}
            >
              <Download className="w-4 h-4" />
              Export
            </button>

            <button className="p-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50">
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        {isEmpty ? (
          // Empty state
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Plus className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Start Building Your Dataset
              </h3>
              <p className="text-gray-500 mb-6">
                Add columns to define what kind of data you want to generate.
                You can use AI prompts, choice lists, or reference other
                columns.
              </p>
              <button
                onClick={handleAddColumn}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 mx-auto font-medium"
              >
                <Plus className="w-5 h-5" />
                Add Your First Column
              </button>
            </div>
          </div>
        ) : (
          // Spreadsheet view
          <div className="h-full overflow-auto">
            <table className="w-full border-collapse table-fixed">
              {/* Header Row */}
              <thead className="sticky top-0 z-10">
                <tr>
                  {/* Row number header */}
                  <th className="w-10 bg-gray-100 border border-gray-300 p-0">
                    <div className="h-6 flex items-center justify-center text-xs font-medium text-gray-500">
                      #
                    </div>
                  </th>

                  {/* Column headers */}
                  {columns.map((col, index) => (
                    <th
                      key={col.header}
                      className={`border border-gray-300 p-0 relative group cursor-pointer hover:bg-gray-200 ${
                        col.type === "empty" ? "bg-gray-50" : "bg-blue-50"
                      }`}
                      onClick={() => handleEditColumn(col)}
                      style={{ 
                        width: columnWidths[col.header] || '96px',
                        minWidth: '60px',
                        maxWidth: '300px'
                      }}
                    >
                      <div className="h-6 px-2 flex items-center justify-center">
                        <div className="flex items-center gap-1 w-full">
                          {col.type !== "empty" && getColumnIcon(col.type)}
                          <span className={`font-medium text-xs truncate ${
                            col.type === "empty" ? "text-gray-500" : "text-gray-900"
                          }`}>
                            {col.name || col.header}
                          </span>
                          {col.name && col.name !== col.header && (
                            <div className="text-xs text-gray-400 truncate">
                              {col.header}
                            </div>
                          )}
                          {col.type !== "empty" && (
                            <div className="w-2 h-2 bg-blue-500 rounded-full ml-auto"></div>
                          )}
                          {col.type !== "empty" && (
                            <button
                              className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-700"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteColumn(col);
                              }}
                            >
                              <X className="w-3 h-3" />
                            </button>
                          )}
                        </div>
                      </div>

                      {col.type !== "empty" && getColumnDescription(col) && (
                        <div className="absolute top-7 left-0 right-0 z-20 bg-gray-800 text-white text-xs p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                          {getColumnDescription(col)}
                        </div>
                      )}

                      {/* Column resizer */}
                      <div 
                        className="absolute right-0 top-0 w-2 h-full cursor-col-resize hover:bg-blue-500 opacity-0 group-hover:opacity-100 z-10"
                        onMouseDown={(e) => handleMouseDown(e, col.header)}
                        onClick={(e) => e.stopPropagation()}
                      ></div>
                    </th>
                  ))}

                  {/* Add column button */}
                  <th className="w-10 bg-gray-50 border border-gray-300 p-0">
                    <button
                      className="h-6 w-full flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                      onClick={handleAddColumn}
                    >
                      <Plus className="w-3 h-3" />
                    </button>
                  </th>
                </tr>
              </thead>

              {/* Data rows */}
              <tbody>
                {data.map((row, rowIndex) => (
                  <tr key={rowIndex} className="hover:bg-blue-50">
                    <td className="w-10 bg-gray-50 border border-gray-300 text-center text-xs text-gray-500 font-medium">
                      {rowIndex + 1}
                    </td>

                    {columns.map((col, colIndex) => (
                      <td
                        key={`${rowIndex}-${colIndex}`}
                        className={`border border-gray-300 p-1 cursor-cell relative text-xs ${
                          selectedCell.row === rowIndex &&
                          selectedCell.col === colIndex
                            ? "ring-2 ring-blue-500 bg-blue-50"
                            : ""
                        }`}
                        onClick={() =>
                          setSelectedCell({ row: rowIndex, col: colIndex })
                        }
                        style={{ height: '20px' }}
                      >
                        <div className="h-4 flex items-center overflow-hidden">
                          <span className="text-gray-900 truncate">
                            {row[col.name || col.header] || ""}
                          </span>
                        </div>
                      </td>
                    ))}

                    <td className="w-10 border border-gray-300 bg-gray-50"></td>
                  </tr>
                ))}

                {/* Generating row indicator */}
                {isGenerating && (
                  <tr>
                    <td className="w-10 bg-gray-50 border border-gray-300 text-center text-xs text-gray-500">
                      {data.filter(row => Object.keys(row).length > 0).length + 1}
                    </td>
                    {columns.map((col, index) => (
                      <td
                        key={index}
                        className={`border border-gray-300 p-1 ${
                          col.type !== "empty" 
                            ? "bg-gradient-to-r from-gray-100 via-gray-200 to-gray-100 animate-pulse" 
                            : "bg-gray-50"
                        }`}
                        style={{ height: '20px' }}
                      >
                        {col.type !== "empty" && (
                          <span className="text-gray-400 italic text-xs">
                            Generating...
                          </span>
                        )}
                      </td>
                    ))}
                    <td className="w-10 border border-gray-300 bg-gray-50"></td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Column Definition Modal */}
      {showColumnModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl mx-4 max-h-96 overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">
                {selectedColumn
                  ? `Edit Column: ${selectedColumn.name}`
                  : "Add New Column"}
              </h2>
            </div>

            <div className="p-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Column Name *
                  </label>
                  <input
                    type="text"
                    className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                      modalErrors.name ? "border-red-300" : "border-gray-300"
                    }`}
                    placeholder="e.g., customer_name"
                    value={columnName}
                    onChange={(e) => setColumnName(e.target.value)}
                  />
                  {modalErrors.name && (
                    <p className="text-red-500 text-xs mt-1">
                      {modalErrors.name}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Column Type *
                  </label>
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      type="button"
                      className={`p-3 border-2 rounded-lg text-left ${
                        selectedColumnType === "prompt"
                          ? "border-blue-500 bg-blue-50"
                          : "border-gray-300 hover:border-blue-500 hover:bg-blue-50"
                      }`}
                      onClick={() => setSelectedColumnType("prompt")}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <FileText
                          className={`w-4 h-4 ${selectedColumnType === "prompt" ? "text-blue-600" : "text-gray-600"}`}
                        />
                        <span
                          className={`font-medium ${selectedColumnType === "prompt" ? "text-blue-600" : "text-gray-900"}`}
                        >
                          Prompt
                        </span>
                      </div>
                      <span
                        className={`text-xs ${selectedColumnType === "prompt" ? "text-blue-600" : "text-gray-600"}`}
                      >
                        AI generates text
                      </span>
                    </button>

                    <button
                      type="button"
                      className={`p-3 border-2 rounded-lg text-left ${
                        selectedColumnType === "choice"
                          ? "border-green-500 bg-green-50"
                          : "border-gray-300 hover:border-green-500 hover:bg-green-50"
                      }`}
                      onClick={() => setSelectedColumnType("choice")}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <List
                          className={`w-4 h-4 ${selectedColumnType === "choice" ? "text-green-600" : "text-gray-600"}`}
                        />
                        <span
                          className={`font-medium ${selectedColumnType === "choice" ? "text-green-600" : "text-gray-900"}`}
                        >
                          Choice
                        </span>
                      </div>
                      <span
                        className={`text-xs ${selectedColumnType === "choice" ? "text-green-600" : "text-gray-600"}`}
                      >
                        Pick from list
                      </span>
                    </button>

                    <button
                      type="button"
                      className={`p-3 border-2 rounded-lg text-left ${
                        selectedColumnType === "reference"
                          ? "border-purple-500 bg-purple-50"
                          : "border-gray-300 hover:border-purple-500 hover:bg-purple-50"
                      }`}
                      onClick={() => setSelectedColumnType("reference")}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Link
                          className={`w-4 h-4 ${selectedColumnType === "reference" ? "text-purple-600" : "text-gray-600"}`}
                        />
                        <span
                          className={`font-medium ${selectedColumnType === "reference" ? "text-purple-600" : "text-gray-900"}`}
                        >
                          Reference
                        </span>
                      </div>
                      <span
                        className={`text-xs ${selectedColumnType === "reference" ? "text-purple-600" : "text-gray-600"}`}
                      >
                        Use other columns
                      </span>
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {selectedColumnType === "prompt" && "Prompt Template *"}
                    {selectedColumnType === "choice" && "Choices *"}
                    {selectedColumnType === "reference" &&
                      "Reference Template *"}
                  </label>

                  {selectedColumnType === "prompt" && (
                    <textarea
                      className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                        modalErrors.config
                          ? "border-red-300"
                          : "border-gray-300"
                      }`}
                      rows={3}
                      placeholder="Generate realistic customer names for a B2B software company"
                      value={columnConfig}
                      onChange={(e) => setColumnConfig(e.target.value)}
                    />
                  )}

                  {selectedColumnType === "choice" && (
                    <div>
                      <textarea
                        className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                          modalErrors.config
                            ? "border-red-300"
                            : "border-gray-300"
                        }`}
                        rows={3}
                        placeholder={
                          'Simple list: Startup, SMB, Enterprise\n\nWeighted JSON:\n{"Active": 0.7, "Inactive": 0.3}'
                        }
                        value={columnConfig}
                        onChange={(e) => setColumnConfig(e.target.value)}
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Comma-separated list or JSON for weighted choices
                      </p>
                    </div>
                  )}

                  {selectedColumnType === "reference" && (
                    <textarea
                      className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                        modalErrors.config
                          ? "border-red-300"
                          : "border-gray-300"
                      }`}
                      rows={3}
                      placeholder="Create professional email for {customer_name}"
                      value={columnConfig}
                      onChange={(e) => setColumnConfig(e.target.value)}
                    />
                  )}

                  {modalErrors.config && (
                    <p className="text-red-500 text-xs mt-1">
                      {modalErrors.config}
                    </p>
                  )}

                  <p className="text-xs text-gray-500 mt-1">
                    Use {"{column_name}"} to reference other columns
                  </p>
                </div>

                {modalErrors.save && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-700 text-sm">{modalErrors.save}</p>
                  </div>
                )}
              </div>
            </div>

            <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
              <button
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg"
                onClick={() => setShowColumnModal(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                onClick={handleSaveColumn}
              >
                Save Column
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExcelLikeViewer;
