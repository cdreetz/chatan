// Main application state
let rowCount = 0;
let currentRowElements = {};
let schemaData = null;
let columns = [];

// Initialize the application
async function init() {
  await loadSchema();
  setupColumnHeaders();
  makeColumnsResizable(document.querySelector("table"));
  startDataPolling();
}

// Load schema metadata from the API
async function loadSchema() {
  try {
    const response = await fetch("/api/schema?" + new Date().getTime());
    schemaData = await response.json();
    columns = Object.keys(schemaData);
  } catch (error) {
    console.error("Error loading schema:", error);
  }
}

// Setup column headers with metadata
function setupColumnHeaders() {
  const headerRow = document.querySelector("thead tr");

  // Clear existing headers except row number
  while (headerRow.children.length > 1) {
    headerRow.removeChild(headerRow.lastChild);
  }

  // Add column headers with metadata
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.className = "column-header";

    const columnDiv = document.createElement("div");
    columnDiv.className = "column-name";
    columnDiv.textContent = column;

    const metaDiv = document.createElement("div");
    metaDiv.className = "column-meta";
    metaDiv.onclick = () => openEditModal(column);

    if (schemaData && schemaData[column]) {
      const meta = schemaData[column];
      if (meta.type === "generator") {
        metaDiv.classList.add("generator");
        metaDiv.textContent = `${meta.provider}: ${meta.prompt.substring(0, 50)}...`;
        metaDiv.title = meta.prompt;
      } else if (meta.type === "sample" || meta.type === "weighted_sample") {
        metaDiv.classList.add("sample");
        const choices = Array.isArray(meta.choices)
          ? meta.choices
          : Object.keys(meta.choices);
        metaDiv.textContent = `Sample: [${choices.slice(0, 3).join(", ")}${choices.length > 3 ? "..." : ""}]`;
        metaDiv.title = JSON.stringify(meta.choices, null, 2);
      } else {
        metaDiv.textContent = `${meta.type}: ${JSON.stringify(meta.value || meta.name || "unknown").substring(0, 30)}`;
      }
    } else {
      metaDiv.textContent = "unknown";
    }

    th.appendChild(columnDiv);
    th.appendChild(metaDiv);
    headerRow.appendChild(th);
  });
}

// Start polling for data updates
function startDataPolling() {
  fetchData();
}

// Fetch data from the API
async function fetchData() {
  try {
    const response = await fetch("/api/data?" + new Date().getTime());
    const data = await response.json();

    // Handle current rows updates
    if (data.current_rows) {
      Object.values(data.current_rows).forEach((currentRow) => {
        updateCurrentRow(currentRow);
      });
    }

    // Handle completed rows
    if (data.rows.length > rowCount) {
      rowCount = data.rows.length;
      updateStatus(data.completed);
    }

    if (data.completed) {
      document.getElementById("statusDot").classList.add("complete");
      document.getElementById("statusText").textContent = "Complete";
      currentRowElements = {};
      return;
    }
  } catch (error) {
    console.error("Error fetching data:", error);
  }

  // Continue polling
  setTimeout(fetchData, 100);
}

// Update a row that's currently being generated
function updateCurrentRow(currentRow) {
  const tbody = document.getElementById("tableBody");
  const rowIndex = currentRow.index;

  // Remove loading message if present
  if (
    tbody.children.length === 1 &&
    tbody.children[0].cells.length === columns.length + 1
  ) {
    tbody.innerHTML = "";
  }

  // Create new row element if we don't have one for this index
  if (!currentRowElements[rowIndex]) {
    const rowElement = document.createElement("tr");
    rowElement.className = "new-row";
    rowElement.dataset.rowIndex = rowIndex;

    // Row number
    const numCell = document.createElement("td");
    numCell.className = "row-number";
    numCell.textContent = rowIndex + 1;
    rowElement.appendChild(numCell);

    // Create empty cells for all columns
    columns.forEach((col) => {
      const td = document.createElement("td");
      td.className = "cell-content cell-generating";
      td.textContent = "...";
      td.id = `cell-${rowIndex}-${col}`;
      rowElement.appendChild(td);
    });

    // Insert in correct position (maintain row order)
    const existingRows = Array.from(tbody.children);
    let insertBefore = null;
    for (let i = 0; i < existingRows.length; i++) {
      const existingIndex = parseInt(existingRows[i].dataset.rowIndex);
      if (existingIndex > rowIndex) {
        insertBefore = existingRows[i];
        break;
      }
    }

    if (insertBefore) {
      tbody.insertBefore(rowElement, insertBefore);
    } else {
      tbody.appendChild(rowElement);
    }

    currentRowElements[rowIndex] = rowElement;
  }

  // Update cells with values
  Object.entries(currentRow.cells).forEach(([col, value]) => {
    const cell = document.getElementById(`cell-${rowIndex}-${col}`);
    if (cell) {
      cell.textContent = value || "";
      cell.classList.remove("cell-generating");
    }
  });
}

// Update the status display
function updateStatus(completed) {
  document.getElementById("rowCount").textContent = `${rowCount} rows`;
  if (completed) {
    document.getElementById("statusText").textContent = "Complete";
    document.getElementById("statusDot").classList.add("complete");
  }
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", init);
