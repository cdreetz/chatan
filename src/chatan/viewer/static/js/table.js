// Table functionality for resizable columns and data display

// Make table columns resizable
function makeColumnsResizable(table) {
  const headers = table.querySelectorAll("th");
  headers.forEach((th, index) => {
    // Skip the last column
    if (index === headers.length - 1) return;

    const resizer = document.createElement("div");
    resizer.className = "col-resizer";
    th.appendChild(resizer);

    let startX, startWidth;

    resizer.addEventListener("mousedown", (e) => {
      e.preventDefault();
      startX = e.clientX;
      startWidth = th.offsetWidth;
      document.addEventListener("mousemove", doDrag);
      document.addEventListener("mouseup", stopDrag);

      // Prevent text selection during resize
      document.body.style.userSelect = "none";
    });

    function doDrag(e) {
      const width = Math.max(50, startWidth + e.clientX - startX); // Minimum width of 50px
      th.style.width = width + "px";
    }

    function stopDrag() {
      document.removeEventListener("mousemove", doDrag);
      document.removeEventListener("mouseup", stopDrag);
      document.body.style.userSelect = "";
    }
  });
}

// Add a completed row to the table
function addCompletedRow(rowData, rowIndex) {
  const tbody = document.getElementById("tableBody");

  // Remove loading message if present
  if (
    tbody.children.length === 1 &&
    tbody.children[0].classList.contains("loading")
  ) {
    tbody.innerHTML = "";
  }

  const tr = document.createElement("tr");
  tr.dataset.rowIndex = rowIndex;

  // Row number
  const numCell = document.createElement("td");
  numCell.className = "row-number";
  numCell.textContent = rowIndex + 1;
  tr.appendChild(numCell);

  // Data cells
  columns.forEach((col) => {
    const td = document.createElement("td");
    td.className = "cell-content";
    td.textContent = rowData[col] || "";
    tr.appendChild(td);
  });

  tbody.appendChild(tr);
}

// Clear the table
function clearTable() {
  const tbody = document.getElementById("tableBody");
  tbody.innerHTML = `
        <tr>
            <td colspan="100%" class="loading">
                Waiting for data...
            </td>
        </tr>
    `;
}

// Update table with new data
function updateTable(data) {
  const tbody = document.getElementById("tableBody");

  // If we have completed rows, make sure they're displayed
  if (data.rows && data.rows.length > 0) {
    // Clear loading if present
    if (
      tbody.children.length === 1 &&
      tbody.children[0].classList.contains("loading")
    ) {
      tbody.innerHTML = "";
    }

    // Add any missing completed rows
    data.rows.forEach((row, index) => {
      const existingRow = tbody.querySelector(`tr[data-row-index="${index}"]`);
      if (!existingRow && !currentRowElements[index]) {
        addCompletedRow(row, index);
      }
    });
  }
}

// Utility function to get cell value safely
function getCellValue(rowData, column) {
  const value = rowData[column];
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

// Export functions for use in other modules
window.tableUtils = {
  makeColumnsResizable,
  addCompletedRow,
  clearTable,
  updateTable,
  getCellValue,
};
