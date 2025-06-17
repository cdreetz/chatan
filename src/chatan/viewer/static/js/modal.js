// Modal functionality for editing column schemas

// Open the edit modal for a specific column
function openEditModal(column) {
  const modal = document.getElementById("editModal");
  const columnNameSpan = document.getElementById("editColumnName");
  const editForm = document.getElementById("editForm");

  columnNameSpan.textContent = column;

  if (schemaData && schemaData[column]) {
    const meta = schemaData[column];
    editForm.innerHTML = "";

    if (meta.type === "generator") {
      editForm.innerHTML = `
                <div class="form-group">
                    <label class="form-label">Provider</label>
                    <input type="text" class="form-input" value="${meta.provider}" readonly>
                </div>
                <div class="form-group">
                    <label class="form-label">Prompt Template</label>
                    <textarea class="form-textarea" id="editPrompt">${meta.prompt}</textarea>
                </div>
                <div class="form-group">
                    <label class="form-label">Variables</label>
                    <input type="text" class="form-input" value="${meta.variables.join(", ")}" readonly>
                </div>
            `;
    } else if (meta.type === "sample" || meta.type === "weighted_sample") {
      editForm.innerHTML = `
                <div class="form-group">
                    <label class="form-label">Type</label>
                    <input type="text" class="form-input" value="${meta.type}" readonly>
                </div>
                <div class="form-group">
                    <label class="form-label">Choices (JSON format)</label>
                    <textarea class="form-textarea" id="editChoices">${JSON.stringify(meta.choices, null, 2)}</textarea>
                </div>
            `;
    } else {
      editForm.innerHTML = `
                <div class="form-group">
                    <label class="form-label">Type</label>
                    <input type="text" class="form-input" value="${meta.type}" readonly>
                </div>
                <div class="form-group">
                    <label class="form-label">Value</label>
                    <input type="text" class="form-input" value="${JSON.stringify(meta.value || meta.name)}" readonly>
                </div>
            `;
    }
  }

  modal.style.display = "block";
}

// Close the edit modal
function closeEditModal() {
  document.getElementById("editModal").style.display = "none";
}

// Save changes and trigger regeneration
async function saveChanges() {
  const column = document.getElementById("editColumnName").textContent;
  const meta = schemaData[column];

  if (meta.type === "generator") {
    const newPrompt = document.getElementById("editPrompt").value;
    meta.prompt = newPrompt;
  } else if (meta.type === "sample" || meta.type === "weighted_sample") {
    const newChoices = document.getElementById("editChoices").value;
    try {
      meta.choices = JSON.parse(newChoices);
    } catch (error) {
      alert("Invalid JSON format for choices");
      return;
    }
  }

  // Update schema on server
  try {
    const response = await fetch("/api/update_schema", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(schemaData),
    });

    if (response.ok) {
      alert("Schema updated! Regeneration would happen here.");
      closeEditModal();
      setupColumnHeaders(); // Refresh headers
    } else {
      throw new Error("Server responded with error");
    }
  } catch (error) {
    console.error("Error updating schema:", error);
    alert("Error updating schema");
  }
}

// Event listeners
document.addEventListener("DOMContentLoaded", function () {
  // Close modal when clicking outside
  window.onclick = function (event) {
    const modal = document.getElementById("editModal");
    if (event.target === modal) {
      closeEditModal();
    }
  };

  // Close modal when clicking X
  document.querySelector(".close").onclick = closeEditModal;
});
