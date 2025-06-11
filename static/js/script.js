// static/js/script.js
document.addEventListener("DOMContentLoaded", function () {
  // File input styling and validation
  const fileInput = document.getElementById("file");
  if (fileInput) {
    fileInput.addEventListener("change", function (e) {
      const fileName = e.target.files[0]?.name;
      if (fileName) {
        // Check if file is CSV
        const fileExt = fileName.split(".").pop().toLowerCase();
        if (fileExt !== "csv") {
          alert("Please upload a CSV file only.");
          fileInput.value = "";
          return;
        }
      }
    });
  }

  // Fungsi untuk menampilkan overlay loading
  function showLoading() {
    document.getElementById("loading-overlay").style.display = "flex";
  }

  // Fungsi untuk menyembunyikan overlay loading
  function hideLoading() {
    document.getElementById("loading-overlay").style.display = "none";
  }

  // Attach ke form submission
  // const button = document.querySelector("button[type='submit']");
  // button.addEventListener("click", function () {
  //   showLoading();
  // });
  const forms = document.querySelectorAll("form");
  forms.forEach((form) => {
    form.addEventListener("submit", function () {
      const button = document.querySelector("button[type='submit']");
      button.innerHTML = "Loading...";
      button.disabled = true;
      showLoading();
    });
  });
});
