document
  .getElementById("upload-form")
  .addEventListener("submit", async function (e) {
    e.preventDefault(); // Prevent the default form submission

    const formData = new FormData();
    formData.append("document", document.getElementById("document").files[0]);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      document.getElementById("response").innerText =
        data.message || data.error;
    } catch (error) {
      document.getElementById("response").innerText =
        "An error occurred: " + error;
    }
  });
