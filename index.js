const express = require("express");
const { spawn } = require("child_process");
const cors = require("cors");

const app = express();
app.use(cors());

// Route: localhost:6000/scrape?url=your_website&query=your_query
app.get("/scrape", (req, res) => {
  const { url, query } = req.query;
  console.log(`🔄 Scraping started for: ${url} | Query: ${query}`);

  if (!url || !query) {
    return res
      .status(400)
      .json({ error: "Missing 'url' or 'query' parameter" });
  }

  const pythonProcess = spawn("python", ["script.py", url, query]);

  let output = "";
  let errorOutput = "";

  // Capture Python stdout (JSON output)
  pythonProcess.stdout.on("data", (data) => {
    output += data.toString();
  });

  // Capture Python stderr (logs)
  pythonProcess.stderr.on("data", (data) => {
    console.error(data.toString());
  });

  pythonProcess.on("close", (code) => {
    try {
      const result = JSON.parse(output);
      console.log("Scraping completed successfully!");
      res.status(200).json({ success: true, data: result });
    } catch (error) {
      console.error("JSON Parsing Error:", error);
      res
        .status(500)
        .json({ success: false, error: "Invalid response from Python script" });
    }
  });

  pythonProcess.on("error", (err) => {
    console.error(`❌ Failed to start Python process: ${err}`);
    res.status(500).json({
      success: false,
      error: "Python script execution failed",
      details: err.message,
    });
  });
});

// Start the server
app.listen(5001, () => {
  console.log("🚀 Node.js API running on port 5001");
});
