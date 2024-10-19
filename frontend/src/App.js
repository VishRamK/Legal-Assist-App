import React from "react";
import DocumentUpload from "./components/DocumentUpload";
import Trial from "./components/Trial";

const App = () => {
  return (
    <div>
      <h1>Mock Trial Simulation</h1>
      <DocumentUpload />
      <Trial />
    </div>
  );
};

export default App;
