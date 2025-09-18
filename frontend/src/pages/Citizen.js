import React, { useState } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function Citizen() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState(null);

  const submitIncident = async () => {
    const res = await fetch(`${API_URL}/incident`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input }),
    });
    const data = await res.json();
    setResponse(data);
    setInput("");
  };

  return (
    <div>
      <h2>Citizen Report</h2>
      <textarea
        placeholder="Describe the incident..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        rows={4}
        cols={50}
      />
      <br />
      <button onClick={submitIncident}>Submit Incident</button>

      {response && (
        <div style={{ marginTop: 20 }}>
          <h3>Citizen Guidance</h3>
          <p>{response.citizen_guidance}</p>

          <h3>Responder Summary</h3>
          <p><b>Severity:</b> {response.severity}</p>
          <p>{response.responder_summary}</p>
        </div>
      )}
    </div>
  );
}
