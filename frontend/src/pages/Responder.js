import React, { useState, useEffect } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function Responder() {
  const [incidents, setIncidents] = useState([]);

  const fetchIncidents = async () => {
    const res = await fetch(`${API_URL}/incidents`);
    const data = await res.json();
    setIncidents(data);
  };

  useEffect(() => {
    fetchIncidents();
    const interval = setInterval(fetchIncidents, 5000); // refresh every 5s
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h2>Responder Dashboard</h2>
      {incidents.length === 0 ? (
        <p>No incidents reported yet.</p>
      ) : (
        incidents.slice().reverse().map((inc, idx) => (
          <div key={idx} style={{ border: "1px solid #ddd", padding: 10, marginBottom: 10 }}>
            <p><b>Severity:</b> {inc.severity}</p>
            <p><b>Summary:</b> {inc.responder_summary}</p>
            <p><b>Citizen Guidance:</b> {inc.citizen_guidance}</p>
          </div>
        ))
      )}
    </div>
  );
}
