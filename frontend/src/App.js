import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Citizen from "./pages/Citizen";
import Responder from "./pages/Responder";

function App() {
  return (
    <Router>
      <div style={{ padding: 20 }}>
        <h1>🚨 SafeLink POC</h1>
        <nav style={{ marginBottom: 20 }}>
          <Link to="/citizen" style={{ marginRight: 10 }}>Citizen</Link>
          <Link to="/responder">Responder</Link>
        </nav>
        <Routes>
          <Route path="/citizen" element={<Citizen />} />
          <Route path="/responder" element={<Responder />} />
          <Route path="/" element={<Citizen />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
