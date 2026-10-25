import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import SermonStudio from "./pages/SermonStudio";
import Announcements from "./pages/Announcements";
import Ask from "./pages/Ask";
import Letters from "./pages/Letters";
import Library from "./pages/Library";
import Settings from "./pages/Settings";

const nav = [
  { to: "/", label: "Home" },
  { to: "/sermon", label: "Sermon Studio" },
  { to: "/announcements", label: "Announcements" },
  { to: "/ask", label: "Ask the Office" },
  { to: "/letters", label: "Letters" },
  { to: "/library", label: "Library" },
  { to: "/settings", label: "Settings" },
];

export default function App() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <h1>Local Office AI</h1>
        <nav>
          {nav.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <p className="muted small">Runs on this computer. Nothing leaves the building.</p>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/sermon" element={<SermonStudio />} />
          <Route path="/announcements" element={<Announcements />} />
          <Route path="/ask" element={<Ask />} />
          <Route path="/letters" element={<Letters />} />
          <Route path="/library" element={<Library />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}
