import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div>
      <h2>What do you want to do?</h2>
      <div className="grid">
        <Link className="tile" to="/sermon"><strong>Sermon Studio</strong><span>Recording in, summary, guide, posts out.</span></Link>
        <Link className="tile" to="/announcements"><strong>Announcements</strong><span>Messy notes in, every channel out.</span></Link>
        <Link className="tile" to="/ask"><strong>Ask the Office</strong><span>Answers from our own documents, with sources.</span></Link>
        <Link className="tile" to="/letters"><strong>Letters</strong><span>Thank-you, follow-up, benevolence, on letterhead.</span></Link>
      </div>
    </div>
  );
}
