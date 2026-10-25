import WorkflowRunner from "../components/WorkflowRunner";

export default function Announcements() {
  return (
    <WorkflowRunner
      workflowId="announcements"
      title="Announcement Hub"
      description="Paste the week's announcements in any shape. Get bulletin, newsletter, email, social and slide versions."
      inputName="raw"
      inputLabel="Raw announcements"
      placeholder="fall festival sat 18th 4-7pm back lawn, need 6 trunks..."
    />
  );
}
