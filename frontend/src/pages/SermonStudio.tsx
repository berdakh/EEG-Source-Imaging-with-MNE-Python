import WorkflowRunner from "../components/WorkflowRunner";

export default function SermonStudio() {
  return (
    <WorkflowRunner
      workflowId="sermon_studio"
      title="Sermon Studio"
      description="Paste the sermon transcript. Audio upload and local transcription arrive in week two."
      inputName="transcript"
      inputLabel="Sermon transcript"
      placeholder="Paste the transcript here..."
    />
  );
}
