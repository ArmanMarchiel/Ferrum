/* The two row icons. They were HTML strings injected with innerHTML before;
   as components they are typed and cannot produce malformed markup. */

export function FolderIcon() {
  return (
    <svg
      className="ico"
      viewBox="0 0 16 16"
      fill="none"
      stroke="#636b7a"
      strokeWidth="1.3"
      strokeLinejoin="round"
    >
      <path d="M1.5 4.2c0-.6.4-1 1-1h3.2l1.4 1.6h6.4c.6 0 1 .4 1 1v6.5c0 .6-.4 1-1 1H2.5c-.6 0-1-.4-1-1z" />
    </svg>
  );
}

export function ProjectIcon() {
  return (
    <svg
      className="ico"
      viewBox="0 0 16 16"
      fill="none"
      stroke="#12325b"
      strokeWidth="1.3"
      strokeLinejoin="round"
    >
      <path d="M8 1.6 14 5v6L8 14.4 2 11V5z" />
      <path d="M2 5l6 3.4L14 5M8 8.4v6" />
    </svg>
  );
}

export const ItemIcon = ({ kind }: { kind: 'folder' | 'project' }) =>
  kind === 'folder' ? <FolderIcon /> : <ProjectIcon />;
