import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { Home } from './pages/Home';
import { Editor } from './pages/Editor';

/* The two screens the product has: the filesystem at `/`, and the editor at
   `/editor`. `?project=<id>` on the editor reopens a saved project -- the
   editor reads it from the query string, as it did before the port. */
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/editor" element={<Editor />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
