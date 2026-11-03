import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import DataCollectionPage from './components/DataCollectionPage.tsx';
import './index.css';

const isCollectionRoute = window.location.pathname === '/collect';
createRoot(document.getElementById('root')!).render(
  isCollectionRoute ? <DataCollectionPage /> : <App />
);
