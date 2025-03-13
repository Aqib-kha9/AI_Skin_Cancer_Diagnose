import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import ImageUploader from "./pages/imageUploader";

const darkTheme = createTheme({
  palette: {
    mode: "dark", // Enables dark mode
    primary: {
      main: "#1db954", // Green accent color
    },
    background: {
      default: "#121212", // Dark background
      paper: "#1e1e1e", // Slightly lighter dark for cards
    },
    text: {
      primary: "#ffffff", // White text
      secondary: "#b3b3b3", // Light gray for subtext
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline /> {/* Ensures consistent dark mode styling */}
      <ImageUploader />
    </ThemeProvider>
  );
}

export default App;
