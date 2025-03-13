import React, { useState } from "react";
import axios from "axios";
import { Box, Button, Typography, CircularProgress, Alert, Card, CardContent } from "@mui/material";

const ImageUploader = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setLoading(true);
      setError(null);
      const response = await axios.post("http://localhost:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(response)
      setPrediction(response.data);
    } catch (err) {
      setError("Failed to get prediction. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ textAlign: "center", padding: 4, minHeight: "100vh", backgroundColor: "#121212", color: "#fff" }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: "bold", color: "#1db954" }}>
        🔍 AI Skin Disease Diagnosis
      </Typography>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{
          marginBottom: "20px",
          padding: "10px",
          backgroundColor: "#1db954",
          color: "#fff",
          borderRadius: "5px",
          border: "none",
        }}
      />

      {preview && (
        <Box sx={{ display: "flex", justifyContent: "center", marginBottom: 2 }}>
          <img
            src={preview}
            alt="Selected Preview"
            style={{
              width: 200,
              height: 200,
              borderRadius: 10,
              border: "3px solid #1db954",
            }}
          />
        </Box>
      )}

      <Button
        variant="contained"
        onClick={handleUpload}
        disabled={loading}
        sx={{
          marginBottom: 2,
          backgroundColor: "#1db954",
          "&:hover": { backgroundColor: "#17a34a" },
        }}
      >
        {loading ? <CircularProgress size={24} sx={{ color: "#fff" }} /> : "🔍 Predict"}
      </Button>

      {error && <Alert severity="error" sx={{ margin: "auto", maxWidth: 400 }}>{error}</Alert>}

      {prediction && (
        <Card sx={{ maxWidth: 400, margin: "auto", marginTop: 2, padding: 2, backgroundColor: "#222", color: "#fff" }}>
          <CardContent>
            <Typography variant="h6" sx={{ color: "#1db954" }}>🔬 Prediction Result</Typography>
            <Typography variant="body1" sx={{ marginTop: 1 }}>
              <strong>🩺 Condition:</strong> {prediction.disease}
            </Typography>
            <Typography variant="body1">
              <strong>📊 Confidence:</strong> {prediction.confidence}
            </Typography>
            <Typography variant="body1" >
            <strong>Risk Level:</strong> {prediction.risk_level}
            </Typography>
            <Typography variant="body1" sx={{ marginTop: 2, fontStyle: "italic", color: "#bbb" }}>
              {prediction.explanation || "No additional information available."}
            </Typography>
            <Typography variant="body1">
              <strong></strong> {prediction.recommendation}
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default ImageUploader;