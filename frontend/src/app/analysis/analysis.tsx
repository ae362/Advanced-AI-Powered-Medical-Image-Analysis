import React, { useState } from 'react';
import MainLayout from '../components/layout/MainLayout';
import ImageUpload from '../components/analysis/ImageUpload';
import ImageViewer from '../components/analysis/ImageViewer';
import AnalysisResults from '../components/analysis/AnalysisResults';
import { Typography, Button, Space } from 'antd';

const { Title } = Typography;

const AnalysisPage: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any | null>(null);

  const handleImageSelect = (imageUrl: string) => {
    setSelectedImage(imageUrl);
  };

  const handleRunAnalysis = () => {
    // TODO: Implement actual analysis logic
    setAnalysisResults({
      prediction: 'Anomaly Detected',
      confidence: 0.85,
      anomalies: ['Irregular growth in left hemisphere', 'Unusual tissue density in frontal lobe'],
    });
  };

  return (
    <MainLayout>
      <Title level={2}>Image Analysis</Title>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <ImageUpload />
        {selectedImage && (
          <>
            <ImageViewer src={selectedImage} />
            <Button type="primary" onClick={handleRunAnalysis}>
              Run Analysis
            </Button>
          </>
        )}
        {analysisResults && <AnalysisResults {...analysisResults} />}
      </Space>
    </MainLayout>
  );
};

export default AnalysisPage;

