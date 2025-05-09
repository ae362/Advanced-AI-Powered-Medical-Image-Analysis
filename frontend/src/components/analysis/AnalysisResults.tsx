import React from 'react';
import { Card, Typography, Progress } from 'antd';

const { Title, Text } = Typography;

interface AnalysisResultsProps {
  prediction: string;
  confidence: number;
  anomalies: string[];
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ prediction, confidence, anomalies }) => {
  return (
    <Card title="AI Analysis Results">
      <Title level={4}>Prediction: {prediction}</Title>
      <Text>Confidence:</Text>
      <Progress percent={confidence * 100} status="active" />
      <Title level={4}>Detected Anomalies:</Title>
      <ul>
        {anomalies.map((anomaly, index) => (
          <li key={index}>{anomaly}</li>
        ))}
      </ul>
    </Card>
  );
};

export default AnalysisResults;

