import React from 'react';
import { Typography, Descriptions, Image, Space, Button } from 'antd';
import { EditOutlined, PrinterOutlined, DownloadOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

interface ReportViewerProps {
  report: {
    id: string;
    patientName: string;
    analysisType: string;
    date: string;
    status: string;
    results: string;
    imageUrl: string;
  };
}

const ReportViewer: React.FC<ReportViewerProps> = ({ report }) => {
  return (
    <div>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Title level={3}>Report: {report.id}</Title>
          <Space>
            <Button icon={<EditOutlined />}>Edit</Button>
            <Button icon={<PrinterOutlined />}>Print</Button>
            <Button icon={<DownloadOutlined />}>Export PDF</Button>
          </Space>
        </div>
        <Descriptions bordered>
          <Descriptions.Item label="Patient Name">{report.patientName}</Descriptions.Item>
          <Descriptions.Item label="Analysis Type">{report.analysisType}</Descriptions.Item>
          <Descriptions.Item label="Date">{report.date}</Descriptions.Item>
          <Descriptions.Item label="Status">{report.status}</Descriptions.Item>
        </Descriptions>
        <Title level={4}>Analysis Results</Title>
        <Paragraph>{report.results}</Paragraph>
        <Title level={4}>Image</Title>
        <Image src={report.imageUrl} alt="Analysis Image" width={400} />
      </Space>
    </div>
  );
};

export default ReportViewer;

