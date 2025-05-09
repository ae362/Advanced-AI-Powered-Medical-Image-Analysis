import React from 'react';
import { Table, Space, Button } from 'antd';
import { EyeOutlined, EditOutlined, PrinterOutlined, DeleteOutlined } from '@ant-design/icons';

interface Report {
  id: string;
  patientName: string;
  analysisType: string;
  date: string;
  status: string;
}

const columns = [
  {
    title: 'Patient Name',
    dataIndex: 'patientName',
    key: 'patientName',
  },
  {
    title: 'Analysis Type',
    dataIndex: 'analysisType',
    key: 'analysisType',
  },
  {
    title: 'Date',
    dataIndex: 'date',
    key: 'date',
  },
  {
    title: 'Status',
    dataIndex: 'status',
    key: 'status',
  },
  {
    title: 'Actions',
    key: 'actions',
    render: (text: string, record: Report) => (
      <Space size="middle">
        <Button icon={<EyeOutlined />} />
        <Button icon={<EditOutlined />} />
        <Button icon={<PrinterOutlined />} />
        <Button icon={<DeleteOutlined />} danger />
      </Space>
    ),
  },
];

const data: Report[] = [
  {
    id: '1',
    patientName: 'John Brown',
    analysisType: 'Brain Tumor Detection',
    date: '2023-06-01',
    status: 'Completed',
  },
  {
    id: '2',
    patientName: 'Jim Green',
    analysisType: 'Lung Cancer Screening',
    date: '2023-06-02',
    status: 'Pending Review',
  },
  {
    id: '3',
    patientName: 'Joe Black',
    analysisType: 'Breast Cancer Analysis',
    date: '2023-06-03',
    status: 'In Progress',
  },
];

const ReportList: React.FC = () => {
  return <Table columns={columns} dataSource={data} />;
};

export default ReportList;

