import React from 'react';
import { Table } from 'antd';
import { ColumnsType } from 'antd/es/table';

interface ReportData {
  key: string;
  patient: string;
  type: string;
  date: string;
  status: string;
}

const columns: ColumnsType<ReportData> = [
  {
    title: 'Patient',
    dataIndex: 'patient',
    key: 'patient',
  },
  {
    title: 'Type',
    dataIndex: 'type',
    key: 'type',
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
];

const data: ReportData[] = [
  {
    key: '1',
    patient: 'John Brown',
    type: 'Brain Tumor Analysis',
    date: '2023-06-01',
    status: 'Completed',
  },
  {
    key: '2',
    patient: 'Jim Green',
    type: 'Lung Cancer Screening',
    date: '2023-06-02',
    status: 'Pending',
  },
  {
    key: '3',
    patient: 'Joe Black',
    type: 'Breast Cancer Analysis',
    date: '2023-06-03',
    status: 'In Progress',
  },
];

const RecentReports: React.FC = () => {
  return <Table columns={columns} dataSource={data} />;
};

export default RecentReports;

