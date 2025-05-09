import React from 'react';
import { Form, Input, Button, Table } from 'antd';

const AdminSettings: React.FC = () => {
  const columns = [
    {
      title: 'Username',
      dataIndex: 'username',
      key: 'username',
    },
    {
      title: 'Role',
      dataIndex: 'role',
      key: 'role',
    },
    {
      title: 'Actions',
      key: 'actions',
      render: () => (
        <Button type="link">Edit</Button>
      ),
    },
  ];

  const data = [
    {
      key: '1',
      username: 'john_doe',
      role: 'Admin',
    },
    {
      key: '2',
      username: 'jane_smith',
      role: 'Doctor',
    },
  ];

  return (
    <div>
      <h3>User Management</h3>
      <Table columns={columns} dataSource={data} />
      <h3>System Logs</h3>
      <Button>View Logs</Button>
    </div>
  );
};

export default AdminSettings;

