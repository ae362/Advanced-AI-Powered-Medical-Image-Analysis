import React from 'react';
import { List, Typography } from 'antd';

const { Text } = Typography;

const data = [
  'System update scheduled for tonight at 2 AM.',
  'New AI model for brain tumor detection is now available.',
  'Reminder: Weekly team meeting at 10 AM tomorrow.',
  'Patient John Doe has a new analysis ready for review.',
];

const Notifications: React.FC = () => {
  return (
    <List
      header={<div>System Notifications</div>}
      bordered
      dataSource={data}
      renderItem={(item) => (
        <List.Item>
          <Text>{item}</Text>
        </List.Item>
      )}
    />
  );
};

export default Notifications;

