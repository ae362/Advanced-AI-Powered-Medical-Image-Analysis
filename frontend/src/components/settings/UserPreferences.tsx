import React from 'react';
import { Form, Select, Switch } from 'antd';

const { Option } = Select;

const UserPreferences: React.FC = () => {
  return (
    <Form layout="vertical">
      <Form.Item name="language" label="Language">
        <Select defaultValue="en">
          <Option value="en">English</Option>
          <Option value="es">Spanish</Option>
          <Option value="fr">French</Option>
        </Select>
      </Form.Item>
      <Form.Item name="theme" label="Theme">
        <Select defaultValue="light">
          <Option value="light">Light</Option>
          <Option value="dark">Dark</Option>
        </Select>
      </Form.Item>
      <Form.Item name="notifications" label="Enable Notifications" valuePropName="checked">
        <Switch />
      </Form.Item>
    </Form>
  );
};

export default UserPreferences;

