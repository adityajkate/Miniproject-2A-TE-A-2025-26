import React, { useEffect } from 'react';
import './Notification.css';

const Notification = ({ message, onClose }) => {
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => {
        onClose();
      }, 5000); // Auto-dismiss after 5 seconds

      return () => clearTimeout(timer);
    }
  }, [message, onClose]);

  if (!message) {
    return null;
  }

  return (
    <div className="notification-toast">
      <div className="notification-message">{message}</div>
      <button className="notification-close-btn" onClick={onClose}>
        &times;
      </button>
    </div>
  );
};

export default Notification;
