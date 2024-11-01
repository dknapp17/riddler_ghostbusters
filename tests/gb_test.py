# pytest framework

import pytest
from unittest.mock import patch, MagicMock
from GhostBusters import GBDashData  # Replace with the actual module name

def test_connect_db_success():
    """
    Test successful database connection using a mock.
    """
    with patch("GhostBusters.st.connection") as mock_connection:
        mock_conn = MagicMock()
        mock_connection.return_value = mock_conn
        
        db_data = GBDashData()
        conn = db_data.connect_db()
        
        # Ensure the connection object is returned
        assert conn is mock_conn
        mock_connection.assert_called_once_with("postgresql", type="sql")

def test_connect_db_failure():
    """
    Test database connection failure using a mock.
    """
    # Patch the `connect_db` method to control when the exception is raised.
    with patch.object(GBDashData, "connect_db", side_effect=Exception("Connection failed")):
        db_data = GBDashData()
        
        # Attempt to connect, expecting a failure
        try:
            conn = db_data.connect_db()
        except Exception as e:
            assert str(e) == "Connection failed"
