import streamlit as st

# Initialize connection.
conn = st.connection("postgresql", type="sql")

# Perform query.
df = conn.query('SELECT * FROM riddler.ghostbusters limit 10;', ttl="10m")

# Print results.
for row in df.itertuples():
    st.write(f"{row.game_id} has a :{row.game_result}:")