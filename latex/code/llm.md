Extract any explicit coordinates from the messages you receive.

- **Only** extract coordinates if they are explicitly mentioned in the message (e.g., numerical latitude and longitude).
- **Do not** infer or look up coordinates based on place names, city names, or any other information.
- **If multiple pairs of coordinates are present, extract only the first pair mentioned.**
- If there are no explicit coordinates in the message, reply "nan".

Answer in JSON format only: {"latitude": <e.g., "14.832185">, "longitude": <e.g., "4.212666">}. No explanation, just the answer!
