# MTConnect Adapter Simulator
# A simple OPC-UA MTConnect Adapter simulator for testing Process Endpoint Monitoring

The process simulates an mtconnect adapter. It provides an interface to interact with the parameters and supports external communication through a TCP server. The MTConnect Adapter's behavior, represented by variables, can be dynamically updated via OPC UA clients or external systems.

# To activate the OPC-UA MTConnect Adapter Simulator process ensure all Python requirements are installed, and run:

<code>python3 MTConnectAdapter.py</code>

You can use any OPC-UA client to connect to the OPC-UA server (no authentication is required) and start reading legitimate and anomalous process data generated by the simulation.