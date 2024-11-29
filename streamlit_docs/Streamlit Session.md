The latest update regarding Streamlit Session State introduces a new configuration option called `runner.enforceSerializableSessionState`. This feature allows developers to restrict Session State to only store pickle-serializable objects[1][3].

## Key Points of the Update

- **New Configuration Option**: The `runner.enforceSerializableSessionState` option can be set to enforce serializable objects in Session State[1][3].

- **Purpose**: This update is designed to help detect incompatibility during development or prepare for future execution environments that may require serializing all data in Session State[1][3].

- **Implementation**: Developers can enable this option by creating a global or project config file or using it as a command-line flag[3].

## How It Works

When the `runner.enforceSerializableSessionState` option is set to `true`:

1. Only pickle-serializable objects are allowed in Session State[3].
2. Adding unserializable data to Session State will result in an exception[3].
3. The system checks serializability by attempting to call `pickle.dumps(obj)` on the object[3].

## Example Configuration

To enable this feature, add the following to your `.streamlit/config.toml` file:

```toml
[runner]
enforceSerializableSessionState = true
```

This update provides developers with more control over data persistence in Streamlit applications, helping to ensure compatibility with various execution environments and improving overall app stability.

Here's an example of how to use Streamlit Session State based on the latest version of Streamlit:

```python
import streamlit as st

# Initialize the counter in Session State if it doesn't exist
if 'counter' not in st.session_state:
    st.session_state.counter = 0

def increment_counter():
    st.session_state.counter += 1

def decrement_counter():
    st.session_state.counter -= 1

st.title('Streamlit Session State Counter Example')

# Display the current count
st.write(f"Current count: {st.session_state.counter}")

# Create buttons to increment and decrement the counter
col1, col2 = st.columns(2)
with col1:
    st.button("Increment", on_click=increment_counter)
with col2:
    st.button("Decrement", on_click=decrement_counter)

# Display the session state
st.write("Session State contents:")
st.write(st.session_state)
```

This example demonstrates several key features of Streamlit's Session State:

1. **Initialization**: We check if the 'counter' key exists in the session state and initialize it if it doesn't[1][4].

2. **State Persistence**: The counter value persists across reruns of the app, allowing us to maintain state[1][4].

3. **Callback Functions**: We define `increment_counter()` and `decrement_counter()` functions to update the state[3][4].

4. **Widget Integration**: The buttons use the `on_click` parameter to trigger the callback functions when clicked[3][4].

5. **State Access**: We can easily access and display the current state using `st.session_state.counter`[1][4].

6. **State Inspection**: At the end, we display the entire contents of the session state, which can be useful for debugging[4].

This example showcases how Session State can be used to create interactive apps with persistent data across reruns. It's particularly useful for scenarios like counters, data annotation, pagination, or any situation where you need to maintain state between user interactions[2].

