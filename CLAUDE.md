## General
- Always note accepted edits in CHANGELOG.md
- Always use double quotes unless already inside a quoted string
- Always use pathlib rather than os.path as long as possible
- Always import logger from .logging and frequently log as INFO level internal actions.
## Documentation - docstrings
- Always use Google style docstrings formatted for markdown (not restructuredtext)
- Always include information on exceptions explicitly raised in the code
- Consider executable code examples in docstrings when appropriate and concise
## Testing
- Always write tests in a "given-when-then" style: given this scenario, when I call this function, then I expect this result
- Always add a docstring to tests explaining the purpose of the test and the implications of failure(s)
- Always comment on inconsistencies or ambiguities in functions being tested
- Always run new tests to verify implementation before considering implementation as complete
-
